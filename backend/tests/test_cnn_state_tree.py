import json

import pytest
import torch

from app.engine.cnn.model import build_cnn
from app.engine.cnn.state_tree import build_cnn_forward_state_tree
from app.schemas.cnn import CNNSpec, ConvLayerSpec, PoolLayerSpec


def _image(c, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(c, h, w, generator=g).tolist()


def test_conv_only_state_tree_shape_and_json_safe():
    spec = CNNSpec(
        input_channels=1, input_height=6, input_width=6, seed=1,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=0)],
    )
    net = build_cnn(spec)
    tree = build_cnn_forward_state_tree(net, _image(1, 6, 6))
    json.dumps(tree)

    l0 = tree["layers"][0]
    assert l0["kind"] == "conv"
    assert l0["in_shape"] == [1, 6, 6]
    assert l0["out_shape"] == [2, 4, 4]
    assert len(l0["weights"]) == 2 and len(l0["weights"][0]) == 1
    assert len(l0["weights"][0][0]) == 3 and len(l0["weights"][0][0][0]) == 3
    assert len(l0["pre_activation"]) == 2
    assert len(l0["pre_activation"][0]) == 4 and len(l0["pre_activation"][0][0]) == 4


def test_conv_pre_activation_matches_manual_convolution():
    spec = CNNSpec(
        input_channels=1, input_height=5, input_width=5, seed=3,
        layers=[ConvLayerSpec(out_channels=1, kernel_size=3, stride=1, padding=0, activation="linear")],
    )
    net = build_cnn(spec)
    image = _image(1, 5, 5, seed=3)
    tree = build_cnn_forward_state_tree(net, image)

    l0 = tree["layers"][0]
    W = torch.tensor(l0["weights"])[0, 0]  # (3,3)
    b = torch.tensor(l0["biases"])[0]
    x = torch.tensor(image)[0]  # (5,5)

    manual = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            patch = x[i:i + 3, j:j + 3]
            manual[i, j] = torch.sum(patch * W) + b

    assert torch.allclose(torch.tensor(l0["pre_activation"][0]), manual, atol=1e-4)
    # activation is linear, so post == pre
    assert torch.allclose(torch.tensor(l0["post_activation"]), torch.tensor(l0["pre_activation"]), atol=1e-6)


def test_relu_activation_zeroes_negatives():
    spec = CNNSpec(
        input_channels=1, input_height=5, input_width=5, seed=3,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3, activation="relu")],
    )
    net = build_cnn(spec)
    tree = build_cnn_forward_state_tree(net, _image(1, 5, 5, seed=3))
    l0 = tree["layers"][0]
    for ch_pre, ch_post in zip(l0["pre_activation"], l0["post_activation"]):
        for row_pre, row_post in zip(ch_pre, ch_post):
            for z, a in zip(row_pre, row_post):
                assert a == pytest.approx(max(0.0, z), abs=1e-6)


def test_maxpool_output_matches_manual_and_kept_mask_is_consistent():
    spec = CNNSpec(
        input_channels=1, input_height=4, input_width=4,
        layers=[PoolLayerSpec(pool_type="max", kernel_size=2)],
    )
    net = build_cnn(spec)
    image = [[[1, 5, 2, 0], [3, 4, 8, 1], [0, 1, 9, 2], [6, 2, 1, 3]]]  # 1x4x4
    tree = build_cnn_forward_state_tree(net, image)
    l0 = tree["layers"][0]

    # top-left 2x2 window max = 5, top-right = 8, bottom-left = 6, bottom-right = 9
    assert l0["output"] == [[[5, 8], [6, 9]]]

    mask = l0["kept_mask"][0]  # (4,4)
    # exactly 4 pixels kept (one per output cell), rest dropped
    kept_count = sum(sum(row) for row in mask)
    assert kept_count == 4
    # the argmax positions specifically
    assert mask[0][1] == 1  # value 5 at (0,1)
    assert mask[1][2] == 1  # value 8 at (1,2)
    assert mask[3][0] == 1  # value 6 at (3,0)
    assert mask[2][2] == 1  # value 9 at (2,2)


def test_avgpool_output_matches_manual_average():
    spec = CNNSpec(
        input_channels=1, input_height=4, input_width=4,
        layers=[PoolLayerSpec(pool_type="avg", kernel_size=2)],
    )
    net = build_cnn(spec)
    image = [[[1, 5, 2, 0], [3, 4, 8, 1], [0, 1, 9, 2], [6, 2, 1, 3]]]
    tree = build_cnn_forward_state_tree(net, image)
    l0 = tree["layers"][0]
    expected = [[[(1 + 5 + 3 + 4) / 4, (2 + 0 + 8 + 1) / 4], [(0 + 1 + 6 + 2) / 4, (9 + 2 + 1 + 3) / 4]]]
    assert torch.allclose(torch.tensor(l0["output"]), torch.tensor(expected), atol=1e-5)
    assert "kept_mask" not in l0  # avg pool has no argmax concept


def test_conv_then_pool_layer_input_chaining():
    spec = CNNSpec(
        input_channels=1, input_height=8, input_width=8, seed=2,
        layers=[
            ConvLayerSpec(out_channels=2, kernel_size=3, padding=1),  # -> 2x8x8
            PoolLayerSpec(pool_type="max", kernel_size=2),  # -> 2x4x4
        ],
    )
    net = build_cnn(spec)
    tree = build_cnn_forward_state_tree(net, _image(1, 8, 8, seed=2))
    assert tree["layers"][1]["input"] == tree["layers"][0]["post_activation"]


def test_wrong_channel_count_raises():
    spec = CNNSpec(input_channels=1, input_height=4, input_width=4, layers=[ConvLayerSpec(out_channels=1, kernel_size=2)])
    net = build_cnn(spec)
    with pytest.raises(ValueError):
        build_cnn_forward_state_tree(net, _image(2, 4, 4))


def test_wrong_height_raises():
    spec = CNNSpec(input_channels=1, input_height=4, input_width=4, layers=[ConvLayerSpec(out_channels=1, kernel_size=2)])
    net = build_cnn(spec)
    with pytest.raises(ValueError):
        build_cnn_forward_state_tree(net, _image(1, 5, 4))


def test_wrong_width_raises():
    spec = CNNSpec(input_channels=1, input_height=4, input_width=4, layers=[ConvLayerSpec(out_channels=1, kernel_size=2)])
    net = build_cnn(spec)
    with pytest.raises(ValueError):
        build_cnn_forward_state_tree(net, _image(1, 4, 5))


def test_no_hook_handles_leak():
    spec = CNNSpec(
        input_channels=1, input_height=6, input_width=6,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3), PoolLayerSpec(pool_type="max", kernel_size=2)],
    )
    net = build_cnn(spec)
    for _ in range(3):
        build_cnn_forward_state_tree(net, _image(1, 6, 6))
    for stage in net.stages:
        for module in stage.values():
            assert len(module._forward_hooks) == 0


def test_classifier_head_output_included():
    spec = CNNSpec(
        input_channels=1, input_height=6, input_width=6, num_classes=3,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3), PoolLayerSpec(pool_type="max", kernel_size=2)],
    )
    net = build_cnn(spec)
    tree = build_cnn_forward_state_tree(net, _image(1, 6, 6))
    assert len(tree["output"]) == 3
