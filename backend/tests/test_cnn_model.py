import pytest
import torch

from app.engine.cnn.model import build_cnn
from app.schemas.cnn import CNNSpec, ConvLayerSpec, PoolLayerSpec


def test_single_conv_layer_output_shape():
    spec = CNNSpec(
        input_channels=1,
        input_height=8,
        input_width=8,
        seed=1,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=0)],
    )
    net = build_cnn(spec)
    assert net.output_shape == (2, 6, 6)
    x = torch.randn(1, 1, 8, 8)
    y = net(x)
    assert y.shape == (1, 2, 6, 6)


def test_padding_preserves_spatial_size():
    spec = CNNSpec(
        input_channels=1,
        input_height=8,
        input_width=8,
        seed=1,
        layers=[ConvLayerSpec(out_channels=3, kernel_size=3, stride=1, padding=1)],
    )
    net = build_cnn(spec)
    assert net.output_shape == (3, 8, 8)


def test_stride_2_halves_spatial_size():
    spec = CNNSpec(
        input_channels=1,
        input_height=8,
        input_width=8,
        seed=1,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=2, stride=2, padding=0)],
    )
    net = build_cnn(spec)
    assert net.output_shape == (2, 4, 4)


def test_maxpool_shrinks_by_kernel():
    spec = CNNSpec(
        input_channels=2,
        input_height=8,
        input_width=8,
        seed=1,
        layers=[PoolLayerSpec(pool_type="max", kernel_size=2)],
    )
    net = build_cnn(spec)
    assert net.output_shape == (2, 4, 4)
    x = torch.randn(1, 2, 8, 8)
    y = net(x)
    assert y.shape == (1, 2, 4, 4)


def test_avgpool_runs():
    spec = CNNSpec(
        input_channels=1,
        input_height=6,
        input_width=6,
        seed=1,
        layers=[PoolLayerSpec(pool_type="avg", kernel_size=3, stride=3)],
    )
    net = build_cnn(spec)
    assert net.output_shape == (1, 2, 2)


def test_conv_then_pool_stack():
    spec = CNNSpec(
        input_channels=1,
        input_height=10,
        input_width=10,
        seed=1,
        layers=[
            ConvLayerSpec(out_channels=4, kernel_size=3, stride=1, padding=1),  # -> 4x10x10
            PoolLayerSpec(pool_type="max", kernel_size=2),  # -> 4x5x5
            ConvLayerSpec(out_channels=8, kernel_size=3, stride=1, padding=0),  # -> 8x3x3
        ],
    )
    net = build_cnn(spec)
    assert net.output_shape == (8, 3, 3)
    x = torch.randn(1, 1, 10, 10)
    y = net(x)
    assert y.shape == (1, 8, 3, 3)


def test_classifier_head_produces_logits():
    spec = CNNSpec(
        input_channels=1,
        input_height=8,
        input_width=8,
        seed=1,
        num_classes=5,
        layers=[
            ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=0),
            PoolLayerSpec(pool_type="max", kernel_size=2),
        ],
    )
    net = build_cnn(spec)
    x = torch.randn(1, 1, 8, 8)
    y = net(x)
    assert y.shape == (1, 5)


def test_no_classifier_head_when_num_classes_absent():
    spec = CNNSpec(
        input_channels=1,
        input_height=8,
        input_width=8,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3)],
    )
    net = build_cnn(spec)
    assert net.classifier is None


def test_kernel_larger_than_input_raises():
    spec = CNNSpec(
        input_channels=1,
        input_height=4,
        input_width=4,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=5, padding=0)],
    )
    with pytest.raises(ValueError):
        build_cnn(spec)


def test_pool_kernel_larger_than_input_raises():
    spec = CNNSpec(
        input_channels=1,
        input_height=3,
        input_width=3,
        layers=[PoolLayerSpec(pool_type="max", kernel_size=5)],
    )
    with pytest.raises(ValueError):
        build_cnn(spec)


def test_stack_collapsing_to_nonpositive_size_raises():
    spec = CNNSpec(
        input_channels=1,
        input_height=6,
        input_width=6,
        layers=[
            PoolLayerSpec(pool_type="max", kernel_size=3),  # -> 2x2
            PoolLayerSpec(pool_type="max", kernel_size=3),  # would collapse
        ],
    )
    with pytest.raises(ValueError):
        build_cnn(spec)


def test_seed_reproducibility():
    spec = CNNSpec(
        input_channels=1, input_height=6, input_width=6, seed=99,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3)],
    )
    net_a = build_cnn(spec)
    net_b = build_cnn(spec)
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(pa, pb)


def test_manual_conv_matches_module_output():
    spec = CNNSpec(
        input_channels=1, input_height=5, input_width=5, seed=7,
        layers=[ConvLayerSpec(out_channels=1, kernel_size=3, stride=1, padding=0, activation="linear")],
    )
    net = build_cnn(spec)
    x = torch.randn(1, 1, 5, 5)
    y = net(x)

    conv = net.stages[0]["conv"]
    W, b = conv.weight, conv.bias  # (1,1,3,3), (1,)

    manual = torch.zeros(1, 1, 3, 3)
    for i in range(3):
        for j in range(3):
            patch = x[0, 0, i:i + 3, j:j + 3]
            manual[0, 0, i, j] = torch.sum(patch * W[0, 0]) + b[0]

    assert torch.allclose(y, manual, atol=1e-5)


def test_discriminated_union_rejects_unknown_kind():
    with pytest.raises(Exception):
        CNNSpec(
            input_channels=1, input_height=6, input_width=6,
            layers=[{"kind": "not_real", "out_channels": 2, "kernel_size": 3}],
        )
