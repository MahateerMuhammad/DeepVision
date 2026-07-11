import json

import pytest
import torch

from app.engine.ann.model import build_network
from app.engine.ann.state_tree import build_forward_state_tree
from app.schemas.network import LayerSpec, NetworkSpec


def _spec():
    return NetworkSpec(
        input_size=3,
        seed=42,
        layers=[
            LayerSpec(out_features=4, activation="relu"),
            LayerSpec(out_features=2, activation="sigmoid"),
        ],
    )


def test_state_tree_shape():
    net = build_network(_spec())
    x = [1.0, -2.0, 0.5]
    tree = build_forward_state_tree(net, x)

    assert tree["input"] == x
    assert len(tree["layers"]) == 2
    assert len(tree["output"]) == 2

    l0 = tree["layers"][0]
    assert l0["in_features"] == 3
    assert l0["out_features"] == 4
    assert len(l0["weights"]) == 4 and len(l0["weights"][0]) == 3
    assert len(l0["biases"]) == 4
    assert len(l0["pre_activation"]) == 4
    assert len(l0["post_activation"]) == 4
    assert len(l0["equations"]) == 4


def test_state_tree_is_json_serializable():
    net = build_network(_spec())
    tree = build_forward_state_tree(net, [1.0, -2.0, 0.5])
    # must not raise
    json.dumps(tree)


def test_state_tree_matches_direct_forward():
    net = build_network(_spec())
    x = [1.0, -2.0, 0.5]
    tree = build_forward_state_tree(net, x)

    with torch.no_grad():
        y = net(torch.tensor([x])).squeeze(0).numpy()

    assert torch.allclose(torch.tensor(tree["output"]), torch.tensor(y.tolist()), atol=1e-5)


def test_pre_activation_matches_manual_matmul():
    net = build_network(_spec())
    x = [1.0, -2.0, 0.5]
    tree = build_forward_state_tree(net, x)

    l0 = tree["layers"][0]
    W = torch.tensor(l0["weights"])
    b = torch.tensor(l0["biases"])
    manual_z = (W @ torch.tensor(x) + b).tolist()

    assert torch.allclose(torch.tensor(l0["pre_activation"]), torch.tensor(manual_z), atol=1e-4)


def test_relu_post_activation_correct():
    net = build_network(_spec())
    tree = build_forward_state_tree(net, [1.0, -2.0, 0.5])
    l0 = tree["layers"][0]
    expected = [max(0.0, z) for z in l0["pre_activation"]]
    assert l0["post_activation"] == pytest.approx(expected, abs=1e-6)


def test_sigmoid_post_activation_correct():
    net = build_network(_spec())
    tree = build_forward_state_tree(net, [1.0, -2.0, 0.5])
    l1 = tree["layers"][1]
    import math

    expected = [1 / (1 + math.exp(-z)) for z in l1["pre_activation"]]
    assert l1["post_activation"] == pytest.approx(expected, abs=1e-6)


def test_layer_chaining_input_equals_previous_output():
    net = build_network(_spec())
    tree = build_forward_state_tree(net, [1.0, -2.0, 0.5])
    assert tree["layers"][1]["input"] == tree["layers"][0]["post_activation"]


def test_equation_numbers_match_tensor_values():
    net = build_network(_spec())
    tree = build_forward_state_tree(net, [1.0, -2.0, 0.5])
    l0 = tree["layers"][0]
    for eq, z, a in zip(l0["equations"], l0["pre_activation"], l0["post_activation"]):
        assert f"= {z:.4f}" in eq["linear_equation"]
        assert f"= {a:.4f}" in eq["activation_equation"]


def test_wrong_input_length_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        build_forward_state_tree(net, [1.0, 2.0])


def test_no_hook_handles_leak_after_multiple_calls():
    net = build_network(_spec())
    for _ in range(5):
        build_forward_state_tree(net, [1.0, -2.0, 0.5])
    for linear in net.linears:
        assert len(linear._forward_hooks) == 0
    for activation in net.activations:
        assert len(activation._forward_hooks) == 0
