import copy
import json
import math

import pytest
import torch

from app.engine.ann.model import build_network
from app.engine.ann.state_tree import build_backward_state_tree
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


def test_backward_tree_shape_and_json_safe():
    net = build_network(_spec())
    tree = build_backward_state_tree(net, [1.0, -2.0, 0.5], [0.0, 1.0], "mse")
    json.dumps(tree)  # must not raise

    assert "loss" in tree and isinstance(tree["loss"], float)
    l0, l1 = tree["layers"]
    assert len(l0["grad_weights"]) == 4 and len(l0["grad_weights"][0]) == 3
    assert len(l0["grad_biases"]) == 4
    assert len(l0["grad_pre_activation"]) == 4
    assert len(l0["grad_post_activation"]) == 4
    assert l0["grad_input"] is not None and len(l0["grad_input"]) == 3
    assert len(l1["grad_weights"]) == 2 and len(l1["grad_weights"][0]) == 4


def test_loss_value_matches_manual_mse():
    net = build_network(_spec())
    x, target = [1.0, -2.0, 0.5], [0.2, 0.8]
    tree = build_backward_state_tree(net, x, target, "mse")

    with torch.no_grad():
        y = net(torch.tensor([x])).squeeze(0)
    expected = torch.mean((y - torch.tensor(target)) ** 2).item()
    assert tree["loss"] == pytest.approx(expected, abs=1e-5)


def test_relu_zero_gradient_when_neuron_dead():
    net = build_network(_spec())
    x = [1.0, -2.0, 0.5]
    tree = build_backward_state_tree(net, x, [0.0, 1.0], "mse")
    l0 = tree["layers"][0]
    for z, gz in zip(l0["pre_activation"], l0["grad_pre_activation"]):
        if z < 0:
            assert gz == 0.0


def test_gradients_match_autograd_ground_truth():
    """Cross-check every captured gradient tensor against net.parameters().grad
    populated by a completely independent, hook-free forward+backward pass.
    """
    net = build_network(_spec())
    x, target = [1.0, -2.0, 0.5], [0.2, 0.8]
    tree = build_backward_state_tree(net, x, target, "mse")

    # Independent reference computation using the SAME weights (no hooks at all).
    ref_net = build_network(_spec())  # same seed => identical init
    ref_net.load_state_dict(net.state_dict())
    ref_net.zero_grad()
    x_t = torch.tensor([x], dtype=torch.float32)
    y = ref_net(x_t)
    loss = torch.mean((y - torch.tensor([target])) ** 2)
    loss.backward()

    for idx, linear in enumerate(ref_net.linears):
        assert torch.allclose(
            torch.tensor(tree["layers"][idx]["grad_weights"]), linear.weight.grad, atol=1e-4
        )
        assert torch.allclose(
            torch.tensor(tree["layers"][idx]["grad_biases"]), linear.bias.grad, atol=1e-4
        )


def test_gradient_matches_finite_difference():
    """The gold-standard sanity check: perturb one weight by epsilon and confirm
    the numerical slope of the loss matches the analytically captured gradient.
    """
    net = build_network(_spec())
    x, target = [1.0, -2.0, 0.5], [0.2, 0.8]
    tree = build_backward_state_tree(net, x, target, "mse")
    analytic_grad = tree["layers"][1]["grad_weights"][0][0]

    eps = 1e-4

    def loss_with_perturbation(delta: float) -> float:
        probe = build_network(_spec())
        probe.load_state_dict(net.state_dict())
        with torch.no_grad():
            probe.linears[1].weight[0, 0] += delta
        with torch.no_grad():
            y = probe(torch.tensor([x]))
            return torch.mean((y - torch.tensor([target])) ** 2).item()

    numeric_grad = (loss_with_perturbation(eps) - loss_with_perturbation(-eps)) / (2 * eps)
    assert analytic_grad == pytest.approx(numeric_grad, abs=1e-2)


def test_bce_loss_runs_and_matches_manual():
    net = build_network(_spec())
    x, target = [1.0, -2.0, 0.5], [0.2, 0.8]
    tree = build_backward_state_tree(net, x, target, "bce")

    with torch.no_grad():
        y = net(torch.tensor([x])).squeeze(0).tolist()
    expected = -sum(
        t * math.log(p) + (1 - t) * math.log(1 - p) for t, p in zip(target, y)
    ) / len(target)
    assert tree["loss"] == pytest.approx(expected, abs=1e-4)


def test_no_hook_handles_leak_after_backward():
    net = build_network(_spec())
    build_backward_state_tree(net, [1.0, -2.0, 0.5], [0.0, 1.0], "mse")
    for linear in net.linears:
        assert len(linear._forward_hooks) == 0
        assert len(linear._backward_hooks) == 0
    for activation in net.activations:
        assert len(activation._forward_hooks) == 0
        assert len(activation._backward_hooks) == 0


def test_wrong_input_length_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        build_backward_state_tree(net, [1.0, 2.0], [0.0, 1.0], "mse")


def test_unknown_loss_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        build_backward_state_tree(net, [1.0, -2.0, 0.5], [0.0, 1.0], "not_a_real_loss")


def test_repeated_calls_do_not_accumulate_gradients():
    """zero_grad must actually reset grads between calls, or repeated stepping
    through the visualizer would silently accumulate stale gradients.
    """
    net = build_network(_spec())
    x, target = [1.0, -2.0, 0.5], [0.2, 0.8]
    tree1 = build_backward_state_tree(net, x, target, "mse")
    tree2 = build_backward_state_tree(net, x, target, "mse")
    assert torch.tensor(tree1["layers"][0]["grad_weights"]) == pytest.approx(
        torch.tensor(tree2["layers"][0]["grad_weights"]).numpy(), abs=1e-6
    )
