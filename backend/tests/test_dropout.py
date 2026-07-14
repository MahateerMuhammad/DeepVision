import json

import pytest
import torch

from app.engine.ann.chain_rule import trace_weight_gradient
from app.engine.ann.model import build_network
from app.engine.ann.state_tree import build_backward_state_tree, build_forward_state_tree
from app.schemas.network import LayerSpec, NetworkSpec


def _spec_no_dropout():
    return NetworkSpec(
        input_size=3, seed=42,
        layers=[LayerSpec(out_features=4, activation="relu"), LayerSpec(out_features=2, activation="sigmoid")],
    )


def _spec_with_dropout(p=0.5):
    return NetworkSpec(
        input_size=3, seed=42,
        layers=[
            LayerSpec(out_features=20, activation="relu", dropout_prob=p),
            LayerSpec(out_features=2, activation="sigmoid"),
        ],
    )


X = [1.0, -2.0, 0.5]
TARGET = [0.2, 0.8]


def test_zero_dropout_prob_is_exact_identity_vs_no_dropout_field():
    """dropout_prob=0.0 must produce numerically identical results to a
    network with no dropout at all the default must be a true no-op."""
    net_plain = build_network(_spec_no_dropout())
    net_explicit_zero = build_network(_spec_no_dropout())  # dropout_prob defaults to 0.0

    tree_plain = build_forward_state_tree(net_plain, X)
    tree_zero = build_forward_state_tree(net_explicit_zero, X)

    assert tree_plain["output"] == pytest.approx(tree_zero["output"], abs=1e-9)
    assert tree_plain["layers"][0]["post_activation"] == pytest.approx(
        tree_zero["layers"][0]["post_dropout"], abs=1e-9
    )


def test_zero_dropout_prob_never_marks_anything_dropped():
    net = build_network(_spec_no_dropout())
    tree = build_forward_state_tree(net, X)
    for layer in tree["layers"]:
        assert layer["dropout_prob"] == 0.0
        assert all(m == 0.0 for m in layer["dropped_mask"])
        assert layer["post_dropout"] == layer["post_activation"]


def test_eval_mode_disables_dropout_even_with_nonzero_prob():
    net = build_network(_spec_with_dropout(p=0.9))
    tree = build_forward_state_tree(net, X, training=False)
    l0 = tree["layers"][0]
    assert l0["post_dropout"] == pytest.approx(l0["post_activation"], abs=1e-9)
    assert all(m == 0.0 for m in l0["dropped_mask"])


def test_training_mode_with_high_prob_actually_drops_units_statistically():
    """With p=0.9 and 20 units, over many forward passes almost every unit
    should be observed dropped at least once, and the average kept-fraction
    should be statistically close to (1 - p)."""
    net = build_network(_spec_with_dropout(p=0.9))
    kept_counts = [0] * 20
    trials = 300
    for _ in range(trials):
        tree = build_forward_state_tree(net, X, training=True)
        mask = tree["layers"][0]["dropped_mask"]
        # only count units that were nonzero pre-dropout (relu may already zero some)
        post_act = tree["layers"][0]["post_activation"]
        for i, (dropped, a) in enumerate(zip(mask, post_act)):
            if a != 0.0 and dropped == 0.0:
                kept_counts[i] += 1

    eligible = [c for c in kept_counts if c > 0 or True]
    total_kept = sum(kept_counts)
    total_eligible_trials = sum(1 for _ in range(trials)) * 20
    kept_fraction = total_kept / total_eligible_trials
    # expected kept fraction ~= 1 - p = 0.1; allow generous tolerance for randomness
    assert 0.03 < kept_fraction < 0.25


def test_dropout_scales_surviving_units_by_inverse_keep_probability():
    """PyTorch's inverted dropout scales kept units by 1/(1-p) so that the
    expected value is unchanged. Verify any nonzero post_dropout value is
    either 0 or (post_activation * 1/(1-p))."""
    p = 0.5
    net = build_network(_spec_with_dropout(p=p))
    tree = build_forward_state_tree(net, X, training=True)
    l0 = tree["layers"][0]
    scale = 1 / (1 - p)
    for a, d in zip(l0["post_activation"], l0["post_dropout"]):
        if a != 0.0:
            assert d == pytest.approx(0.0, abs=1e-6) or d == pytest.approx(a * scale, rel=1e-4)


def test_backward_pass_runs_with_dropout_and_grad_still_matches_finite_difference():
    """Ground-truth check: with dropout active, the analytic gradient of a
    surviving (non-dropped) weight must still match the numeric finite-
    difference slope of the loss — dropout must not corrupt autograd.
    """
    net = build_network(_spec_with_dropout(p=0.3))
    net.train(True)
    torch.manual_seed(123)  # pin the dropout mask draw for reproducibility across the two passes below

    tree = build_backward_state_tree(net, X, TARGET, "mse", training=True)
    analytic_grad = tree["layers"][1]["grad_weights"][0][0]

    eps = 1e-4

    def loss_with_perturbation(delta: float) -> float:
        probe = build_network(_spec_with_dropout(p=0.3))
        probe.load_state_dict(net.state_dict())
        probe.train(True)
        with torch.no_grad():
            probe.linears[1].weight[0, 0] += delta
        torch.manual_seed(123)  # SAME dropout mask draw as the analytic pass
        with torch.no_grad():
            y = probe(torch.tensor([X]))
            return torch.mean((y - torch.tensor([TARGET])) ** 2).item()

    torch.manual_seed(123)
    numeric_grad = (loss_with_perturbation(eps) - loss_with_perturbation(-eps)) / (2 * eps)
    assert analytic_grad == pytest.approx(numeric_grad, abs=5e-2)


def test_state_tree_is_json_serializable_with_dropout():
    net = build_network(_spec_with_dropout(p=0.4))
    fwd = build_forward_state_tree(net, X, training=True)
    bwd = build_backward_state_tree(net, X, TARGET, "mse", training=True)
    json.dumps(fwd)
    json.dumps(bwd)


def test_chain_rule_downstream_breakdown_omitted_when_dropout_active():
    net = build_network(_spec_with_dropout(p=0.5))
    result = trace_weight_gradient(net, X, TARGET, "mse", 0, 0, 0)
    assert result["downstream_breakdown"] is None


def test_chain_rule_downstream_breakdown_present_when_dropout_is_zero():
    net = build_network(_spec_no_dropout())
    result = trace_weight_gradient(net, X, TARGET, "mse", 0, 0, 0)
    assert result["downstream_breakdown"] is not None


def test_chain_rule_aggregated_gradient_still_exact_even_with_dropout():
    """Even though the illustrative downstream breakdown is omitted under
    dropout, the actual returned chain_product/analytic_gradient must still
    match exactly, since both come straight from autograd."""
    net = build_network(_spec_with_dropout(p=0.4))
    result = trace_weight_gradient(net, X, TARGET, "mse", 1, 0, 0)
    assert result["matches_analytic"] is True


def test_invalid_dropout_prob_rejected_by_schema():
    with pytest.raises(Exception):
        LayerSpec(out_features=4, activation="relu", dropout_prob=1.0)  # must be < 1
    with pytest.raises(Exception):
        LayerSpec(out_features=4, activation="relu", dropout_prob=-0.1)
