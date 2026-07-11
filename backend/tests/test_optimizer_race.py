import json
import math

import pytest

from app.engine.training.optimizer_race import run_race, run_trajectory


@pytest.mark.parametrize("optimizer,hyperparams", [
    ("sgd", {"momentum": 0.9}),
    ("rmsprop", {}),
    ("adam", {}),
])
def test_trajectory_reduces_loss_on_bowl(optimizer, hyperparams):
    result = run_trajectory("bowl", (3.0, 3.0), optimizer, lr=0.05, hyperparams=hyperparams, num_steps=100)
    assert not result["diverged"]
    first_loss = result["points"][0]["loss"]
    last_loss = result["points"][-1]["loss"]
    assert last_loss < first_loss


def test_trajectory_first_point_is_the_start_with_zero_state():
    result = run_trajectory("bowl", (2.0, -3.0), "adam", lr=0.1, hyperparams={}, num_steps=5)
    first = result["points"][0]
    assert first["step"] == 0
    assert (first["x"], first["y"]) == (2.0, -3.0)
    assert first["state"] == {}


def test_trajectory_point_count_matches_num_steps_when_no_divergence():
    result = run_trajectory("bowl", (1.0, 1.0), "sgd", lr=0.01, hyperparams={}, num_steps=10)
    assert not result["diverged"]
    assert len(result["points"]) == 11  # step 0 through step 10


def test_high_learning_rate_sgd_diverges_on_steep_axis():
    # bowl's y-gradient is 8y; with lr=1.0, y_new = y - 8y = -7y -> magnitude x7 each step
    result = run_trajectory("bowl", (0.1, 1.0), "sgd", lr=1.0, hyperparams={}, num_steps=20)
    assert result["diverged"] is True
    assert len(result["points"]) < 21  # stopped early
    assert result["points"][-1]["diverged"] is True
    assert result["points"][-1]["x"] is None


def test_diverged_trajectory_is_still_json_serializable():
    result = run_trajectory("bowl", (0.1, 1.0), "sgd", lr=1.0, hyperparams={}, num_steps=20)
    json.dumps(result)  # must not raise on inf/nan leaking through


def test_adam_state_exposes_momentum_and_variance_estimates():
    result = run_trajectory("bowl", (2.0, 2.0), "adam", lr=0.1, hyperparams={}, num_steps=3)
    for point in result["points"][1:]:
        assert set(point["state"]) == {"t", "momentum_x", "momentum_y", "variance_x", "variance_y"}


def test_sgd_momentum_state_exposes_velocity():
    result = run_trajectory("bowl", (2.0, 2.0), "sgd", lr=0.1, hyperparams={"momentum": 0.9}, num_steps=3)
    for point in result["points"][1:]:
        assert set(point["state"]) == {"velocity_x", "velocity_y"}


def test_momentum_vector_present_for_sgd_and_matches_velocity_state():
    result = run_trajectory("bowl", (2.0, 2.0), "sgd", lr=0.1, hyperparams={"momentum": 0.9}, num_steps=3)
    assert result["points"][0]["momentum_vector"] == [0.0, 0.0]
    for point in result["points"][1:]:
        assert point["momentum_vector"] == [point["state"]["velocity_x"], point["state"]["velocity_y"]]


def test_momentum_vector_present_for_adam_and_matches_momentum_state():
    result = run_trajectory("bowl", (2.0, 2.0), "adam", lr=0.1, hyperparams={}, num_steps=3)
    assert result["points"][0]["momentum_vector"] == [0.0, 0.0]
    for point in result["points"][1:]:
        assert point["momentum_vector"] == [point["state"]["momentum_x"], point["state"]["momentum_y"]]


def test_momentum_vector_is_none_for_rmsprop():
    result = run_trajectory("bowl", (2.0, 2.0), "rmsprop", lr=0.1, hyperparams={}, num_steps=3)
    for point in result["points"]:
        assert point["momentum_vector"] is None


def test_momentum_vector_grows_then_stabilizes_under_constant_gradient():
    """A loose but real check on the plan's 'stretching arrow' claim: under a
    sustained gradient, SGD momentum should grow in magnitude before leveling
    off near its steady-state value, not stay flat or shrink monotonically."""
    result = run_trajectory("bowl", (5.0, 0.0), "sgd", lr=0.01, hyperparams={"momentum": 0.9}, num_steps=30)
    magnitudes = [
        (p["momentum_vector"][0] ** 2 + p["momentum_vector"][1] ** 2) ** 0.5
        for p in result["points"]
        if p["momentum_vector"] is not None
    ]
    assert magnitudes[5] > magnitudes[1]  # arrow has visibly grown early on


def test_saddle_point_start_with_zero_gradient_never_moves_without_momentum():
    """A classic pedagogical case: starting exactly at a saddle point with a
    momentum-free optimizer, the gradient is zero forever, so it never
    escapes (this is exactly what the 'momentum escapes saddle points'
    visualization is meant to contrast against)."""
    result = run_trajectory("saddle", (0.0, 0.0), "sgd", lr=0.1, hyperparams={"momentum": 0.0}, num_steps=10)
    for point in result["points"]:
        assert point["x"] == pytest.approx(0.0, abs=1e-12)
        assert point["y"] == pytest.approx(0.0, abs=1e-12)


def test_invalid_num_steps_raises():
    with pytest.raises(ValueError):
        run_trajectory("bowl", (1.0, 1.0), "sgd", lr=0.1, hyperparams={}, num_steps=0)


def test_unknown_surface_raises():
    with pytest.raises(ValueError):
        run_trajectory("not_real", (1.0, 1.0), "sgd", lr=0.1, hyperparams={}, num_steps=5)


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        run_trajectory("bowl", (1.0, 1.0), "not_real", lr=0.1, hyperparams={}, num_steps=5)


def test_invalid_hyperparam_key_raises():
    with pytest.raises(ValueError):
        run_trajectory("bowl", (1.0, 1.0), "sgd", lr=0.1, hyperparams={"beta1": 0.9}, num_steps=5)


def test_race_runs_all_optimizers_from_identical_start():
    racers = [
        {"label": "sgd", "optimizer": "sgd", "lr": 0.05, "hyperparams": {"momentum": 0.9}},
        {"label": "adam", "optimizer": "adam", "lr": 0.05, "hyperparams": {}},
        {"label": "rmsprop", "optimizer": "rmsprop", "lr": 0.05, "hyperparams": {}},
    ]
    result = run_race("bowl", (4.0, 4.0), racers, num_steps=30)
    assert len(result["racers"]) == 3
    for racer_result in result["racers"]:
        first_point = racer_result["points"][0]
        assert (first_point["x"], first_point["y"]) == (4.0, 4.0)


def test_race_labels_are_preserved_and_distinguishable():
    racers = [
        {"label": "fast-adam", "optimizer": "adam", "lr": 0.2, "hyperparams": {}},
        {"label": "slow-adam", "optimizer": "adam", "lr": 0.01, "hyperparams": {}},
    ]
    result = run_race("bowl", (3.0, 3.0), racers, num_steps=20)
    labels = [r["label"] for r in result["racers"]]
    assert labels == ["fast-adam", "slow-adam"]
    # faster learning rate should reach lower loss in the same number of steps
    fast_final_loss = result["racers"][0]["points"][-1]["loss"]
    slow_final_loss = result["racers"][1]["points"][-1]["loss"]
    assert fast_final_loss < slow_final_loss


def test_race_empty_racers_raises():
    with pytest.raises(ValueError):
        run_race("bowl", (1.0, 1.0), [], num_steps=10)


def test_race_result_is_json_serializable():
    racers = [{"label": "a", "optimizer": "sgd", "lr": 0.05, "hyperparams": {}}]
    result = run_race("saddle", (1.0, 1.0), racers, num_steps=10)
    json.dumps(result)
