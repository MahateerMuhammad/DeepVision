import pytest

from app.engine.training.optimizers import (
    OPTIMIZER_HYPERPARAMS,
    adam_step,
    get_optimizer_step,
    rmsprop_step,
    sgd_step,
    validate_hyperparams,
)


def test_sgd_no_momentum_is_plain_gradient_descent():
    pos, state = sgd_step((1.0, 2.0), (0.5, -1.0), {}, lr=0.1, momentum=0.0)
    assert pos == pytest.approx((1.0 - 0.1 * 0.5, 2.0 - 0.1 * -1.0))
    assert state == {"velocity_x": pytest.approx(-0.05), "velocity_y": pytest.approx(0.1)}


def test_sgd_momentum_accumulates_across_two_steps():
    # Step 1: v = 0.9*0 - lr*grad = -lr*grad
    pos1, state1 = sgd_step((0.0, 0.0), (1.0, 1.0), {}, lr=0.1, momentum=0.9)
    assert state1["velocity_x"] == pytest.approx(-0.1)
    assert pos1 == pytest.approx((-0.1, -0.1))

    # Step 2: v = 0.9*(-0.1) - 0.1*1.0 = -0.19
    pos2, state2 = sgd_step(pos1, (1.0, 1.0), state1, lr=0.1, momentum=0.9)
    assert state2["velocity_x"] == pytest.approx(-0.19)
    assert pos2 == pytest.approx((-0.1 - 0.19, -0.1 - 0.19))


def test_sgd_zero_gradient_is_a_fixed_point_without_momentum():
    pos, state = sgd_step((3.0, -4.0), (0.0, 0.0), {}, lr=0.5, momentum=0.0)
    assert pos == (3.0, -4.0)


def test_rmsprop_matches_hand_computation_first_step():
    grad = (2.0, -4.0)
    pos, state = rmsprop_step((0.0, 0.0), grad, {}, lr=0.1, decay=0.9, eps=1e-8)

    sx = 0.1 * grad[0] ** 2  # decay=0.9 -> (1-decay)=0.1
    sy = 0.1 * grad[1] ** 2
    expected_x = 0.0 - 0.1 * grad[0] / (sx**0.5 + 1e-8)
    expected_y = 0.0 - 0.1 * grad[1] / (sy**0.5 + 1e-8)

    assert state["avg_sq_grad_x"] == pytest.approx(sx)
    assert pos == pytest.approx((expected_x, expected_y))


def test_rmsprop_constant_gradient_step_size_shrinks_over_time():
    """With a constant gradient, RMSProp's running average grows, so each
    step's displacement should monotonically shrink toward a fixed point."""
    pos = (0.0, 0.0)
    state = {}
    displacements = []
    for _ in range(5):
        new_pos, state = rmsprop_step(pos, (1.0, 0.0), state, lr=0.1)
        displacements.append(abs(new_pos[0] - pos[0]))
        pos = new_pos

    for earlier, later in zip(displacements, displacements[1:]):
        assert later <= earlier + 1e-9


def test_adam_first_step_matches_hand_computation():
    grad = (1.0, -2.0)
    pos, state = adam_step((0.0, 0.0), grad, {}, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    mx = 0.1 * grad[0]  # (1-beta1)*grad
    vx = 0.001 * grad[0] ** 2  # (1-beta2)*grad^2
    mx_hat = mx / (1 - 0.9**1)
    vx_hat = vx / (1 - 0.999**1)
    expected_x = 0.0 - 0.1 * mx_hat / (vx_hat**0.5 + 1e-8)

    assert state["t"] == 1
    assert pos[0] == pytest.approx(expected_x, rel=1e-6)


def test_adam_bias_correction_makes_early_steps_close_to_signed_lr():
    """A well-known Adam property: with a constant gradient, the very first
    step's magnitude is approximately `lr` regardless of gradient scale,
    because bias correction cancels the initialization-at-zero bias."""
    pos, _ = adam_step((0.0, 0.0), (1000.0, -1000.0), {}, lr=0.1)
    assert abs(pos[0]) == pytest.approx(0.1, abs=1e-3)
    assert abs(pos[1]) == pytest.approx(0.1, abs=1e-3)


def test_adam_state_step_counter_increments():
    state = {}
    pos = (0.0, 0.0)
    for expected_t in range(1, 4):
        pos, state = adam_step(pos, (0.5, 0.5), state, lr=0.01)
        assert state["t"] == expected_t


@pytest.mark.parametrize("name,fn", [("sgd", sgd_step), ("rmsprop", rmsprop_step), ("adam", adam_step)])
def test_all_optimizers_move_monotonically_downhill_on_a_simple_bowl(name, fn):
    """Sanity property every optimizer must satisfy: on a convex bowl, each
    step must not move further from the minimum than the step before
    (distance-to-origin is non-increasing, step over step). Adaptive
    optimizers (Adam/RMSProp) normalize by gradient magnitude and so take
    small, roughly lr-sized steps regardless of distance — hence this checks
    per-step monotonicity rather than a fixed distance after N steps.
    """
    pos = (5.0, 5.0)
    state = {}
    distance = (pos[0] ** 2 + pos[1] ** 2) ** 0.5
    for _ in range(50):
        grad = (2 * pos[0], 2 * pos[1])  # gradient of x^2 + y^2
        pos, state = fn(pos, grad, state, lr=0.05)
        new_distance = (pos[0] ** 2 + pos[1] ** 2) ** 0.5
        assert new_distance <= distance + 1e-9
        distance = new_distance
    assert distance < 5.0


def test_get_optimizer_step_returns_correct_function():
    assert get_optimizer_step("sgd") is sgd_step
    assert get_optimizer_step("adam") is adam_step
    assert get_optimizer_step("rmsprop") is rmsprop_step


def test_get_optimizer_step_unknown_name_raises():
    with pytest.raises(ValueError):
        get_optimizer_step("not_a_real_optimizer")


def test_validate_hyperparams_accepts_known_keys():
    validate_hyperparams("sgd", {"momentum": 0.9})
    validate_hyperparams("adam", {"beta1": 0.9, "beta2": 0.999})


def test_validate_hyperparams_rejects_unknown_keys():
    with pytest.raises(ValueError):
        validate_hyperparams("sgd", {"beta1": 0.9})  # sgd has no beta1


def test_validate_hyperparams_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        validate_hyperparams("not_a_real_optimizer", {})


def test_hyperparam_registry_keys_match_step_function_registry():
    from app.engine.training.optimizers import OPTIMIZER_STEPS

    assert set(OPTIMIZER_HYPERPARAMS) == set(OPTIMIZER_STEPS)
