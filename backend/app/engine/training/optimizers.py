"""Manual, from-scratch implementations of the textbook optimizer update
rules — deliberately not using torch.optim, so every internal state variable
(momentum, RMSProp's running average, Adam's m_t/v_t) is exposed as plain
numbers for the frontend to visualize step by step.

Each `*_step` function is a pure function: (position, gradient, state,
hyperparameters) -> (new_position, new_state). No hidden mutation, so a
single step is trivial to unit-test against a hand-worked example.
"""

Point = tuple[float, float]
Gradient = tuple[float, float]


def sgd_step(pos: Point, grad: Gradient, state: dict, lr: float, momentum: float = 0.0) -> tuple[Point, dict]:
    vx = momentum * state.get("velocity_x", 0.0) - lr * grad[0]
    vy = momentum * state.get("velocity_y", 0.0) - lr * grad[1]
    new_pos = (pos[0] + vx, pos[1] + vy)
    return new_pos, {"velocity_x": vx, "velocity_y": vy}


def rmsprop_step(
    pos: Point, grad: Gradient, state: dict, lr: float, decay: float = 0.9, eps: float = 1e-8
) -> tuple[Point, dict]:
    sx = decay * state.get("avg_sq_grad_x", 0.0) + (1 - decay) * grad[0] ** 2
    sy = decay * state.get("avg_sq_grad_y", 0.0) + (1 - decay) * grad[1] ** 2
    new_x = pos[0] - lr * grad[0] / (sx**0.5 + eps)
    new_y = pos[1] - lr * grad[1] / (sy**0.5 + eps)
    return (new_x, new_y), {"avg_sq_grad_x": sx, "avg_sq_grad_y": sy}


def adam_step(
    pos: Point,
    grad: Gradient,
    state: dict,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Point, dict]:
    t = state.get("t", 0) + 1
    mx = beta1 * state.get("momentum_x", 0.0) + (1 - beta1) * grad[0]
    my = beta1 * state.get("momentum_y", 0.0) + (1 - beta1) * grad[1]
    vx = beta2 * state.get("variance_x", 0.0) + (1 - beta2) * grad[0] ** 2
    vy = beta2 * state.get("variance_y", 0.0) + (1 - beta2) * grad[1] ** 2

    mx_hat = mx / (1 - beta1**t)
    my_hat = my / (1 - beta1**t)
    vx_hat = vx / (1 - beta2**t)
    vy_hat = vy / (1 - beta2**t)

    new_x = pos[0] - lr * mx_hat / (vx_hat**0.5 + eps)
    new_y = pos[1] - lr * my_hat / (vy_hat**0.5 + eps)
    return (new_x, new_y), {
        "t": t,
        "momentum_x": mx,
        "momentum_y": my,
        "variance_x": vx,
        "variance_y": vy,
    }


OPTIMIZER_STEPS = {"sgd": sgd_step, "rmsprop": rmsprop_step, "adam": adam_step}

# Hyperparameter keys each optimizer accepts beyond `lr`, used to validate
# and filter user-supplied hyperparameter dicts before calling *_step.
OPTIMIZER_HYPERPARAMS = {
    "sgd": {"momentum"},
    "rmsprop": {"decay", "eps"},
    "adam": {"beta1", "beta2", "eps"},
}


def get_optimizer_step(name: str):
    if name not in OPTIMIZER_STEPS:
        raise ValueError(f"unknown optimizer '{name}', expected one of {sorted(OPTIMIZER_STEPS)}")
    return OPTIMIZER_STEPS[name]


def validate_hyperparams(name: str, hyperparams: dict) -> None:
    if name not in OPTIMIZER_HYPERPARAMS:
        raise ValueError(f"unknown optimizer '{name}', expected one of {sorted(OPTIMIZER_HYPERPARAMS)}")
    allowed = OPTIMIZER_HYPERPARAMS[name]
    unknown = set(hyperparams) - allowed
    if unknown:
        raise ValueError(f"optimizer '{name}' does not accept hyperparameters {sorted(unknown)}")
