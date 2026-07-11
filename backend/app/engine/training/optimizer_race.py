import math

from app.engine.training.loss_surfaces import get_surface
from app.engine.training.optimizers import get_optimizer_step, validate_hyperparams

DIVERGENCE_MAGNITUDE = 1e6


def _is_finite_point(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y) and abs(x) < DIVERGENCE_MAGNITUDE and abs(y) < DIVERGENCE_MAGNITUDE


def _momentum_vector(optimizer_name: str, state: dict) -> list[float] | None:
    """The 'physical velocity arrow' for the Adam-memory visualization: SGD's
    momentum term for sgd, Adam's first moment (m_t) for adam, and None for
    RMSProp, which has no momentum concept (only a per-axis adaptive scale).
    """
    if optimizer_name == "sgd":
        return [state.get("velocity_x", 0.0), state.get("velocity_y", 0.0)]
    if optimizer_name == "adam":
        return [state.get("momentum_x", 0.0), state.get("momentum_y", 0.0)]
    return None


def run_trajectory(
    surface_name: str,
    start: tuple[float, float],
    optimizer_name: str,
    lr: float,
    hyperparams: dict,
    num_steps: int,
) -> dict:
    """Step one optimizer across one analytic surface for up to `num_steps`
    iterations, recording position/loss/internal-state at every tick.

    Stops early (without raising) the moment a step produces a non-finite or
    blown-up position — this IS the "learning rate too high -> diverges"
    behavior the frontend visualizes, not an error condition.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    validate_hyperparams(optimizer_name, hyperparams)

    surface = get_surface(surface_name)
    fn, grad_fn = surface["fn"], surface["grad"]
    step_fn = get_optimizer_step(optimizer_name)

    pos = tuple(start)
    state: dict = {}
    points = [
        {
            "step": 0,
            "x": pos[0],
            "y": pos[1],
            "loss": fn(*pos),
            "state": {},
            "momentum_vector": _momentum_vector(optimizer_name, {}),
            "diverged": False,
        }
    ]
    diverged = False

    for step in range(1, num_steps + 1):
        grad = grad_fn(*pos)
        try:
            new_pos, new_state = step_fn(pos, grad, state, lr=lr, **hyperparams)
        except OverflowError:
            new_pos, new_state = (math.inf, math.inf), state

        if not _is_finite_point(*new_pos):
            points.append(
                {
                    "step": step,
                    "x": None,
                    "y": None,
                    "loss": None,
                    "state": None,
                    "momentum_vector": None,
                    "diverged": True,
                }
            )
            diverged = True
            break

        pos, state = new_pos, new_state
        loss = fn(*pos)
        points.append(
            {
                "step": step,
                "x": pos[0],
                "y": pos[1],
                "loss": loss,
                "state": dict(state),
                "momentum_vector": _momentum_vector(optimizer_name, state),
                "diverged": False,
            }
        )

    return {
        "surface": surface_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "hyperparams": hyperparams,
        "diverged": diverged,
        "points": points,
    }


def run_race(
    surface_name: str,
    start: tuple[float, float],
    racers: list[dict],
    num_steps: int,
) -> dict:
    """Run several optimizers from the SAME starting point on the SAME
    surface, side by side — the "optimizer racing" visualization.

    Each racer dict: {"label": str, "optimizer": str, "lr": float, "hyperparams": dict}.
    """
    if not racers:
        raise ValueError("racers must be a non-empty list")

    results = []
    for racer in racers:
        trajectory = run_trajectory(
            surface_name,
            start,
            racer["optimizer"],
            racer["lr"],
            racer.get("hyperparams", {}),
            num_steps,
        )
        results.append({"label": racer["label"], **trajectory})

    return {"surface": surface_name, "start": list(start), "num_steps": num_steps, "racers": results}
