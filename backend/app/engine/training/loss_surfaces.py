"""Analytic 2D "loss landscape" functions used to teach optimizer behavior.

Each surface exposes a closed-form value and closed-form gradient so that
optimizer trajectories (SGD/Adam/RMSprop) can be computed exactly, without
needing an actual trained network. Gradients are cross-checked against
central finite differences in tests/test_loss_surfaces.py.
"""

ROSENBROCK_A = 1.0
ROSENBROCK_B = 100.0


def bowl(x: float, y: float) -> float:
    """Convex elongated bowl: steep in y, shallow in x. Global min at (0, 0).
    The asymmetry is deliberate — it's the classic example of why plain SGD
    oscillates across the steep axis while momentum smooths the descent.
    """
    return x**2 + 4 * y**2


def bowl_grad(x: float, y: float) -> tuple[float, float]:
    return 2 * x, 8 * y


def saddle(x: float, y: float) -> float:
    """Saddle point at the origin: minimum along x, maximum along y."""
    return x**2 - y**2


def saddle_grad(x: float, y: float) -> tuple[float, float]:
    return 2 * x, -2 * y


def rosenbrock(x: float, y: float) -> float:
    """The Rosenbrock 'banana' function: a narrow, curved, non-convex valley.
    Global minimum at (a, a^2) = (1, 1) with the default constants.
    """
    a, b = ROSENBROCK_A, ROSENBROCK_B
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rosenbrock_grad(x: float, y: float) -> tuple[float, float]:
    a, b = ROSENBROCK_A, ROSENBROCK_B
    dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dy = 2 * b * (y - x**2)
    return dx, dy


SURFACES: dict[str, dict] = {
    "bowl": {"fn": bowl, "grad": bowl_grad},
    "saddle": {"fn": saddle, "grad": saddle_grad},
    "rosenbrock": {"fn": rosenbrock, "grad": rosenbrock_grad},
}


def get_surface(name: str) -> dict:
    if name not in SURFACES:
        raise ValueError(f"unknown surface '{name}', expected one of {sorted(SURFACES)}")
    return SURFACES[name]


def generate_grid(
    name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int = 40,
) -> dict:
    """Sample the surface on a resolution x resolution grid for 3D rendering."""
    if resolution < 2:
        raise ValueError("resolution must be >= 2")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("min must be strictly less than max for both axes")

    fn = get_surface(name)["fn"]
    xs = [x_min + i * (x_max - x_min) / (resolution - 1) for i in range(resolution)]
    ys = [y_min + i * (y_max - y_min) / (resolution - 1) for i in range(resolution)]
    z = [[fn(x, y) for x in xs] for y in ys]

    return {"surface": name, "x": xs, "y": ys, "z": z}
