import itertools
import json

import pytest

from app.engine.training.loss_surfaces import SURFACES, generate_grid, get_surface

EPS = 1e-5
PROBE_POINTS = [(-2.0, -2.0), (-1.0, 0.5), (0.0, 0.0), (0.3, -0.7), (1.5, 1.5), (2.7, -1.3)]


@pytest.mark.parametrize("name", sorted(SURFACES))
@pytest.mark.parametrize("x,y", PROBE_POINTS)
def test_closed_form_gradient_matches_finite_difference(name, x, y):
    surface = get_surface(name)
    fn, grad = surface["fn"], surface["grad"]

    numeric_dx = (fn(x + EPS, y) - fn(x - EPS, y)) / (2 * EPS)
    numeric_dy = (fn(x, y + EPS) - fn(x, y - EPS)) / (2 * EPS)
    analytic_dx, analytic_dy = grad(x, y)

    assert analytic_dx == pytest.approx(numeric_dx, abs=1e-2)
    assert analytic_dy == pytest.approx(numeric_dy, abs=1e-2)


def test_bowl_minimum_is_at_origin_with_zero_gradient():
    from app.engine.training.loss_surfaces import bowl, bowl_grad

    assert bowl(0, 0) == 0
    assert bowl_grad(0, 0) == (0, 0)
    # any other point must have strictly positive value
    for x, y in PROBE_POINTS:
        if (x, y) != (0, 0):
            assert bowl(x, y) > 0


def test_saddle_gradient_is_zero_at_origin_but_it_is_not_a_minimum():
    from app.engine.training.loss_surfaces import saddle, saddle_grad

    assert saddle_grad(0, 0) == (0, 0)
    # moving along y decreases the value -> origin is NOT a minimum
    assert saddle(0, 0.1) < saddle(0, 0)
    # moving along x increases the value -> origin IS a min along that axis
    assert saddle(0.1, 0) > saddle(0, 0)


def test_rosenbrock_global_minimum_is_zero_at_a_asquared():
    from app.engine.training.loss_surfaces import ROSENBROCK_A, rosenbrock, rosenbrock_grad

    assert rosenbrock(ROSENBROCK_A, ROSENBROCK_A**2) == pytest.approx(0.0, abs=1e-9)
    dx, dy = rosenbrock_grad(ROSENBROCK_A, ROSENBROCK_A**2)
    assert dx == pytest.approx(0.0, abs=1e-9)
    assert dy == pytest.approx(0.0, abs=1e-9)


def test_get_surface_unknown_name_raises():
    with pytest.raises(ValueError):
        get_surface("not_a_real_surface")


def test_generate_grid_shape_is_resolution_by_resolution():
    grid = generate_grid("bowl", -2, 2, -2, 2, resolution=15)
    assert len(grid["x"]) == 15
    assert len(grid["y"]) == 15
    assert len(grid["z"]) == 15
    assert all(len(row) == 15 for row in grid["z"])


def test_generate_grid_values_match_direct_function_calls():
    from app.engine.training.loss_surfaces import saddle

    grid = generate_grid("saddle", -1, 1, -1, 1, resolution=5)
    for row_idx, y in enumerate(grid["y"]):
        for col_idx, x in enumerate(grid["x"]):
            assert grid["z"][row_idx][col_idx] == pytest.approx(saddle(x, y), abs=1e-9)


def test_generate_grid_endpoints_match_requested_bounds():
    grid = generate_grid("rosenbrock", -3, 5, -1, 7, resolution=10)
    assert grid["x"][0] == pytest.approx(-3)
    assert grid["x"][-1] == pytest.approx(5)
    assert grid["y"][0] == pytest.approx(-1)
    assert grid["y"][-1] == pytest.approx(7)


def test_generate_grid_is_json_serializable():
    grid = generate_grid("bowl", -2, 2, -2, 2, resolution=8)
    json.dumps(grid)


@pytest.mark.parametrize(
    "x_min,x_max,y_min,y_max",
    [(1, 1, -1, 1), (-1, 1, 2, 2), (5, -5, -1, 1)],
)
def test_generate_grid_rejects_degenerate_or_inverted_ranges(x_min, x_max, y_min, y_max):
    with pytest.raises(ValueError):
        generate_grid("bowl", x_min, x_max, y_min, y_max, resolution=10)


def test_generate_grid_rejects_resolution_below_two():
    with pytest.raises(ValueError):
        generate_grid("bowl", -1, 1, -1, 1, resolution=1)


def test_generate_grid_unknown_surface_raises():
    with pytest.raises(ValueError):
        generate_grid("not_a_real_surface", -1, 1, -1, 1)
