import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

ANN_SPEC = {
    "input_size": 3,
    "seed": 42,
    "layers": [
        {"out_features": 4, "activation": "relu"},
        {"out_features": 2, "activation": "sigmoid"},
    ],
}


def _create_ann():
    resp = client.post("/networks", json={"spec": ANN_SPEC})
    assert resp.status_code == 200
    return resp.json()


def test_loss_surface_endpoint():
    resp = client.post(
        "/training/loss-surface",
        json={"surface": "bowl", "x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2, "resolution": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["z"]) == 10


def test_loss_surface_unknown_surface_422():
    resp = client.post(
        "/training/loss-surface",
        json={"surface": "not_real", "x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2},
    )
    assert resp.status_code == 422  # pydantic Literal rejects it


def test_loss_surface_bad_range_400():
    resp = client.post(
        "/training/loss-surface",
        json={"surface": "bowl", "x_min": 2, "x_max": -2, "y_min": -2, "y_max": 2},
    )
    assert resp.status_code == 400


def test_race_endpoint_runs_multiple_optimizers():
    resp = client.post(
        "/training/race",
        json={
            "surface": "bowl",
            "start": [3.0, 3.0],
            "racers": [
                {"label": "sgd", "optimizer": "sgd", "lr": 0.05, "hyperparams": {"momentum": 0.9}},
                {"label": "adam", "optimizer": "adam", "lr": 0.05, "hyperparams": {}},
            ],
            "num_steps": 20,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["racers"]) == 2


def test_race_invalid_hyperparam_400():
    resp = client.post(
        "/training/race",
        json={
            "surface": "bowl",
            "start": [1.0, 1.0],
            "racers": [{"label": "bad", "optimizer": "sgd", "lr": 0.05, "hyperparams": {"beta1": 0.9}}],
            "num_steps": 10,
        },
    )
    assert resp.status_code == 400


def test_race_empty_racers_422():
    resp = client.post(
        "/training/race", json={"surface": "bowl", "start": [1.0, 1.0], "racers": [], "num_steps": 10}
    )
    assert resp.status_code == 422  # min_length=1 violated


def test_weight_loss_surface_endpoint():
    body = _create_ann()
    resp = client.post(
        "/training/weight-loss-surface",
        json={
            "network_id": body["network_id"],
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "loss": "mse",
            "coord1": {"layer_index": 0, "row": 0, "col": 0},
            "coord2": {"layer_index": 1, "row": 1, "col": 2},
            "range1": [-2, 2],
            "range2": [-2, 2],
            "resolution": 9,
        },
    )
    assert resp.status_code == 200
    surface = resp.json()
    assert len(surface["z"]) == 9


def test_weight_loss_surface_unknown_network_404():
    resp = client.post(
        "/training/weight-loss-surface",
        json={
            "network_id": "does-not-exist",
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "coord1": {"layer_index": 0, "row": 0, "col": 0},
            "coord2": {"layer_index": 1, "row": 1, "col": 2},
            "range1": [-2, 2],
            "range2": [-2, 2],
        },
    )
    assert resp.status_code == 404


def test_weight_loss_surface_bad_coord_400():
    body = _create_ann()
    resp = client.post(
        "/training/weight-loss-surface",
        json={
            "network_id": body["network_id"],
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "coord1": {"layer_index": 99, "row": 0, "col": 0},
            "coord2": {"layer_index": 1, "row": 1, "col": 2},
            "range1": [-2, 2],
            "range2": [-2, 2],
        },
    )
    assert resp.status_code == 400


def test_weight_loss_surface_leaves_network_forward_pass_unaffected():
    """After computing a weight-loss-surface slice, the network's actual
    weights must be restored, so a subsequent forward pass through the
    normal /networks/forward endpoint must be unaffected."""
    body = _create_ann()
    nid = body["network_id"]
    x = [1.0, -2.0, 0.5]

    before = client.post("/networks/forward", json={"network_id": nid, "input": x}).json()
    client.post(
        "/training/weight-loss-surface",
        json={
            "network_id": nid,
            "input": x,
            "target": [0.2, 0.8],
            "coord1": {"layer_index": 0, "row": 0, "col": 0},
            "coord2": {"layer_index": 1, "row": 1, "col": 2},
            "range1": [-5, 5],
            "range2": [-5, 5],
            "resolution": 6,
        },
    )
    after = client.post("/networks/forward", json={"network_id": nid, "input": x}).json()
    assert before["output"] == pytest.approx(after["output"], abs=1e-6)


def test_batchnorm_endpoint():
    resp = client.post(
        "/training/batchnorm",
        json={"batch": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["batch_size"] == 4
    for m in body["output_mean"]:
        assert abs(m) < 1e-4


def test_batchnorm_single_sample_400():
    resp = client.post("/training/batchnorm", json={"batch": [[1.0, 2.0]]})
    assert resp.status_code == 400


def test_batchnorm_custom_gamma_beta():
    resp = client.post(
        "/training/batchnorm",
        json={"batch": [[1.0], [2.0], [3.0]], "gamma": [2.0], "beta": [5.0]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["output_mean"][0] == pytest.approx(5.0, abs=1e-3)
