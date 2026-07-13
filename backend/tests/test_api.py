import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SPEC = {
    "input_size": 3,
    "seed": 42,
    "layers": [
        {"out_features": 4, "activation": "relu"},
        {"out_features": 2, "activation": "sigmoid"},
    ],
}


def _create_network():
    resp = client.post("/networks", json={"spec": SPEC})
    assert resp.status_code == 200
    return resp.json()


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_network_returns_id_and_param_count():
    body = _create_network()
    assert "network_id" in body
    # (3*4+4) + (4*2+2) = 16 + 10 = 26
    assert body["param_count"] == 26


def test_forward_pass_endpoint():
    body = _create_network()
    resp = client.post(
        "/networks/forward",
        json={"network_id": body["network_id"], "input": [1.0, -2.0, 0.5]},
    )
    assert resp.status_code == 200
    tree = resp.json()
    assert len(tree["layers"]) == 2
    assert len(tree["output"]) == 2


def test_forward_pass_unknown_network_404():
    resp = client.post("/networks/forward", json={"network_id": "does-not-exist", "input": [1, 2, 3]})
    assert resp.status_code == 404


def test_forward_pass_bad_input_length_400():
    body = _create_network()
    resp = client.post(
        "/networks/forward", json={"network_id": body["network_id"], "input": [1.0, 2.0]}
    )
    assert resp.status_code == 400


def test_backward_pass_endpoint():
    body = _create_network()
    resp = client.post(
        "/networks/backward",
        json={
            "network_id": body["network_id"],
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "loss": "mse",
        },
    )
    assert resp.status_code == 200
    tree = resp.json()
    assert "loss" in tree
    assert len(tree["layers"][0]["grad_weights"]) == 4


def test_trace_endpoint_matches_analytic_gradient():
    body = _create_network()
    resp = client.post(
        "/networks/trace",
        json={
            "network_id": body["network_id"],
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "loss": "mse",
            "layer_index": 0,
            "weight_row": 1,
            "weight_col": 2,
        },
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["matches_analytic"] is True


def test_trace_endpoint_out_of_range_400():
    body = _create_network()
    resp = client.post(
        "/networks/trace",
        json={
            "network_id": body["network_id"],
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "loss": "mse",
            "layer_index": 99,
            "weight_row": 0,
            "weight_col": 0,
        },
    )
    assert resp.status_code == 400


def test_invalid_spec_rejected_by_validation():
    bad_spec = {"input_size": 3, "layers": []}
    resp = client.post("/networks", json={"spec": bad_spec})
    assert resp.status_code == 422  # pydantic validation error


def test_delete_network_then_404_on_use():
    body = _create_network()
    del_resp = client.delete(f"/networks/{body['network_id']}")
    assert del_resp.status_code == 204

    resp = client.post(
        "/networks/forward", json={"network_id": body["network_id"], "input": [1.0, -2.0, 0.5]}
    )
    assert resp.status_code == 404


def test_step_returns_loss_history_and_state_tree():
    body = _create_network()
    resp = client.post(
        "/networks/step",
        json={
            "network_id": body["network_id"],
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "loss": "mse",
            "learning_rate": 0.5,
            "num_steps": 10,
            "training": False,
        },
    )
    assert resp.status_code == 200
    result = resp.json()
    # loss_history has one entry per step plus the final post-update loss
    assert len(result["loss_history"]) == 11
    assert result["num_steps"] == 10
    assert "state_tree" in result and "loss" in result["state_tree"]
    # the state tree's loss closes out the descent curve
    assert result["state_tree"]["loss"] == pytest.approx(result["loss_history"][-1], abs=1e-6)


def test_step_reduces_loss():
    body = _create_network()
    resp = client.post(
        "/networks/step",
        json={
            "network_id": body["network_id"],
            "input": [1.0, -2.0, 0.5],
            "target": [0.2, 0.8],
            "loss": "mse",
            "learning_rate": 0.5,
            "num_steps": 30,
            "training": False,
        },
    ).json()
    # plain SGD on a fixed sample must monotonically-ish drive loss down
    assert resp["loss_history"][-1] < resp["loss_history"][0]


def test_step_updates_weights_persistently():
    body = _create_network()
    nid = body["network_id"]
    payload = {
        "network_id": nid,
        "input": [1.0, -2.0, 0.5],
        "target": [0.2, 0.8],
        "loss": "mse",
        "learning_rate": 0.3,
        "num_steps": 5,
        "training": False,
    }
    first = client.post("/networks/step", json=payload).json()
    w_before = first["state_tree"]["layers"][0]["weights"]

    # a fresh backward at the same weights (no update) must report the same loss
    # the previous step ended on — proving the update persisted on the registry
    bwd = client.post(
        "/networks/backward",
        json={"network_id": nid, "input": payload["input"], "target": payload["target"], "loss": "mse", "training": False},
    ).json()
    assert bwd["loss"] == pytest.approx(first["loss_history"][-1], abs=1e-6)

    # a second round of steps continues descending from the persisted weights
    second = client.post("/networks/step", json=payload).json()
    w_after = second["state_tree"]["layers"][0]["weights"]
    assert w_before != w_after
    assert second["loss_history"][0] == pytest.approx(first["loss_history"][-1], abs=1e-6)


def test_step_unknown_network_404():
    resp = client.post(
        "/networks/step",
        json={"network_id": "nope", "input": [1, 2, 3], "target": [0.2, 0.8]},
    )
    assert resp.status_code == 404


def test_step_invalid_num_steps_422():
    body = _create_network()
    resp = client.post(
        "/networks/step",
        json={"network_id": body["network_id"], "input": [1, 2, 3], "target": [0.2, 0.8], "num_steps": 0},
    )
    assert resp.status_code == 422


def test_step_bad_input_length_400():
    body = _create_network()
    resp = client.post(
        "/networks/step",
        json={"network_id": body["network_id"], "input": [1.0, 2.0], "target": [0.2, 0.8]},
    )
    assert resp.status_code == 400


def test_full_vcr_style_session_forward_then_backward_then_trace():
    """Simulate the frontend's actual usage pattern: create once, then step
    through forward, backward, and inspect a specific weight via trace — all
    against the same persisted network.
    """
    body = _create_network()
    nid = body["network_id"]
    x = [0.5, 0.5, -1.0]
    target = [1.0, 0.0]

    fwd = client.post("/networks/forward", json={"network_id": nid, "input": x}).json()
    bwd = client.post(
        "/networks/backward",
        json={"network_id": nid, "input": x, "target": target, "loss": "mse"},
    ).json()
    trace = client.post(
        "/networks/trace",
        json={
            "network_id": nid,
            "input": x,
            "target": target,
            "loss": "mse",
            "layer_index": 1,
            "weight_row": 0,
            "weight_col": 3,
        },
    ).json()

    # forward and backward outputs must agree (deterministic net, same input)
    assert fwd["output"] == pytest.approx(bwd["output"], abs=1e-5)
    assert trace["matches_analytic"] is True
