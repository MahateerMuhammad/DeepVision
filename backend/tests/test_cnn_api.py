import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

CONV_SPEC = {
    "input_channels": 1,
    "input_height": 8,
    "input_width": 8,
    "seed": 42,
    "layers": [
        {"kind": "conv", "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1},
        {"kind": "pool", "pool_type": "max", "kernel_size": 2},
    ],
}

CLASSIFIER_SPEC = {**CONV_SPEC, "num_classes": 4}

IMAGE_8x8 = [[[(i + j) / 16 for j in range(8)] for i in range(8)]]


def _create_cnn(spec=CONV_SPEC):
    resp = client.post("/cnn", json={"spec": spec})
    assert resp.status_code == 200
    return resp.json()


def test_create_cnn_returns_id_and_output_shape():
    body = _create_cnn()
    assert "network_id" in body
    assert body["output_shape"] == [2, 4, 4]


def test_create_cnn_invalid_spec_400():
    bad_spec = {**CONV_SPEC, "layers": [{"kind": "conv", "out_channels": 2, "kernel_size": 99}]}
    resp = client.post("/cnn", json={"spec": bad_spec})
    assert resp.status_code == 400


def test_create_cnn_malformed_layer_kind_422():
    bad_spec = {**CONV_SPEC, "layers": [{"kind": "not_real"}]}
    resp = client.post("/cnn", json={"spec": bad_spec})
    assert resp.status_code == 422


def test_forward_pass_endpoint():
    body = _create_cnn()
    resp = client.post("/cnn/forward", json={"network_id": body["network_id"], "image": IMAGE_8x8})
    assert resp.status_code == 200
    tree = resp.json()
    assert len(tree["layers"]) == 2
    assert tree["layers"][0]["out_shape"] == [2, 8, 8]
    assert tree["layers"][1]["out_shape"] == [2, 4, 4]
    assert "kept_mask" in tree["layers"][1]


def test_forward_pass_unknown_network_404():
    resp = client.post("/cnn/forward", json={"network_id": "does-not-exist", "image": IMAGE_8x8})
    assert resp.status_code == 404


def test_forward_pass_wrong_shape_400():
    body = _create_cnn()
    bad_image = [[[0.0] * 8 for _ in range(7)]]  # wrong height
    resp = client.post("/cnn/forward", json={"network_id": body["network_id"], "image": bad_image})
    assert resp.status_code == 400


def test_receptive_field_endpoint():
    body = _create_cnn()
    resp = client.post(
        "/cnn/receptive-field",
        json={"network_id": body["network_id"], "layer_index": 1, "channel": 0, "row": 0, "col": 0},
    )
    assert resp.status_code == 200
    rf = resp.json()
    assert "clipped_row_range" in rf and "clipped_col_range" in rf


def test_receptive_field_out_of_range_400():
    body = _create_cnn()
    resp = client.post(
        "/cnn/receptive-field",
        json={"network_id": body["network_id"], "layer_index": 99, "channel": 0, "row": 0, "col": 0},
    )
    assert resp.status_code == 400


def test_saliency_endpoint():
    body = _create_cnn(CLASSIFIER_SPEC)
    resp = client.post(
        "/cnn/saliency",
        json={"network_id": body["network_id"], "image": IMAGE_8x8, "target_class": 1},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert len(result["probabilities"]) == 4
    assert sum(result["probabilities"]) == pytest.approx(1.0, abs=1e-3)


def test_saliency_without_classifier_head_400():
    body = _create_cnn(CONV_SPEC)  # no num_classes
    resp = client.post(
        "/cnn/saliency",
        json={"network_id": body["network_id"], "image": IMAGE_8x8, "target_class": 0},
    )
    assert resp.status_code == 400


def test_saliency_out_of_range_class_400():
    body = _create_cnn(CLASSIFIER_SPEC)
    resp = client.post(
        "/cnn/saliency",
        json={"network_id": body["network_id"], "image": IMAGE_8x8, "target_class": 99},
    )
    assert resp.status_code == 400


def test_sliding_kernel_endpoint_does_not_require_a_network():
    resp = client.post(
        "/cnn/sliding-kernel",
        json={
            "image": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "kernel": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "stride": 1,
            "padding": 0,
        },
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["output"] == [[5]]


def test_sliding_kernel_invalid_kernel_400():
    resp = client.post(
        "/cnn/sliding-kernel",
        json={"image": [[1, 2], [3, 4]], "kernel": [[0] * 5 for _ in range(5)], "stride": 1, "padding": 0},
    )
    assert resp.status_code == 400


def test_delete_cnn_then_404_on_use():
    body = _create_cnn()
    del_resp = client.delete(f"/cnn/{body['network_id']}")
    assert del_resp.status_code == 204

    resp = client.post("/cnn/forward", json={"network_id": body["network_id"], "image": IMAGE_8x8})
    assert resp.status_code == 404


def test_full_session_forward_then_receptive_field_then_saliency():
    """Simulate the frontend flow: create a classifier CNN, run a forward
    pass, back-trace a receptive field, then ask for saliency — all against
    the same persisted network.
    """
    body = _create_cnn(CLASSIFIER_SPEC)
    nid = body["network_id"]

    fwd = client.post("/cnn/forward", json={"network_id": nid, "image": IMAGE_8x8}).json()
    assert len(fwd["output"]) == 4

    rf = client.post(
        "/cnn/receptive-field",
        json={"network_id": nid, "layer_index": 0, "channel": 0, "row": 3, "col": 3},
    ).json()
    assert rf["receptive_field_size"] == [3, 3]

    sal = client.post(
        "/cnn/saliency", json={"network_id": nid, "image": IMAGE_8x8, "target_class": 2}
    ).json()
    assert sal["predicted_class"] == sal["logits"].index(max(sal["logits"]))
