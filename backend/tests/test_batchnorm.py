import json

import pytest
import torch

from app.engine.training.batchnorm import compute_batchnorm_trace


def test_default_gamma_beta_normalizes_to_zero_mean_unit_std():
    batch = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    result = compute_batchnorm_trace(batch)
    for m in result["output_mean"]:
        assert m == pytest.approx(0.0, abs=1e-5)
    for s in result["output_std"]:
        assert s == pytest.approx(1.0, abs=1e-3)


def test_custom_gamma_beta_shifts_and_scales_output_distribution():
    batch = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    gamma, beta = [2.0], [10.0]
    result = compute_batchnorm_trace(batch, gamma=gamma, beta=beta)
    # output = 2*normalized + 10 -> mean should land on beta, std on |gamma|
    assert result["output_mean"][0] == pytest.approx(10.0, abs=1e-4)
    assert result["output_std"][0] == pytest.approx(2.0, abs=1e-3)


def test_manual_hand_computation_for_tiny_batch():
    # feature values [0, 2, 4] -> mean=2, biased variance = ((2^2)+(0)+(2^2))/3 = 8/3
    batch = [[0.0], [2.0], [4.0]]
    result = compute_batchnorm_trace(batch, eps=0.0)
    assert result["mean"][0] == pytest.approx(2.0)
    assert result["variance"][0] == pytest.approx(8 / 3)

    expected_normalized = [(v[0] - 2.0) / (8 / 3) ** 0.5 for v in batch]
    actual_normalized = [row[0] for row in result["normalized"]]
    assert actual_normalized == pytest.approx(expected_normalized, abs=1e-5)


def test_output_equals_normalized_when_gamma_one_beta_zero():
    batch = [[1.0, 5.0], [3.0, 7.0], [5.0, 9.0]]
    result = compute_batchnorm_trace(batch, gamma=[1.0, 1.0], beta=[0.0, 0.0])
    assert torch.allclose(torch.tensor(result["output"]), torch.tensor(result["normalized"]), atol=1e-6)


def test_constant_feature_column_does_not_explode_with_default_eps():
    """A feature with zero variance must not produce inf/NaN when eps>0."""
    batch = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]
    result = compute_batchnorm_trace(batch)  # default eps
    assert result["variance"][0] == pytest.approx(0.0, abs=1e-9)
    for v in result["normalized"]:
        assert v[0] == pytest.approx(0.0, abs=1e-2)  # (x-mean)/sqrt(eps) ~ 0 since x==mean exactly


def test_result_is_json_serializable():
    batch = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result = compute_batchnorm_trace(batch)
    json.dumps(result)


def test_batch_size_and_num_features_reported_correctly():
    batch = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = compute_batchnorm_trace(batch)
    assert result["batch_size"] == 2
    assert result["num_features"] == 3


def test_empty_batch_raises():
    with pytest.raises(ValueError):
        compute_batchnorm_trace([])


def test_single_sample_batch_raises():
    with pytest.raises(ValueError):
        compute_batchnorm_trace([[1.0, 2.0]])


def test_ragged_batch_rows_raise():
    with pytest.raises(ValueError):
        compute_batchnorm_trace([[1.0, 2.0], [3.0]])


def test_wrong_length_gamma_raises():
    with pytest.raises(ValueError):
        compute_batchnorm_trace([[1.0, 2.0], [3.0, 4.0]], gamma=[1.0])


def test_wrong_length_beta_raises():
    with pytest.raises(ValueError):
        compute_batchnorm_trace([[1.0, 2.0], [3.0, 4.0]], beta=[0.0, 0.0, 0.0])
