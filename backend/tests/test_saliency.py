import json

import pytest
import torch

from app.engine.cnn.model import build_cnn
from app.engine.cnn.saliency import compute_saliency_map
from app.schemas.cnn import CNNSpec, ConvLayerSpec, PoolLayerSpec


def _classifier_spec(seed=5):
    return CNNSpec(
        input_channels=1, input_height=8, input_width=8, seed=seed, num_classes=4,
        layers=[
            ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=1),
            PoolLayerSpec(pool_type="max", kernel_size=2),
        ],
    )


def _image(seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 8, 8, generator=g).tolist()


def test_saliency_output_shapes_and_json_safe():
    net = build_cnn(_classifier_spec())
    result = compute_saliency_map(net, _image(), target_class=2)
    json.dumps(result)

    assert len(result["logits"]) == 4
    assert len(result["probabilities"]) == 4
    assert len(result["saliency_map"]) == 8 and len(result["saliency_map"][0]) == 8
    assert len(result["input_gradient"]) == 1  # channels
    assert len(result["input_gradient"][0]) == 8 and len(result["input_gradient"][0][0]) == 8


def test_probabilities_sum_to_one():
    net = build_cnn(_classifier_spec())
    result = compute_saliency_map(net, _image(), target_class=0)
    assert sum(result["probabilities"]) == pytest.approx(1.0, abs=1e-4)


def test_predicted_class_matches_argmax_of_logits():
    net = build_cnn(_classifier_spec())
    result = compute_saliency_map(net, _image(), target_class=1)
    assert result["predicted_class"] == result["logits"].index(max(result["logits"]))


def test_saliency_map_is_nonnegative():
    net = build_cnn(_classifier_spec())
    result = compute_saliency_map(net, _image(), target_class=3)
    for row in result["saliency_map"]:
        for v in row:
            assert v >= 0.0


def test_gradient_matches_finite_difference_probe():
    """Ground-truth check: perturb a single input pixel and confirm the
    numerical slope of the target logit matches the captured input_gradient
    at that pixel.
    """
    net = build_cnn(_classifier_spec())
    image = _image()
    result = compute_saliency_map(net, image, target_class=1)
    analytic = result["input_gradient"][0][3][4]

    eps = 1e-3

    def logit_with_perturbation(delta: float) -> float:
        probe_image = [[row[:] for row in image[0]]]
        probe_image[0][3][4] += delta
        x = torch.tensor(probe_image, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return net(x)[0, 1].item()

    numeric = (logit_with_perturbation(eps) - logit_with_perturbation(-eps)) / (2 * eps)
    assert analytic == pytest.approx(numeric, abs=1e-2)


def test_no_classifier_head_raises():
    spec = CNNSpec(
        input_channels=1, input_height=6, input_width=6,
        layers=[ConvLayerSpec(out_channels=2, kernel_size=3)],
    )
    net = build_cnn(spec)
    with pytest.raises(ValueError):
        compute_saliency_map(net, torch.rand(1, 6, 6).tolist(), target_class=0)


def test_out_of_range_target_class_raises():
    net = build_cnn(_classifier_spec())
    with pytest.raises(ValueError):
        compute_saliency_map(net, _image(), target_class=99)


def test_wrong_image_shape_raises():
    net = build_cnn(_classifier_spec())
    with pytest.raises(ValueError):
        compute_saliency_map(net, torch.rand(1, 5, 8).tolist(), target_class=0)


def test_repeated_calls_do_not_accumulate_gradients():
    net = build_cnn(_classifier_spec())
    image = _image()
    r1 = compute_saliency_map(net, image, target_class=2)
    r2 = compute_saliency_map(net, image, target_class=2)
    assert torch.allclose(torch.tensor(r1["input_gradient"]), torch.tensor(r2["input_gradient"]), atol=1e-6)
