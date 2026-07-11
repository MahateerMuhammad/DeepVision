import torch

from app.engine.cnn.model import DynamicCNN
from app.engine.cnn.state_tree import validate_image
from app.engine.common.serialize import to_list


def compute_saliency_map(net: DynamicCNN, image: list[list[list[float]]], target_class: int) -> dict:
    """Vanilla gradient saliency: ∂(logit for target_class)/∂(input image).
    Answers "why did the network predict this class?" by highlighting which
    input pixels most affect that class's score.
    """
    if net.classifier is None:
        raise ValueError("network has no classifier head (spec.num_classes is unset); saliency requires one")
    if not (0 <= target_class < net.spec.num_classes):
        raise ValueError(f"target_class {target_class} out of range [0, {net.spec.num_classes - 1}]")

    net.zero_grad(set_to_none=True)
    x_tensor = validate_image(net, image)
    x_tensor.requires_grad_(True)

    logits = net(x_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = int(torch.argmax(logits, dim=1).item())

    score = logits[0, target_class]
    score.backward()

    grad = x_tensor.grad[0]  # (C, H, W)
    saliency_map = grad.abs().amax(dim=0)  # (H, W): max-over-channels vanilla gradient saliency

    return {
        "target_class": target_class,
        "predicted_class": predicted_class,
        "logits": to_list(logits[0]),
        "probabilities": to_list(probabilities[0]),
        "input_gradient": to_list(grad),
        "saliency_map": to_list(saliency_map),
    }
