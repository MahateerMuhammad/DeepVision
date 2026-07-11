import torch
import torch.nn.functional as F


def compute_loss(output: torch.Tensor, target: list[float], loss_name: str) -> torch.Tensor:
    """output is shape (1, out_features). target semantics depend on loss_name."""
    if loss_name == "mse":
        target_t = torch.tensor([target], dtype=torch.float32)
        return F.mse_loss(output, target_t)

    if loss_name == "bce":
        target_t = torch.tensor([target], dtype=torch.float32)
        return F.binary_cross_entropy(output, target_t)

    if loss_name == "cross_entropy":
        if len(target) != 1:
            raise ValueError(
                "cross_entropy target must be a single-element list holding the class index, "
                "e.g. [2]"
            )
        target_idx = torch.tensor([int(target[0])], dtype=torch.long)
        return F.cross_entropy(output, target_idx)

    raise ValueError(f"unknown loss function: {loss_name}")
