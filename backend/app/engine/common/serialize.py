import torch

ROUND_DECIMALS = 6


def to_list(tensor: torch.Tensor) -> list:
    """Detach a tensor from the graph and convert to plain, JSON-safe nested lists."""
    return tensor.detach().cpu().numpy().round(ROUND_DECIMALS).tolist()


def to_scalar(tensor: torch.Tensor) -> float:
    return round(float(tensor.detach().cpu().item()), ROUND_DECIMALS)
