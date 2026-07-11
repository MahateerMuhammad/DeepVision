import torch

from app.engine.ann.loss import compute_loss
from app.engine.ann.model import DynamicANN


def _validate_coord(net: DynamicANN, layer_index: int, row: int, col: int | None) -> None:
    if not (0 <= layer_index < len(net.linears)):
        raise ValueError(f"layer_index {layer_index} out of range [0, {len(net.linears) - 1}]")
    linear = net.linears[layer_index]
    if not (0 <= row < linear.out_features):
        raise ValueError(f"row {row} out of range [0, {linear.out_features - 1}]")
    if col is not None and not (0 <= col < linear.in_features):
        raise ValueError(f"col {col} out of range [0, {linear.in_features - 1}]")


def _get_value(net: DynamicANN, layer_index: int, row: int, col: int | None) -> float:
    linear = net.linears[layer_index]
    tensor = linear.bias if col is None else linear.weight
    return tensor[row].item() if col is None else tensor[row, col].item()


def _set_value(net: DynamicANN, layer_index: int, row: int, col: int | None, value: float) -> None:
    linear = net.linears[layer_index]
    with torch.no_grad():
        if col is None:
            linear.bias[row] = value
        else:
            linear.weight[row, col] = value


def compute_weight_loss_surface(
    net: DynamicANN,
    x: list[float],
    target: list[float],
    loss_name: str,
    coord1: dict,
    coord2: dict,
    range1: tuple[float, float],
    range2: tuple[float, float],
    resolution: int = 25,
) -> dict:
    """Hold every parameter fixed except two real weights/biases, sweep those
    two over a grid, and recompute the network's actual loss at each point —
    a genuine 2D slice through the real (potentially non-convex) loss surface
    of the trained network, rather than an analytic stand-in.

    coord1/coord2: {"layer_index": int, "row": int, "col": int | None}
    (col=None addresses a bias term instead of a weight).
    """
    if resolution < 2:
        raise ValueError("resolution must be >= 2")
    if range1[0] >= range1[1] or range2[0] >= range2[1]:
        raise ValueError("range min must be strictly less than max for both axes")

    _validate_coord(net, coord1["layer_index"], coord1["row"], coord1.get("col"))
    _validate_coord(net, coord2["layer_index"], coord2["row"], coord2.get("col"))

    original_1 = _get_value(net, coord1["layer_index"], coord1["row"], coord1.get("col"))
    original_2 = _get_value(net, coord2["layer_index"], coord2["row"], coord2.get("col"))

    xs = [range1[0] + i * (range1[1] - range1[0]) / (resolution - 1) for i in range(resolution)]
    ys = [range2[0] + i * (range2[1] - range2[0]) / (resolution - 1) for i in range(resolution)]

    x_tensor = torch.tensor([x], dtype=torch.float32)
    target_list = target

    z = [[0.0] * resolution for _ in range(resolution)]
    try:
        with torch.no_grad():
            for i, yv in enumerate(ys):
                _set_value(net, coord2["layer_index"], coord2["row"], coord2.get("col"), yv)
                for j, xv in enumerate(xs):
                    _set_value(net, coord1["layer_index"], coord1["row"], coord1.get("col"), xv)
                    output = net(x_tensor)
                    loss = compute_loss(output, target_list, loss_name)
                    z[i][j] = round(loss.item(), 6)
    finally:
        _set_value(net, coord1["layer_index"], coord1["row"], coord1.get("col"), original_1)
        _set_value(net, coord2["layer_index"], coord2["row"], coord2.get("col"), original_2)

    return {
        "coord1": coord1,
        "coord2": coord2,
        "original_point": [original_1, original_2],
        "x": xs,
        "y": ys,
        "z": z,
    }
