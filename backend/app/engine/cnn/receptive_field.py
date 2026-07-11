from app.engine.cnn.model import DynamicCNN


def compute_receptive_field(net: DynamicCNN, layer_index: int, channel: int, row: int, col: int) -> dict:
    """Trace one output pixel backward through every preceding conv/pool
    stage to find the rectangular patch of the original input image that
    influenced it — a purely geometric (kernel/stride/padding) calculation,
    exact regardless of the actual weight values.

    For a single output position, the corresponding input range at that
    stage is [pos*stride - padding, pos*stride - padding + kernel_size - 1].
    Applying this repeatedly, stage by stage, from `layer_index` back to the
    input accumulates the full receptive field (pooling has no padding).
    """
    if not (0 <= layer_index < len(net.spec.layers)):
        raise ValueError(f"layer_index {layer_index} out of range [0, {len(net.spec.layers) - 1}]")

    out_c, out_h, out_w = net.stage_shapes[layer_index]["out"]
    if not (0 <= channel < out_c):
        raise ValueError(f"channel {channel} out of range [0, {out_c - 1}]")
    if not (0 <= row < out_h):
        raise ValueError(f"row {row} out of range [0, {out_h - 1}]")
    if not (0 <= col < out_w):
        raise ValueError(f"col {col} out of range [0, {out_w - 1}]")

    row_start, row_end = row, row
    col_start, col_end = col, col

    for idx in range(layer_index, -1, -1):
        layer = net.spec.layers[idx]
        if layer.kind == "conv":
            k, s, p = layer.kernel_size, layer.stride, layer.padding
        else:
            k = layer.kernel_size
            s = layer.stride or layer.kernel_size
            p = 0

        row_start = row_start * s - p
        row_end = row_end * s - p + k - 1
        col_start = col_start * s - p
        col_end = col_end * s - p + k - 1

    in_h, in_w = net.spec.input_height, net.spec.input_width
    clipped_row_start = max(0, row_start)
    clipped_row_end = min(in_h - 1, row_end)
    clipped_col_start = max(0, col_start)
    clipped_col_end = min(in_w - 1, col_end)

    return {
        "layer_index": layer_index,
        "channel": channel,
        "row": row,
        "col": col,
        "raw_row_range": [row_start, row_end],
        "raw_col_range": [col_start, col_end],
        "clipped_row_range": [clipped_row_start, clipped_row_end],
        "clipped_col_range": [clipped_col_start, clipped_col_end],
        "receptive_field_size": [row_end - row_start + 1, col_end - col_start + 1],
    }
