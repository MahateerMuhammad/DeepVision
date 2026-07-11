import torch

from app.engine.cnn.hooks import CNNForwardHookRecorder
from app.engine.cnn.model import DynamicCNN
from app.engine.common.serialize import to_list


def validate_image(net: DynamicCNN, image: list[list[list[float]]]) -> torch.Tensor:
    c, h, w = net.spec.input_channels, net.spec.input_height, net.spec.input_width
    if len(image) != c:
        raise ValueError(f"image has {len(image)} channels, network expects {c}")
    for ch in image:
        if len(ch) != h:
            raise ValueError(f"image channel has height {len(ch)}, network expects {h}")
        for row in ch:
            if len(row) != w:
                raise ValueError(f"image row has width {len(row)}, network expects {w}")
    return torch.tensor([image], dtype=torch.float32)


def _kept_mask(indices: torch.Tensor, in_h: int, in_w: int) -> list:
    """For a max-pool stage, build a same-shape-as-input (C,H,W) 0/1 mask
    marking which input pixels survived (were the argmax) vs were discarded.
    """
    c = indices.shape[1]
    mask = torch.zeros(c, in_h, in_w)
    flat_indices = indices[0]  # (C, out_h, out_w), values are flat row-major indices into H*W
    for ch in range(c):
        for flat_idx in flat_indices[ch].flatten().tolist():
            row, col = divmod(int(flat_idx), in_w)
            mask[ch, row, col] = 1.0
    return to_list(mask)


def build_cnn_forward_state_tree(net: DynamicCNN, image: list[list[list[float]]]) -> dict:
    x_tensor = validate_image(net, image)

    with CNNForwardHookRecorder(net) as rec:
        output = net(x_tensor)

    layers = []
    for idx, (stage, layer_spec, shapes) in enumerate(
        zip(net.stages, net.spec.layers, net.stage_shapes)
    ):
        in_shape = list(shapes["in"])
        out_shape = list(shapes["out"])

        if layer_spec.kind == "conv":
            conv = stage["conv"]
            layers.append(
                {
                    "layer_index": idx,
                    "kind": "conv",
                    "activation": layer_spec.activation,
                    "kernel_size": layer_spec.kernel_size,
                    "stride": layer_spec.stride,
                    "padding": layer_spec.padding,
                    "in_shape": in_shape,
                    "out_shape": out_shape,
                    "input": to_list(rec.stage_inputs[idx])[0],
                    "weights": to_list(conv.weight),
                    "biases": to_list(conv.bias),
                    "pre_activation": to_list(rec.pre_activations[idx])[0],
                    "post_activation": to_list(rec.stage_outputs[idx])[0],
                }
            )
        else:
            stride = layer_spec.stride or layer_spec.kernel_size
            entry = {
                "layer_index": idx,
                "kind": "pool",
                "pool_type": layer_spec.pool_type,
                "kernel_size": layer_spec.kernel_size,
                "stride": stride,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "input": to_list(rec.stage_inputs[idx])[0],
                "output": to_list(rec.stage_outputs[idx])[0],
            }
            if rec.pool_indices[idx] is not None:
                entry["kept_mask"] = _kept_mask(rec.pool_indices[idx], in_shape[1], in_shape[2])
            layers.append(entry)

    return {
        "input": image,
        "layers": layers,
        "output": to_list(output)[0],
    }
