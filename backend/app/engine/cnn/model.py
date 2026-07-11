import torch
import torch.nn as nn

from app.schemas.cnn import CNNSpec

_ACTIVATION_FACTORY = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
}


def _conv_out_dim(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def _pool_out_dim(size: int, kernel: int, stride: int) -> int:
    return (size - kernel) // stride + 1


class DynamicCNN(nn.Module):
    """CNN assembled from a CNNSpec: a stack of conv(+activation) and pool
    stages, with an optional linear classifier head.

    Each stage is one nn.ModuleDict so that hooks can be attached per stage
    and state-tree indices line up 1:1 with `spec.layers`.
    """

    def __init__(self, spec: CNNSpec):
        super().__init__()
        self.spec = spec
        if spec.seed is not None:
            torch.manual_seed(spec.seed)

        self.stages = nn.ModuleList()
        self.stage_shapes: list[dict] = []  # computed (in/out) C,H,W per stage, for validation & receptive field math

        c, h, w = spec.input_channels, spec.input_height, spec.input_width

        for idx, layer in enumerate(spec.layers):
            if layer.kind == "conv":
                if layer.kernel_size > h + 2 * layer.padding or layer.kernel_size > w + 2 * layer.padding:
                    raise ValueError(
                        f"layer {idx}: kernel_size {layer.kernel_size} is larger than the "
                        f"padded input ({h + 2 * layer.padding}x{w + 2 * layer.padding})"
                    )
                conv = nn.Conv2d(
                    in_channels=c,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
                act = _ACTIVATION_FACTORY[layer.activation]()
                stage = nn.ModuleDict({"conv": conv, "act": act})

                out_h = _conv_out_dim(h, layer.kernel_size, layer.stride, layer.padding)
                out_w = _conv_out_dim(w, layer.kernel_size, layer.stride, layer.padding)
                out_c = layer.out_channels
            else:  # pool
                stride = layer.stride or layer.kernel_size
                if layer.kernel_size > h or layer.kernel_size > w:
                    raise ValueError(
                        f"layer {idx}: pool kernel_size {layer.kernel_size} is larger than "
                        f"input ({h}x{w})"
                    )
                if layer.pool_type == "max":
                    pool = nn.MaxPool2d(layer.kernel_size, stride=stride, return_indices=True)
                else:
                    pool = nn.AvgPool2d(layer.kernel_size, stride=stride)
                stage = nn.ModuleDict({"pool": pool})

                out_h = _pool_out_dim(h, layer.kernel_size, stride)
                out_w = _pool_out_dim(w, layer.kernel_size, stride)
                out_c = c

            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"layer {idx}: output spatial size collapsed to non-positive "
                    f"({out_c}x{out_h}x{out_w})"
                )

            self.stages.append(stage)
            self.stage_shapes.append(
                {"in": (c, h, w), "out": (out_c, out_h, out_w), "kind": layer.kind}
            )
            c, h, w = out_c, out_h, out_w

        self.output_shape = (c, h, w)

        self.classifier: nn.Module | None = None
        if spec.num_classes is not None:
            self.classifier = nn.Linear(c * h * w, spec.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            if "conv" in stage:
                x = stage["act"](stage["conv"](x))
            else:
                pool = stage["pool"]
                if isinstance(pool, nn.MaxPool2d):
                    x, _indices = pool(x)
                else:
                    x = pool(x)
        if self.classifier is not None:
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_cnn(spec: CNNSpec) -> DynamicCNN:
    return DynamicCNN(spec)
