import torch
import torch.nn as nn

from app.schemas.network import NetworkSpec

_ACTIVATION_FACTORY = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
}


class DynamicANN(nn.Module):
    """Feedforward network assembled from a NetworkSpec.

    Linear, activation, and dropout are kept as separate submodules (rather
    than fused into one call) so that forward/backward hooks can
    independently observe the pre-activation, post-activation, and
    post-dropout tensors at every layer.

    A Dropout module is always present, even when dropout_prob is 0 —
    nn.Dropout(p=0) is an exact no-op passthrough in both train and eval
    mode, so this keeps the module list structure (and hook indexing)
    uniform without any conditional branching in forward().
    """

    def __init__(self, spec: NetworkSpec):
        super().__init__()
        self.spec = spec
        if spec.seed is not None:
            torch.manual_seed(spec.seed)

        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = spec.input_size
        for layer in spec.layers:
            self.linears.append(nn.Linear(in_features, layer.out_features))
            self.activations.append(_ACTIVATION_FACTORY[layer.activation]())
            self.dropouts.append(nn.Dropout(p=layer.dropout_prob))
            in_features = layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, activation, dropout in zip(self.linears, self.activations, self.dropouts):
            x = linear(x)
            x = activation(x)
            x = dropout(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_network(spec: NetworkSpec) -> DynamicANN:
    return DynamicANN(spec)
