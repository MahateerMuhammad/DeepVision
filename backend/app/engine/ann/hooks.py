import torch

from app.engine.ann.model import DynamicANN


class ForwardHookRecorder:
    """Registers forward hooks on every Linear/Activation/Dropout submodule of
    a DynamicANN and records the tensors flowing through them.

    PyTorch guarantees hooks fire in the order modules are executed, and a
    DynamicANN always executes linear_i, activation_i, dropout_i in lockstep,
    so the recorded lists line up index-for-index with `net.linears` /
    `net.activations` / `net.dropouts`.
    """

    def __init__(self, net: DynamicANN):
        self.net = net
        self.layer_inputs: list[torch.Tensor] = []
        self.pre_activations: list[torch.Tensor] = []
        self.post_activations: list[torch.Tensor] = []
        self.post_dropouts: list[torch.Tensor] = []
        self._handles = []

    def __enter__(self) -> "ForwardHookRecorder":
        for linear in self.net.linears:
            self._handles.append(linear.register_forward_hook(self._on_linear))
        for activation in self.net.activations:
            self._handles.append(activation.register_forward_hook(self._on_activation))
        for dropout in self.net.dropouts:
            self._handles.append(dropout.register_forward_hook(self._on_dropout))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _on_linear(self, module, inputs, output) -> None:
        self.layer_inputs.append(inputs[0].detach().clone())
        self.pre_activations.append(output.detach().clone())

    def _on_activation(self, module, inputs, output) -> None:
        self.post_activations.append(output.detach().clone())

    def _on_dropout(self, module, inputs, output) -> None:
        self.post_dropouts.append(output.detach().clone())


class BackwardHookRecorder:
    """Registers full backward hooks on every Linear/Activation submodule and
    records the local gradient tensors flowing through them during
    loss.backward().

    Backward hooks fire in *reverse* execution order (last layer first), so
    results are written into index-addressed slots (one per layer) rather
    than appended, keeping alignment with `net.linears` regardless of firing
    order.
    """

    def __init__(self, net: DynamicANN):
        self.net = net
        n = len(net.linears)
        self.grad_layer_input: list[torch.Tensor | None] = [None] * n
        self.grad_pre_activation: list[torch.Tensor | None] = [None] * n
        self.grad_post_activation: list[torch.Tensor | None] = [None] * n
        self.grad_post_dropout: list[torch.Tensor | None] = [None] * n
        self._handles = []

    def __enter__(self) -> "BackwardHookRecorder":
        for idx, linear in enumerate(self.net.linears):
            self._handles.append(linear.register_full_backward_hook(self._make_linear_hook(idx)))
        for idx, activation in enumerate(self.net.activations):
            self._handles.append(
                activation.register_full_backward_hook(self._make_activation_hook(idx))
            )
        for idx, dropout in enumerate(self.net.dropouts):
            self._handles.append(dropout.register_full_backward_hook(self._make_dropout_hook(idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_linear_hook(self, idx: int):
        def hook(module, grad_input, grad_output):
            if grad_input[0] is not None:
                self.grad_layer_input[idx] = grad_input[0].detach().clone()

        return hook

    def _make_activation_hook(self, idx: int):
        def hook(module, grad_input, grad_output):
            self.grad_pre_activation[idx] = grad_input[0].detach().clone()
            self.grad_post_activation[idx] = grad_output[0].detach().clone()

        return hook

    def _make_dropout_hook(self, idx: int):
        def hook(module, grad_input, grad_output):
            self.grad_post_dropout[idx] = grad_output[0].detach().clone()

        return hook
