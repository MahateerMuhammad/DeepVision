import torch

from app.engine.cnn.model import DynamicCNN


class CNNForwardHookRecorder:
    """Registers forward hooks on every stage (conv+activation, or pool) of a
    DynamicCNN and records the tensors flowing through them, indexed by
    stage/layer position so results line up with `net.spec.layers`.
    """

    def __init__(self, net: DynamicCNN):
        self.net = net
        n = len(net.stages)
        self.stage_inputs: list[torch.Tensor | None] = [None] * n
        self.pre_activations: list[torch.Tensor | None] = [None] * n  # conv stages only
        self.stage_outputs: list[torch.Tensor | None] = [None] * n
        self.pool_indices: list[torch.Tensor | None] = [None] * n  # max-pool stages only
        self._handles = []

    def __enter__(self) -> "CNNForwardHookRecorder":
        for idx, stage in enumerate(self.net.stages):
            if "conv" in stage:
                self._handles.append(stage["conv"].register_forward_hook(self._make_conv_hook(idx)))
                self._handles.append(stage["act"].register_forward_hook(self._make_act_hook(idx)))
            else:
                self._handles.append(stage["pool"].register_forward_hook(self._make_pool_hook(idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_conv_hook(self, idx: int):
        def hook(module, inputs, output):
            self.stage_inputs[idx] = inputs[0].detach().clone()
            self.pre_activations[idx] = output.detach().clone()

        return hook

    def _make_act_hook(self, idx: int):
        def hook(module, inputs, output):
            self.stage_outputs[idx] = output.detach().clone()

        return hook

    def _make_pool_hook(self, idx: int):
        def hook(module, inputs, output):
            self.stage_inputs[idx] = inputs[0].detach().clone()
            if isinstance(output, tuple):  # MaxPool2d(return_indices=True)
                out_tensor, indices = output
                self.stage_outputs[idx] = out_tensor.detach().clone()
                self.pool_indices[idx] = indices.detach().clone()
            else:  # AvgPool2d
                self.stage_outputs[idx] = output.detach().clone()

        return hook
