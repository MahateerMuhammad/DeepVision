import torch

from app.engine.ann.hooks import BackwardHookRecorder, ForwardHookRecorder
from app.engine.ann.loss import compute_loss
from app.engine.ann.model import DynamicANN
from app.engine.common.serialize import to_list, to_scalar


def _neuron_equations(
    layer_input: list[float],
    weights: list[list[float]],
    biases: list[float],
    pre_activation: list[float],
    post_activation: list[float],
    activation_name: str,
) -> list[dict]:
    """Build one LaTeX-renderable equation pair (linear + activation) per output neuron."""
    equations = []
    for j, (w_row, b_j, z_j, a_j) in enumerate(
        zip(weights, biases, pre_activation, post_activation)
    ):
        terms = " + ".join(f"({w:.4f})({x:.4f})" for w, x in zip(w_row, layer_input))
        linear_latex = f"z_{{{j}}} = {terms} + ({b_j:.4f}) = {z_j:.4f}"
        activation_latex = f"a_{{{j}}} = {activation_name}({z_j:.4f}) = {a_j:.4f}"
        equations.append(
            {
                "neuron": j,
                "linear_equation": linear_latex,
                "activation_equation": activation_latex,
            }
        )
    return equations


def _dropped_mask(post_activation: list[float], post_dropout: list[float]) -> list[float]:
    """1.0 where dropout observably zeroed an originally-nonzero activation,
    0.0 otherwise (including units that were already zero pre-dropout, e.g.
    a dead ReLU — those can't be distinguished from "dropped" by value alone,
    so they're reported as not-dropped rather than guessed at).
    """
    return [1.0 if (a != 0.0 and d == 0.0) else 0.0 for a, d in zip(post_activation, post_dropout)]


def build_forward_state_tree(net: DynamicANN, x: list[float], training: bool = True) -> dict:
    """Run a single-sample forward pass and capture every intermediate tensor
    into a JSON-serializable "state tree" the frontend can step through.

    `training=False` puts Dropout into eval mode (a no-op) for deterministic
    inspection; `training=True` (default) lets Dropout actually drop units,
    which is the point of the "Dropout Rain" visualization.
    """
    if len(x) != net.spec.input_size:
        raise ValueError(
            f"input length {len(x)} does not match network input_size {net.spec.input_size}"
        )

    net.train(training)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    with ForwardHookRecorder(net) as rec:
        output = net(x_tensor)

    layers = []
    for idx, (linear, layer_spec) in enumerate(zip(net.linears, net.spec.layers)):
        layer_input = to_list(rec.layer_inputs[idx])[0]
        weights = to_list(linear.weight)
        biases = to_list(linear.bias)
        pre_activation = to_list(rec.pre_activations[idx])[0]
        post_activation = to_list(rec.post_activations[idx])[0]
        post_dropout = to_list(rec.post_dropouts[idx])[0]

        layers.append(
            {
                "layer_index": idx,
                "in_features": linear.in_features,
                "out_features": linear.out_features,
                "activation": layer_spec.activation,
                "input": layer_input,
                "weights": weights,
                "biases": biases,
                "pre_activation": pre_activation,
                "post_activation": post_activation,
                "dropout_prob": layer_spec.dropout_prob,
                "post_dropout": post_dropout,
                "dropped_mask": _dropped_mask(post_activation, post_dropout),
                "equations": _neuron_equations(
                    layer_input, weights, biases, pre_activation, post_activation, layer_spec.activation
                ),
            }
        )

    return {
        "input": x,
        "training": training,
        "layers": layers,
        "output": to_list(output)[0],
    }


def build_backward_state_tree(
    net: DynamicANN, x: list[float], target: list[float], loss_name: str = "mse", training: bool = True
) -> dict:
    """Run a single-sample forward pass, compute loss, backpropagate, and
    capture every intermediate value AND local gradient into a JSON-safe tree.
    """
    if len(x) != net.spec.input_size:
        raise ValueError(
            f"input length {len(x)} does not match network input_size {net.spec.input_size}"
        )

    net.train(training)
    net.zero_grad(set_to_none=True)
    x_tensor = torch.tensor([x], dtype=torch.float32, requires_grad=True)

    # Backward hooks must be registered BEFORE forward() executes: they attach
    # to the autograd graph as it is built, and cannot be retrofitted onto an
    # already-completed forward pass.
    with ForwardHookRecorder(net) as fwd_rec, BackwardHookRecorder(net) as bwd_rec:
        output = net(x_tensor)
        loss = compute_loss(output, target, loss_name)
        loss.backward()

    layers = []
    for idx, (linear, layer_spec) in enumerate(zip(net.linears, net.spec.layers)):
        layer_input = to_list(fwd_rec.layer_inputs[idx])[0]
        weights = to_list(linear.weight)
        biases = to_list(linear.bias)
        pre_activation = to_list(fwd_rec.pre_activations[idx])[0]
        post_activation = to_list(fwd_rec.post_activations[idx])[0]
        post_dropout = to_list(fwd_rec.post_dropouts[idx])[0]

        grad_layer_input = bwd_rec.grad_layer_input[idx]
        grad_post_dropout = bwd_rec.grad_post_dropout[idx]

        layers.append(
            {
                "layer_index": idx,
                "in_features": linear.in_features,
                "out_features": linear.out_features,
                "activation": layer_spec.activation,
                "input": layer_input,
                "weights": weights,
                "biases": biases,
                "pre_activation": pre_activation,
                "post_activation": post_activation,
                "dropout_prob": layer_spec.dropout_prob,
                "post_dropout": post_dropout,
                "dropped_mask": _dropped_mask(post_activation, post_dropout),
                "equations": _neuron_equations(
                    layer_input, weights, biases, pre_activation, post_activation, layer_spec.activation
                ),
                "grad_weights": to_list(linear.weight.grad),
                "grad_biases": to_list(linear.bias.grad),
                "grad_pre_activation": to_list(bwd_rec.grad_pre_activation[idx])[0],
                "grad_post_activation": to_list(bwd_rec.grad_post_activation[idx])[0],
                "grad_post_dropout": to_list(grad_post_dropout)[0] if grad_post_dropout is not None else None,
                "grad_input": to_list(grad_layer_input)[0] if grad_layer_input is not None else None,
            }
        )

    return {
        "input": x,
        "target": target,
        "loss_name": loss_name,
        "training": training,
        "loss": to_scalar(loss),
        "layers": layers,
        "output": to_list(output)[0],
    }
