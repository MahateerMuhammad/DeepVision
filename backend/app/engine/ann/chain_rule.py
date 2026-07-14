from app.engine.ann.model import DynamicANN
from app.engine.ann.state_tree import build_backward_state_tree

_MATCH_TOLERANCE = 1e-3


def _local_activation_derivative(activation_name: str, z: float, a: float) -> float:
    """da/dz at the point (z, a) already computed by the forward pass."""
    if activation_name == "relu":
        return 1.0 if z > 0 else 0.0
    if activation_name == "leaky_relu":
        return 1.0 if z > 0 else 0.01
    if activation_name == "sigmoid":
        return a * (1 - a)
    if activation_name == "tanh":
        return 1 - a * a
    if activation_name == "linear":
        return 1.0
    raise ValueError(f"unknown activation: {activation_name}")


def trace_weight_gradient(
    net: DynamicANN,
    x: list[float],
    target: list[float],
    loss_name: str,
    layer_index: int,
    weight_row: int,
    weight_col: int | None = None,
) -> dict:
    """Isolate one parameter and reconstruct the exact chain-rule factors that
    produce its gradient: ∂L/∂param = (∂L/∂a) · (∂a/∂z) · (∂z/∂param).

    ∂L/∂a is read directly from the captured backward pass, which is already
    the fully-aggregated, autograd-correct value (it legitimately sums over
    every downstream path when a neuron fans out to multiple next-layer
    neurons that summation IS the multivariate chain rule, not an
    approximation of it). For pedagogy, when this layer isn't the last one,
    we additionally expand that sum into its per-downstream-neuron terms.
    """
    tree = build_backward_state_tree(net, x, target, loss_name)
    layers = tree["layers"]

    if not (0 <= layer_index < len(layers)):
        raise ValueError(f"layer_index {layer_index} out of range [0, {len(layers) - 1}]")

    layer = layers[layer_index]
    j = weight_row
    if not (0 <= j < layer["out_features"]):
        raise ValueError(f"weight_row {j} out of range [0, {layer['out_features'] - 1}]")

    if weight_col is None:
        param_kind = "bias"
        param_symbol = f"b_{{{j}}}^{{({layer_index})}}"
        dz_dparam = 1.0
        analytic_grad = layer["grad_biases"][j]
    else:
        k = weight_col
        if not (0 <= k < layer["in_features"]):
            raise ValueError(f"weight_col {k} out of range [0, {layer['in_features'] - 1}]")
        param_kind = "weight"
        param_symbol = f"w_{{{j},{k}}}^{{({layer_index})}}"
        dz_dparam = layer["input"][k]
        analytic_grad = layer["grad_weights"][j][k]

    z_j = layer["pre_activation"][j]
    a_j = layer["post_activation"][j]
    activation_name = layer["activation"]
    da_dz = _local_activation_derivative(activation_name, z_j, a_j)
    dL_da = layer["grad_post_activation"][j]

    chain_product = dL_da * da_dz * dz_dparam

    chain_steps = [
        {
            "name": "loss_to_postactivation",
            "latex": f"\\frac{{\\partial L}}{{\\partial a_{{{j}}}^{{({layer_index})}}}} = {dL_da:.4f}",
            "value": round(dL_da, 6),
        },
        {
            "name": "postactivation_to_preactivation",
            "latex": (
                f"\\frac{{\\partial a_{{{j}}}^{{({layer_index})}}}}"
                f"{{\\partial z_{{{j}}}^{{({layer_index})}}}} = "
                f"{activation_name}'({z_j:.4f}) = {da_dz:.4f}"
            ),
            "value": round(da_dz, 6),
        },
        {
            "name": "preactivation_to_param",
            "latex": (
                f"\\frac{{\\partial z_{{{j}}}^{{({layer_index})}}}}"
                f"{{\\partial {param_symbol}}} = {dz_dparam:.4f}"
            ),
            "value": round(dz_dparam, 6),
        },
    ]

    downstream_breakdown = None
    # The per-term reconstruction below uses next_layer's raw weights, which
    # is only exact when nothing sits between a_j^(L) and z^(L+1) — i.e. no
    # dropout. With dropout active, the aggregated dL/da_j in chain_steps[0]
    # is still exactly correct (it comes straight from autograd), but this
    # illustrative per-neuron decomposition would silently ignore the
    # dropout mask/scale, so it's omitted rather than shown wrong.
    if layer_index + 1 < len(layers) and layer["dropout_prob"] == 0.0:
        next_layer = layers[layer_index + 1]
        next_grad_pre = next_layer["grad_pre_activation"]
        terms = []
        running_sum = 0.0
        for m, w_row in enumerate(next_layer["weights"]):
            w_mj = w_row[j]
            contribution = next_grad_pre[m] * w_mj
            running_sum += contribution
            terms.append(
                {
                    "downstream_neuron": m,
                    "latex": (
                        f"\\frac{{\\partial L}}{{\\partial z_{{{m}}}^{{({layer_index + 1})}}}} "
                        f"\\cdot w_{{{m},{j}}}^{{({layer_index + 1})}} = "
                        f"({next_grad_pre[m]:.4f})({w_mj:.4f}) = {contribution:.4f}"
                    ),
                    "value": round(contribution, 6),
                }
            )
        downstream_breakdown = {
            "explanation": (
                f"\\frac{{\\partial L}}{{\\partial a_{{{j}}}^{{({layer_index})}}}} = "
                f"\\sum_m \\frac{{\\partial L}}{{\\partial z_m^{{({layer_index + 1})}}}} "
                f"\\cdot w_{{m,{j}}}^{{({layer_index + 1})}}"
            ),
            "terms": terms,
            "sum": round(running_sum, 6),
        }

    return {
        "layer_index": layer_index,
        "neuron": j,
        "param_kind": param_kind,
        "row": j,
        "col": weight_col,
        "chain_steps": chain_steps,
        "chain_product": round(chain_product, 6),
        "analytic_gradient": analytic_grad,
        "matches_analytic": abs(chain_product - analytic_grad) < _MATCH_TOLERANCE,
        "downstream_breakdown": downstream_breakdown,
    }
