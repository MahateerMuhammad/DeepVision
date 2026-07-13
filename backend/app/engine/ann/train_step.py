import torch

from app.engine.ann.loss import compute_loss
from app.engine.ann.model import DynamicANN
from app.engine.ann.state_tree import build_backward_state_tree
from app.engine.common.serialize import to_scalar


def apply_gradient_steps(
    net: DynamicANN,
    x: list[float],
    target: list[float],
    loss_name: str = "mse",
    learning_rate: float = 0.1,
    num_steps: int = 1,
    training: bool = True,
) -> dict:
    """Apply `num_steps` plain-SGD updates to the network's parameters in place,
    recording the loss before each update, then return the full backward state
    tree at the new weights.

    This mutates the registered network, so a subsequent forward/backward call
    on the same network_id reflects the learned weights — which is what lets the
    Network Canvas "watch the graph learn" across repeated steps.

    loss_history has length num_steps + 1: the loss measured before each of the
    `num_steps` updates, plus the loss after the final update (so it plots as a
    complete descent curve).
    """
    if len(x) != net.spec.input_size:
        raise ValueError(
            f"input length {len(x)} does not match network input_size {net.spec.input_size}"
        )
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    net.train(training)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    loss_history: list[float] = []
    for _ in range(num_steps):
        net.zero_grad(set_to_none=True)
        output = net(x_tensor)
        loss = compute_loss(output, target, loss_name)
        loss.backward()
        loss_history.append(to_scalar(loss))
        with torch.no_grad():
            for param in net.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad

    # Fresh forward/backward at the updated weights: this is the state the
    # canvas will render, and its loss closes out the descent curve.
    state_tree = build_backward_state_tree(net, x, target, loss_name, training)
    loss_history.append(state_tree["loss"])

    return {
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "loss_history": loss_history,
        "state_tree": state_tree,
    }
