import torch

from app.engine.ann.model import build_network
from app.schemas.network import LayerSpec, NetworkSpec


def _spec(seed=42):
    return NetworkSpec(
        input_size=3,
        seed=seed,
        layers=[
            LayerSpec(out_features=4, activation="relu"),
            LayerSpec(out_features=2, activation="sigmoid"),
        ],
    )


def test_architecture_shapes():
    net = build_network(_spec())
    assert len(net.linears) == 2
    assert net.linears[0].in_features == 3
    assert net.linears[0].out_features == 4
    assert net.linears[1].in_features == 4
    assert net.linears[1].out_features == 2


def test_forward_pass_output_shape_and_range():
    net = build_network(_spec())
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = net(x)
    assert y.shape == (1, 2)
    # sigmoid output must be in (0, 1)
    assert torch.all(y > 0) and torch.all(y < 1)


def test_seed_is_reproducible():
    net_a = build_network(_spec(seed=7))
    net_b = build_network(_spec(seed=7))
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(pa, pb)


def test_different_seed_differs():
    net_a = build_network(_spec(seed=1))
    net_b = build_network(_spec(seed=2))
    diffs = [not torch.equal(pa, pb) for pa, pb in zip(net_a.parameters(), net_b.parameters())]
    assert any(diffs)


def test_manual_forward_matches_module():
    net = build_network(_spec())
    x = torch.tensor([[1.0, -2.0, 0.5]])

    w0, b0 = net.linears[0].weight, net.linears[0].bias
    w1, b1 = net.linears[1].weight, net.linears[1].bias

    h_pre = x @ w0.T + b0
    h_post = torch.relu(h_pre)
    o_pre = h_post @ w1.T + b1
    o_post = torch.sigmoid(o_pre)

    y = net(x)
    assert torch.allclose(y, o_post, atol=1e-6)


def test_param_count():
    net = build_network(_spec())
    # layer0: (3*4 + 4) = 16, layer1: (4*2 + 2) = 10 -> 26
    assert net.param_count() == 26
