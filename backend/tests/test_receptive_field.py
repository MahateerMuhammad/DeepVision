import pytest
import torch

from app.engine.cnn.model import build_cnn
from app.engine.cnn.receptive_field import compute_receptive_field
from app.schemas.cnn import CNNSpec, ConvLayerSpec, PoolLayerSpec


def test_single_conv_layer_corner_pixel():
    spec = CNNSpec(
        input_channels=1, input_height=5, input_width=5,
        layers=[ConvLayerSpec(out_channels=1, kernel_size=3, stride=1, padding=0)],
    )
    net = build_cnn(spec)
    rf = compute_receptive_field(net, layer_index=0, channel=0, row=0, col=0)
    assert rf["clipped_row_range"] == [0, 2]
    assert rf["clipped_col_range"] == [0, 2]
    assert rf["receptive_field_size"] == [3, 3]


def test_single_conv_layer_last_pixel():
    spec = CNNSpec(
        input_channels=1, input_height=5, input_width=5,
        layers=[ConvLayerSpec(out_channels=1, kernel_size=3, stride=1, padding=0)],
    )
    net = build_cnn(spec)
    # output is 3x3, last valid position (2,2)
    rf = compute_receptive_field(net, layer_index=0, channel=0, row=2, col=2)
    assert rf["clipped_row_range"] == [2, 4]
    assert rf["clipped_col_range"] == [2, 4]


def test_two_stacked_3x3_convs_yield_5x5_receptive_field():
    spec = CNNSpec(
        input_channels=1, input_height=9, input_width=9,
        layers=[
            ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=0),
            ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=0),
        ],
    )
    net = build_cnn(spec)
    rf = compute_receptive_field(net, layer_index=1, channel=0, row=0, col=0)
    assert rf["receptive_field_size"] == [5, 5]
    assert rf["clipped_row_range"] == [0, 4]
    assert rf["clipped_col_range"] == [0, 4]


def test_conv_then_maxpool_doubles_effective_stride():
    spec = CNNSpec(
        input_channels=1, input_height=8, input_width=8,
        layers=[
            ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=1),  # -> 2x8x8, RF=3x3
            PoolLayerSpec(pool_type="max", kernel_size=2),  # -> 2x4x4
        ],
    )
    net = build_cnn(spec)
    rf = compute_receptive_field(net, layer_index=1, channel=0, row=0, col=0)
    # pool output (0,0) <- conv output rows/cols [0,1] <- (with padding=1 conv) input rows [-1,2]
    assert rf["raw_row_range"] == [-1, 2]
    assert rf["clipped_row_range"] == [0, 2]


def test_receptive_field_bounding_box_matches_empirical_gradient_for_conv_only_stack():
    """Cross-check the pure geometric formula against ground truth: backprop
    a single output scalar to the input image and confirm the nonzero-gradient
    footprint's bounding box matches the analytically computed receptive field.
    """
    torch.manual_seed(0)
    spec = CNNSpec(
        input_channels=1, input_height=9, input_width=9, seed=0,
        layers=[
            ConvLayerSpec(out_channels=2, kernel_size=3, stride=1, padding=0, activation="linear"),
            ConvLayerSpec(out_channels=2, kernel_size=3, stride=2, padding=0, activation="linear"),
        ],
    )
    net = build_cnn(spec)
    x = torch.rand(1, 1, 9, 9, requires_grad=True)
    y = net(x)  # shape (1, 2, 2, 2)  since (9-3+1)=7, (7-3)//2+1=3 -> recompute below

    target_row, target_col, target_channel = 0, 0, 0
    scalar = y[0, target_channel, target_row, target_col]
    scalar.backward()

    grad = x.grad[0, 0]
    nonzero = (grad.abs() > 1e-12).nonzero()
    emp_row_min, emp_row_max = nonzero[:, 0].min().item(), nonzero[:, 0].max().item()
    emp_col_min, emp_col_max = nonzero[:, 1].min().item(), nonzero[:, 1].max().item()

    rf = compute_receptive_field(net, layer_index=1, channel=target_channel, row=target_row, col=target_col)

    assert [emp_row_min, emp_row_max] == rf["clipped_row_range"]
    assert [emp_col_min, emp_col_max] == rf["clipped_col_range"]


def test_out_of_range_layer_raises():
    spec = CNNSpec(input_channels=1, input_height=5, input_width=5, layers=[ConvLayerSpec(out_channels=1, kernel_size=3)])
    net = build_cnn(spec)
    with pytest.raises(ValueError):
        compute_receptive_field(net, layer_index=9, channel=0, row=0, col=0)


def test_out_of_range_row_raises():
    spec = CNNSpec(input_channels=1, input_height=5, input_width=5, layers=[ConvLayerSpec(out_channels=1, kernel_size=3)])
    net = build_cnn(spec)
    with pytest.raises(ValueError):
        compute_receptive_field(net, layer_index=0, channel=0, row=99, col=0)


def test_out_of_range_channel_raises():
    spec = CNNSpec(input_channels=1, input_height=5, input_width=5, layers=[ConvLayerSpec(out_channels=2, kernel_size=3)])
    net = build_cnn(spec)
    with pytest.raises(ValueError):
        compute_receptive_field(net, layer_index=0, channel=5, row=0, col=0)
