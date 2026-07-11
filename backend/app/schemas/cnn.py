from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

Activation = Literal["relu", "sigmoid", "tanh", "linear", "leaky_relu"]
PoolKind = Literal["max", "avg"]


class ConvLayerSpec(BaseModel):
    kind: Literal["conv"] = "conv"
    out_channels: int = Field(gt=0)
    kernel_size: int = Field(gt=0)
    stride: int = Field(default=1, gt=0)
    padding: int = Field(default=0, ge=0)
    activation: Activation = "relu"


class PoolLayerSpec(BaseModel):
    kind: Literal["pool"] = "pool"
    pool_type: PoolKind = "max"
    kernel_size: int = Field(gt=0)
    stride: Optional[int] = Field(default=None, gt=0)

    @field_validator("stride")
    @classmethod
    def _default_stride_equals_kernel(cls, v, info):
        return v


CNNLayerSpec = Annotated[Union[ConvLayerSpec, PoolLayerSpec], Field(discriminator="kind")]


class CNNSpec(BaseModel):
    input_channels: int = Field(gt=0)
    input_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    layers: list[CNNLayerSpec] = Field(min_length=1)
    num_classes: Optional[int] = Field(default=None, gt=0)
    seed: Optional[int] = None


class CreateCNNRequest(BaseModel):
    spec: CNNSpec


class CreateCNNResponse(BaseModel):
    network_id: str
    spec: CNNSpec
    param_count: int
    output_shape: list[int]  # [C, H, W] after the conv/pool stack, pre-classifier-head


class CNNForwardRequest(BaseModel):
    network_id: str
    image: list[list[list[float]]]  # (C, H, W)


class SlidingKernelRequest(BaseModel):
    """Standalone 'Filter Factory': apply one hand-authored kernel to a 2D
    grayscale image, no trained network involved."""

    image: list[list[float]]  # (H, W)
    kernel: list[list[float]]  # (kH, kW)
    stride: int = Field(default=1, gt=0)
    padding: int = Field(default=0, ge=0)


class ReceptiveFieldRequest(BaseModel):
    network_id: str
    layer_index: int  # index into spec.layers, must be a conv/pool layer
    channel: int
    row: int
    col: int


class SaliencyRequest(BaseModel):
    network_id: str
    image: list[list[list[float]]]
    target_class: int
