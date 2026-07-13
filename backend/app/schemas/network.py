from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Activation = Literal["relu", "sigmoid", "tanh", "linear", "leaky_relu"]
LossFn = Literal["mse", "cross_entropy", "bce"]


class LayerSpec(BaseModel):
    """One fully-connected layer: Linear(in, out) -> activation -> optional Dropout."""

    out_features: int = Field(gt=0)
    activation: Activation = "relu"
    dropout_prob: float = Field(default=0.0, ge=0.0, lt=1.0)


class NetworkSpec(BaseModel):
    """Full architecture spec for a feedforward ANN."""

    input_size: int = Field(gt=0)
    layers: list[LayerSpec] = Field(min_length=1)
    seed: Optional[int] = None

    @field_validator("layers")
    @classmethod
    def _non_empty(cls, v: list[LayerSpec]) -> list[LayerSpec]:
        if not v:
            raise ValueError("network must have at least one layer")
        return v


class CreateNetworkRequest(BaseModel):
    spec: NetworkSpec


class CreateNetworkResponse(BaseModel):
    network_id: str
    spec: NetworkSpec
    param_count: int


class ForwardPassRequest(BaseModel):
    network_id: str
    input: list[float]
    training: bool = True  # False forces eval mode: Dropout becomes a no-op


class BackwardPassRequest(BaseModel):
    network_id: str
    input: list[float]
    target: list[float]
    loss: LossFn = "mse"
    training: bool = True


class StepRequest(BaseModel):
    network_id: str
    input: list[float]
    target: list[float]
    loss: LossFn = "mse"
    learning_rate: float = Field(default=0.1, gt=0.0)
    num_steps: int = Field(default=1, ge=1, le=500)
    training: bool = True


class TraceRequest(BaseModel):
    network_id: str
    input: list[float]
    target: list[float]
    loss: LossFn = "mse"
    layer_index: int
    weight_row: int
    weight_col: Optional[int] = None  # None => trace a bias term
