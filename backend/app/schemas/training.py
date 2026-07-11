from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.network import LossFn

SurfaceName = Literal["bowl", "saddle", "rosenbrock"]
OptimizerName = Literal["sgd", "rmsprop", "adam"]


class LossSurfaceGridRequest(BaseModel):
    surface: SurfaceName
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: int = Field(default=40, ge=2, le=200)


class RacerSpec(BaseModel):
    label: str
    optimizer: OptimizerName
    lr: float = Field(gt=0)
    hyperparams: dict = Field(default_factory=dict)


class OptimizerRaceRequest(BaseModel):
    surface: SurfaceName
    start: tuple[float, float]
    racers: list[RacerSpec] = Field(min_length=1)
    num_steps: int = Field(gt=0, le=5000)


class WeightCoord(BaseModel):
    layer_index: int
    row: int
    col: Optional[int] = None  # None => bias term


class WeightLossSurfaceRequest(BaseModel):
    network_id: str
    input: list[float]
    target: list[float]
    loss: LossFn = "mse"
    coord1: WeightCoord
    coord2: WeightCoord
    range1: tuple[float, float]
    range2: tuple[float, float]
    resolution: int = Field(default=25, ge=2, le=200)


class BatchNormRequest(BaseModel):
    batch: list[list[float]]
    gamma: Optional[list[float]] = None
    beta: Optional[list[float]] = None
    eps: float = Field(default=1e-5, gt=0)
