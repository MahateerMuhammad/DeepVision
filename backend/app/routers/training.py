from fastapi import APIRouter, HTTPException

from app.core.ann_registry import ann_registry
from app.engine.training.batchnorm import compute_batchnorm_trace
from app.engine.training.loss_surfaces import generate_grid
from app.engine.training.optimizer_race import run_race
from app.engine.training.weight_loss_surface import compute_weight_loss_surface
from app.schemas.training import (
    BatchNormRequest,
    LossSurfaceGridRequest,
    OptimizerRaceRequest,
    WeightLossSurfaceRequest,
)

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/loss-surface")
def loss_surface(req: LossSurfaceGridRequest) -> dict:
    try:
        return generate_grid(req.surface, req.x_min, req.x_max, req.y_min, req.y_max, req.resolution)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/race")
def race(req: OptimizerRaceRequest) -> dict:
    racers = [r.model_dump() for r in req.racers]
    try:
        return run_race(req.surface, req.start, racers, req.num_steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/weight-loss-surface")
def weight_loss_surface(req: WeightLossSurfaceRequest) -> dict:
    try:
        net = ann_registry.get(req.network_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"network {req.network_id} not found")

    try:
        return compute_weight_loss_surface(
            net,
            req.input,
            req.target,
            req.loss,
            req.coord1.model_dump(),
            req.coord2.model_dump(),
            req.range1,
            req.range2,
            req.resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/batchnorm")
def batchnorm(req: BatchNormRequest) -> dict:
    try:
        return compute_batchnorm_trace(req.batch, req.gamma, req.beta, req.eps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
