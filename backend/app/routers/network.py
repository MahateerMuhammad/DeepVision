from fastapi import APIRouter, HTTPException

from app.core.ann_registry import ann_registry
from app.engine.ann.chain_rule import trace_weight_gradient
from app.engine.ann.state_tree import build_backward_state_tree, build_forward_state_tree
from app.schemas.network import (
    BackwardPassRequest,
    CreateNetworkRequest,
    CreateNetworkResponse,
    ForwardPassRequest,
    TraceRequest,
)

router = APIRouter(prefix="/networks", tags=["networks"])


@router.post("", response_model=CreateNetworkResponse)
def create_network(req: CreateNetworkRequest) -> CreateNetworkResponse:
    network_id, net = ann_registry.create(req.spec)
    return CreateNetworkResponse(network_id=network_id, spec=req.spec, param_count=net.param_count())


@router.delete("/{network_id}", status_code=204)
def delete_network(network_id: str) -> None:
    ann_registry.delete(network_id)


@router.post("/forward")
def forward_pass(req: ForwardPassRequest) -> dict:
    try:
        net = ann_registry.get(req.network_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"network {req.network_id} not found")

    try:
        return build_forward_state_tree(net, req.input, req.training)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/backward")
def backward_pass(req: BackwardPassRequest) -> dict:
    try:
        net = ann_registry.get(req.network_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"network {req.network_id} not found")

    try:
        return build_backward_state_tree(net, req.input, req.target, req.loss, req.training)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/trace")
def trace_weight(req: TraceRequest) -> dict:
    try:
        net = ann_registry.get(req.network_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"network {req.network_id} not found")

    try:
        return trace_weight_gradient(
            net,
            req.input,
            req.target,
            req.loss,
            req.layer_index,
            req.weight_row,
            req.weight_col,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
