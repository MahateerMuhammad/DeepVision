from fastapi import APIRouter, HTTPException

from app.core.cnn_registry import cnn_registry
from app.engine.cnn.state_tree import build_cnn_forward_state_tree
from app.engine.cnn.receptive_field import compute_receptive_field
from app.engine.cnn.saliency import compute_saliency_map
from app.engine.cnn.sliding_kernel import sliding_kernel_trace
from app.schemas.cnn import (
    CNNForwardRequest,
    CreateCNNRequest,
    CreateCNNResponse,
    ReceptiveFieldRequest,
    SaliencyRequest,
    SlidingKernelRequest,
)

router = APIRouter(prefix="/cnn", tags=["cnn"])


def _get_network(network_id: str):
    try:
        return cnn_registry.get(network_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"cnn {network_id} not found")


@router.post("", response_model=CreateCNNResponse)
def create_cnn(req: CreateCNNRequest) -> CreateCNNResponse:
    try:
        network_id, net = cnn_registry.create(req.spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CreateCNNResponse(
        network_id=network_id,
        spec=req.spec,
        param_count=net.param_count(),
        output_shape=list(net.output_shape),
    )


@router.delete("/{network_id}", status_code=204)
def delete_cnn(network_id: str) -> None:
    cnn_registry.delete(network_id)


@router.post("/forward")
def forward_pass(req: CNNForwardRequest) -> dict:
    net = _get_network(req.network_id)
    try:
        return build_cnn_forward_state_tree(net, req.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/receptive-field")
def receptive_field(req: ReceptiveFieldRequest) -> dict:
    net = _get_network(req.network_id)
    try:
        return compute_receptive_field(net, req.layer_index, req.channel, req.row, req.col)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/saliency")
def saliency(req: SaliencyRequest) -> dict:
    net = _get_network(req.network_id)
    try:
        return compute_saliency_map(net, req.image, req.target_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sliding-kernel")
def sliding_kernel(req: SlidingKernelRequest) -> dict:
    try:
        return sliding_kernel_trace(req.image, req.kernel, req.stride, req.padding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
