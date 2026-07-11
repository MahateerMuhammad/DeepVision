import uuid
from typing import Callable, Generic, TypeVar

SpecT = TypeVar("SpecT")
ModelT = TypeVar("ModelT")


class InMemoryRegistry(Generic[SpecT, ModelT]):
    """In-memory store of live model instances, keyed by a generated id.

    Deliberately a single-process, non-persistent registry: this is a
    stateful visualization/dev tool (one user stepping through one network
    at a time), not a multi-tenant production service. Shared by every
    network kind (ANN, CNN, ...) via the `builder` function that turns a
    spec into a model.
    """

    def __init__(self, builder: Callable[[SpecT], ModelT]):
        self._builder = builder
        self._instances: dict[str, ModelT] = {}

    def create(self, spec: SpecT) -> tuple[str, ModelT]:
        instance_id = str(uuid.uuid4())
        self._instances[instance_id] = self._builder(spec)
        return instance_id, self._instances[instance_id]

    def get(self, instance_id: str) -> ModelT:
        instance = self._instances.get(instance_id)
        if instance is None:
            raise KeyError(instance_id)
        return instance

    def delete(self, instance_id: str) -> None:
        self._instances.pop(instance_id, None)
