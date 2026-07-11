from app.core.base_registry import InMemoryRegistry
from app.engine.ann.model import build_network

ann_registry = InMemoryRegistry(builder=build_network)
