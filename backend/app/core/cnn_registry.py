from app.core.base_registry import InMemoryRegistry
from app.engine.cnn.model import build_cnn

cnn_registry = InMemoryRegistry(builder=build_cnn)
