import pytest

from app.core.base_registry import InMemoryRegistry


def test_create_returns_unique_ids_and_builds_via_builder():
    registry = InMemoryRegistry(builder=lambda spec: {"value": spec * 2})
    id1, obj1 = registry.create(5)
    id2, obj2 = registry.create(5)
    assert obj1 == {"value": 10}
    assert obj2 == {"value": 10}
    assert id1 != id2


def test_get_returns_the_created_instance():
    registry = InMemoryRegistry(builder=lambda spec: spec.upper())
    instance_id, _ = registry.create("hello")
    assert registry.get(instance_id) == "HELLO"


def test_get_unknown_id_raises_keyerror():
    registry = InMemoryRegistry(builder=lambda spec: spec)
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_delete_removes_instance():
    registry = InMemoryRegistry(builder=lambda spec: spec)
    instance_id, _ = registry.create(1)
    registry.delete(instance_id)
    with pytest.raises(KeyError):
        registry.get(instance_id)


def test_delete_unknown_id_is_a_noop_not_an_error():
    registry = InMemoryRegistry(builder=lambda spec: spec)
    registry.delete("never-existed")  # must not raise


def test_two_registries_do_not_share_state():
    registry_a = InMemoryRegistry(builder=lambda spec: spec)
    registry_b = InMemoryRegistry(builder=lambda spec: spec)
    id_a, _ = registry_a.create("a")
    assert id_a not in registry_b._instances
    with pytest.raises(KeyError):
        registry_b.get(id_a)


def test_builder_exceptions_propagate_and_do_not_register_partial_state():
    def failing_builder(spec):
        raise ValueError("bad spec")

    registry = InMemoryRegistry(builder=failing_builder)
    with pytest.raises(ValueError):
        registry.create("anything")
    assert registry._instances == {}
