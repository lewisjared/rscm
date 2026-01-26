"""
Unit tests for rscm.config.registry module.

Tests ComponentRegistry class and register_component decorator.
"""

from __future__ import annotations

import pytest

from rscm.config.exceptions import ComponentNotFoundError
from rscm.config.registry import ComponentRegistry, register_component


class TestComponentRegistry:
    """Tests for ComponentRegistry class."""

    def test_registry_init_empty(self):
        """ComponentRegistry initializes with empty registry."""
        registry = ComponentRegistry()
        assert registry.list() == []

    def test_register_component(self):
        """register() adds component to registry."""
        registry = ComponentRegistry()

        class MyComponent:
            pass

        registry.register("MyComponent", MyComponent)
        assert registry.is_registered("MyComponent")

    def test_get_registered_component(self):
        """get() returns registered component class."""
        registry = ComponentRegistry()

        class MyComponent:
            pass

        registry.register("MyComponent", MyComponent)
        retrieved = registry.get("MyComponent")
        assert retrieved is MyComponent

    def test_get_unregistered_component_raises(self):
        """get() raises ComponentNotFoundError for unregistered component."""
        registry = ComponentRegistry()
        with pytest.raises(
            ComponentNotFoundError, match="Component 'Unknown' not found"
        ):
            registry.get("Unknown")

    def test_get_includes_available_components_in_error(self):
        """ComponentNotFoundError includes list of available components."""
        registry = ComponentRegistry()

        class CompA:
            pass

        class CompB:
            pass

        registry.register("ComponentA", CompA)
        registry.register("ComponentB", CompB)

        with pytest.raises(
            ComponentNotFoundError,
            match=r"Available components.*ComponentA.*ComponentB",
        ) as exc_info:
            registry.get("Unknown")

        assert exc_info.value.name == "Unknown"
        assert "ComponentA" in exc_info.value.available
        assert "ComponentB" in exc_info.value.available

    def test_list_empty_registry(self):
        """list() returns empty list for empty registry."""
        registry = ComponentRegistry()
        assert registry.list() == []

    def test_list_returns_sorted_names(self):
        """list() returns alphabetically sorted component names."""
        registry = ComponentRegistry()

        class CompZ:
            pass

        class CompA:
            pass

        class CompM:
            pass

        registry.register("Zebra", CompZ)
        registry.register("Alpha", CompA)
        registry.register("Middle", CompM)

        names = registry.list()
        assert names == ["Alpha", "Middle", "Zebra"]

    def test_is_registered_true(self):
        """is_registered() returns True for registered component."""
        registry = ComponentRegistry()

        class MyComponent:
            pass

        registry.register("MyComponent", MyComponent)
        assert registry.is_registered("MyComponent") is True

    def test_is_registered_false(self):
        """is_registered() returns False for unregistered component."""
        registry = ComponentRegistry()
        assert registry.is_registered("Unknown") is False

    def test_register_duplicate_with_same_class(self):
        """register() allows re-registering same name with same class."""
        registry = ComponentRegistry()

        class MyComponent:
            pass

        registry.register("MyComponent", MyComponent)
        # Should not raise
        registry.register("MyComponent", MyComponent)

    def test_register_duplicate_with_different_class_raises(self):
        """register() raises ValueError when re-registering with different class."""
        registry = ComponentRegistry()

        class ComponentA:
            pass

        class ComponentB:
            pass

        registry.register("MyComponent", ComponentA)

        with pytest.raises(
            ValueError, match="Component 'MyComponent' is already registered"
        ):
            registry.register("MyComponent", ComponentB)

    def test_multiple_components(self):
        """Registry can handle multiple components."""
        registry = ComponentRegistry()

        class CompA:
            pass

        class CompB:
            pass

        class CompC:
            pass

        registry.register("A", CompA)
        registry.register("B", CompB)
        registry.register("C", CompC)

        assert registry.get("A") is CompA
        assert registry.get("B") is CompB
        assert registry.get("C") is CompC
        assert registry.list() == ["A", "B", "C"]


class TestRegisterComponentDecorator:
    """Tests for register_component decorator."""

    def test_decorator_registers_component(self):
        """@register_component decorator registers the class."""
        # Create a fresh registry for testing
        test_registry = ComponentRegistry()

        @register_component("TestComponent")
        class MyComponent:
            pass

        # Manually register to test registry for verification
        test_registry.register("TestComponent", MyComponent)

        assert test_registry.is_registered("TestComponent")
        assert test_registry.get("TestComponent") is MyComponent

    def test_decorator_returns_class_unchanged(self):
        """@register_component decorator returns the class unchanged."""

        class OriginalClass:
            value = 42

        decorated = register_component("Test")(OriginalClass)

        assert decorated is OriginalClass
        assert decorated.value == 42

    def test_decorator_preserves_class_attributes(self):
        """@register_component decorator preserves class attributes and methods."""

        @register_component("TestClass")
        class MyClass:
            class_var = "test"

            def __init__(self, x):
                self.x = x

            def method(self):
                return self.x * 2

        instance = MyClass(21)
        assert MyClass.class_var == "test"
        assert instance.x == 21
        assert instance.method() == 42

    def test_decorator_with_different_names(self):
        """@register_component can register class under different name."""
        test_registry = ComponentRegistry()

        @register_component("RegisteredName")
        class ActualClassName:
            pass

        test_registry.register("RegisteredName", ActualClassName)

        assert test_registry.is_registered("RegisteredName")
        assert not test_registry.is_registered("ActualClassName")

    def test_decorator_usage_pattern(self):
        """@register_component follows expected usage pattern."""
        test_registry = ComponentRegistry()

        @register_component("ComponentOne")
        class ComponentOne:
            pass

        @register_component("ComponentTwo")
        class ComponentTwo:
            pass

        test_registry.register("ComponentOne", ComponentOne)
        test_registry.register("ComponentTwo", ComponentTwo)

        assert test_registry.list() == ["ComponentOne", "ComponentTwo"]
