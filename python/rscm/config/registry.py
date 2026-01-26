"""
Component registry for RSCM configuration.

This module provides a registry for component builders, enabling
config files to reference components by name.

Example:
    >>> from rscm.config.registry import component_registry, register_component
    >>>
    >>> @register_component("MyComponent")
    >>> class MyComponentBuilder:
    ...     pass
    >>>
    >>> component_registry.get("MyComponent")
    <class 'MyComponentBuilder'>
"""

from __future__ import annotations

from collections.abc import Callable

from .exceptions import ComponentNotFoundError

__all__ = ["ComponentRegistry", "component_registry", "register_component"]


class ComponentRegistry:
    """
    Registry for component builders.

    Maps component names to their builder classes, enabling configuration
    files to reference components by name.

    Example:
        >>> registry = ComponentRegistry()
        >>> registry.register("TwoLayer", TwoLayerBuilder)
        >>> registry.get("TwoLayer")
        <class 'TwoLayerBuilder'>
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: dict[str, type] = {}

    def register(self, name: str, builder_class: type) -> None:
        """
        Register a component builder by name.

        Parameters
        ----------
        name
            Component name to register.
        builder_class
            Builder class to associate with this name.

        Raises
        ------
        ValueError
            If the name is already registered with a different class.
        """
        if name in self._registry and self._registry[name] is not builder_class:
            msg = f"Component '{name}' is already registered with a different class"
            raise ValueError(msg)
        self._registry[name] = builder_class

    def get(self, name: str) -> type:
        """
        Get builder class by name.

        Parameters
        ----------
        name
            Component name to look up.

        Returns
        -------
        type
            Builder class associated with the name.

        Raises
        ------
        ComponentNotFoundError
            If the component is not registered.
        """
        if name not in self._registry:
            raise ComponentNotFoundError(name, self.list())
        return self._registry[name]

    def list(self) -> list[str]:
        """
        List all registered component names.

        Returns
        -------
        list[str]
            Sorted list of registered component names.
        """
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """
        Check if a component is registered.

        Parameters
        ----------
        name
            Component name to check.

        Returns
        -------
        bool
            True if the component is registered, False otherwise.
        """
        return name in self._registry


# Module-level singleton instance
component_registry = ComponentRegistry()


def register_component(name: str) -> Callable[[type], type]:
    """
    Register a component builder via decorator.

    Parameters
    ----------
    name
        Component name to register.

    Returns
    -------
    Callable[[type], type]
        Decorator that registers the class and returns it unchanged.

    Example:
        >>> @register_component("TwoLayer")
        >>> class TwoLayerBuilder:
        ...     pass
    """

    def decorator(cls: type) -> type:
        component_registry.register(name, cls)
        return cls

    return decorator
