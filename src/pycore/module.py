# lazy_cython_loader.py
"""
Generic lazy loader for Cython compiled modules (.so files).

Usage:
    from lazy_cython_loader import LazyModuleLoader

    # Create loader
    loader = LazyModuleLoader(
        module_name='my_cython_module',
        exports=['FastClass', 'OptimizedClass', 'compute_fast']
    )

    # Export to current namespace
    FastClass = loader.FastClass
    OptimizedClass = loader.OptimizedClass
    compute_fast = loader.compute_fast

    # Or use attribute access
    result = loader.compute_fast(data)
"""

import importlib
from types import ModuleType
from typing import List, Any, Optional


class LazyModuleLoader:
    """
    Lazy loader for Cython compiled modules with automatic member export.

    The module is only imported when first accessed, reducing startup time.
    """

    def __init__(self, module_name: str, exports: Optional[List[str]] = None):
        """
        Initialize lazy loader.

        Args:
            module_name: Name of the module to load (e.g., 'my_module.fast_impl')
            exports: List of member names to export (classes/functions).
                    If None, all public members are available via attribute access.
        """
        self._module_name = module_name
        self._exports = exports or []
        self._module: Optional[ModuleType] = None
        self._loaded = False

    def _ensure_loaded(self):
        """Load the module if not already loaded."""
        if not self._loaded:
            try:
                self._module = importlib.import_module(self._module_name)
                self._loaded = True
            except ImportError as e:
                raise ImportError(
                    f"Failed to load Cython module '{self._module_name}': {e}\n"
                    f"Ensure the .so file is compiled and in Python path."
                ) from e

    def __getattr__(self, name: str) -> Any:
        """
        Lazy attribute access - loads module on first access.

        Args:
            name: Attribute name to retrieve from module

        Returns:
            The requested attribute from the loaded module
        """
        # Avoid infinite recursion for internal attributes
        if name.startswith('_'):
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'")

        self._ensure_loaded()

        try:
            return getattr(self._module, name)
        except AttributeError:
            raise AttributeError(
                f"Module '{self._module_name}' has no attribute '{name}'"
            ) from None

    def get_exports(self) -> dict:
        """
        Get all exported members as a dictionary.

        Returns:
            Dictionary mapping export names to their values
        """
        self._ensure_loaded()

        if self._exports:
            return {name: getattr(self._module, name) for name in self._exports}
        else:
            # Export all public members
            return {
                name: getattr(self._module, name)
                for name in dir(self._module)
                if not name.startswith('_')
            }

    def inject_into_namespace(self, namespace: dict):
        """
        Inject exported members into a namespace (typically globals()).

        Args:
            namespace: Target namespace dictionary (use globals())
        """
        namespace.update(self.get_exports())

    @property
    def is_loaded(self) -> bool:
        """Check if module has been loaded."""
        return self._loaded

    @property
    def module(self) -> ModuleType:
        """Get the loaded module (forces load if needed)."""
        self._ensure_loaded()
        return self._module


# Alternative: Module-level proxy for even cleaner usage
class LazyModuleProxy(ModuleType):
    """
    A module proxy that lazily loads a Cython module.

    This can replace a module in sys.modules for transparent lazy loading.
    """

    def __init__(self, name: str, actual_module_name: str):
        """
        Initialize module proxy.

        Args:
            name: Name for this proxy module
            actual_module_name: Real module to load lazily
        """
        super().__init__(name)
        self._actual_module_name = actual_module_name
        self._actual_module = None

    def _load(self):
        """Load the actual module."""
        if self._actual_module is None:
            self._actual_module = importlib.import_module(
                self._actual_module_name)
            # Copy attributes to this proxy
            self.__dict__.update(self._actual_module.__dict__)

    def __getattr__(self, name: str):
        """Lazy load on attribute access."""
        self._load()
        return getattr(self._actual_module, name)

    def __dir__(self):
        """Show available attributes."""
        self._load()
        return dir(self._actual_module)


# Convenience function for quick setup
def setup_lazy_import(module_name: str, exports: List[str],
                      target_namespace: dict) ->LazyModuleLoader:
    """
    Quick setup for lazy importing Cython modules.

    Args:
        module_name: Module to import
        exports: List of names to export
        target_namespace: Where to inject (typically globals())

    Example:
        # In your __init__.py or main module:
        setup_lazy_import(
            'mypackage.fast_core',
            ['FastProcessor', 'DataVector', 'compute_metric'],
            globals()
        )
    """
    loader = LazyModuleLoader(module_name, exports)
    loader.inject_into_namespace(target_namespace)
    return loader
