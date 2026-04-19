"""
Thread-safe singleton pattern implementations.

Provides two interchangeable approaches:
1. Metaclass-based singleton
2. Decorator-based singleton with identical API
"""

import logging
import threading
from typing import TypeVar, Dict, Type, Any, Optional, cast

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass with comprehensive API.

    Usage:
        class MyClass(metaclass=SingletonMeta):
            pass
    """

    _instances: Dict[Type[Any], Any] = {}
    _lock = threading.Lock()

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        logger.debug("SingletonMeta.__call__ invoked for %s", cls.__name__)

        with cls._lock:
            if cls not in cls._instances:
                logger.debug("Creating new singleton instance for %s", cls.__name__)
                cls._instances[cls] = super().__call__(*args, **kwargs)
                logger.info("Singleton instance created for %s (id: %s)",
                           cls.__name__, id(cls._instances[cls]))
            else:
                logger.debug("Returning existing singleton instance for %s (id: %s)",
                           cls.__name__, id(cls._instances[cls]))

            return cls._instances[cls]

    @classmethod
    def clear_instance(mcs, cls: Type[T]) -> bool:
        """
        Clear the singleton instance for a specific class.

        Args:
            cls: The class to clear the instance for

        Returns:
            True if instance was cleared, False if no instance existed
        """
        logger.debug("Attempting to clear singleton instance for %s", cls.__name__)

        with mcs._lock:
            if cls in mcs._instances:
                instance = mcs._instances[cls]

                # Call cleanup method if it exists
                if hasattr(instance, '_singleton_cleanup') and callable(instance._singleton_cleanup):
                    logger.debug("Calling singleton cleanup for %s", cls.__name__)
                    instance._singleton_cleanup()

                del mcs._instances[cls]
                logger.info("Singleton instance cleared for %s", cls.__name__)
                return True
            else:
                logger.debug("No instance to clear for %s", cls.__name__)
                return False

    @classmethod
    def get_instance(mcs, cls: Type[T]) -> Optional[T]:
        """
        Get the existing singleton instance without creating it.

        Args:
            cls: The class to get the instance for

        Returns:
            The existing instance or None if not created
        """
        instance = mcs._instances.get(cls)
        if instance:
            logger.debug("Retrieved existing instance for %s (id: %s)",
                        cls.__name__, id(instance))
        else:
            logger.debug("No existing instance found for %s", cls.__name__)
        return instance

    @classmethod
    def clear_all_instances(mcs) -> None:
        """Clear all singleton instances (primarily for testing)."""
        logger.debug("Clearing all singleton instances")

        with mcs._lock:
            # Call cleanup for all instances
            for cls, instance in list(mcs._instances.items()):
                if hasattr(instance, '_singleton_cleanup') and callable(instance._singleton_cleanup):
                    logger.debug("Calling cleanup for %s", cls.__name__)
                    instance._singleton_cleanup()

            mcs._instances.clear()
            logger.info("All singleton instances cleared")


class SingletonBase(metaclass=SingletonMeta):
    """
    Base class for singletons using the metaclass approach.

    Provides consistent cleanup interface and utility methods.
    """

    def _singleton_cleanup(self) -> None:
        """
        Override this method to perform cleanup when singleton is cleared.

        This is called automatically when clear_instance() is called.
        """
        logger.debug("Default singleton cleanup called for %s",
                     self.__class__.__name__)


def singleton_class(cls: Type[T]) -> Type[T]:
    """
    Singleton class decorator.

    USAGE NOTES:
    - Recommended for simple classes without inheritance
    - If used with inheritance, call base class __init__ explicitly:
        BaseClass.__init__(self)
    - Avoid super() calls in decorated classes
    """
    # Store original class attributes
    original_dict = dict(cls.__dict__)
    original_bases = cls.__bases__

    # Create a new class with SingletonMeta as metaclass
    # This avoids potential metaclass conflicts
    singleton_cls = SingletonMeta(cls.__name__, original_bases, original_dict)

    # Add utility methods to maintain API alignment
    singleton_cls.clear_instance = classmethod(lambda cls: SingletonMeta.clear_instance(cls))
    singleton_cls.get_instance = classmethod(lambda cls: SingletonMeta.get_instance(cls))

    logger.debug("Applied singleton decorator to %s", cls.__name__)
    return cast(Type[T], singleton_cls)
