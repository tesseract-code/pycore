import logging
from typing import TypeVar, Optional, Type


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically appends per-call context to log messages.

    This adapter extends `logging.LoggerAdapter` to allow passing key-value pairs
    directly as keyword arguments to logging methods. These pairs are appended to
    the message in the format ``"message | key1=val1 key2=val2"``, overriding any
    initial extra context.

    Usage
    -----
    logger = ContextAdapter(logging.getLogger(__name__), {})
    logger.info("User logged in", user_id=123)
    # Output: "User logged in | user_id=123"
    """

    def process(self, msg, kwargs):
        """
        Merge keyword arguments into the message and return a processed message.

        The method extracts the ``extra`` key from `kwargs` and merges all other
        keyword arguments into a dictionary. Each key-value pair is formatted as
        ``"key=value"`` and appended to the original message, separated by `` | ``.
        After processing, an empty dictionary is returned as the new keyword
        arguments to prevent the logging system from interpreting the context keys.

        Parameters
        ----------
        msg : str
            The original log message.
        kwargs : dict
            Keyword arguments passed to the logging call. May contain ``extra``
            and arbitrary context keys.

        Returns
        -------
        msg : str
            The original message with appended context, if any context keys were present.
        kwargs : dict
            An empty dictionary to avoid double passing of context variables.
        """
        context = kwargs.pop('extra', {})
        # Merge kwargs into context if passed directly (e.g. logger.info("msg", key=val))
        # Note: LoggerAdapter standard usage puts context in the constructor,
        # but here we allow per-call context.
        context.update(kwargs)

        if context:
            # Format the message to your specific style
            ctx_str = " ".join([f"{k}={v}" for k, v in context.items()])
            return f"{msg} | {ctx_str}", {}  # Return empty kwargs to avoid double passing

        return msg, kwargs


T = TypeVar('T')


def with_logger(
        cls: Optional[Type[T]] = None,
        *,
        attr_name: str = "_logger",
        logger_name: Optional[str] = None
) -> Type[T]:
    """
    Class decorator that adds a `ContextAdapter` logger instance to a class.

    The logger is stored as an instance attribute named by `attr_name`. If `logger_name`
    is not provided, the fully qualified class name (``module.ClassName``) is used.
    A `ContextAdapter` is created for the logger so that per-call context can be
    appended to messages.

    This decorator can be used with or without arguments:

    - Without arguments: ``@with_logger``
    - With arguments: ``@with_logger(attr_name="log", logger_name="custom")``

    If the class already defines an attribute with the given name, a `UserWarning` is
    issued before overwriting.

    Parameters
    ----------
    cls : type, optional
        The class to decorate. If `None`, the decorator is being called with
        arguments and returns a wrapper function. Default is `None`.
    attr_name : str, optional
        Name of the attribute to hold the logger adapter. Default is ``"_logger"``.
    logger_name : str, optional
        Custom logger name. If `None`, the fully qualified class name is used.
        Default is `None`.

    Returns
    -------
    type or callable
        If `cls` is provided, returns the decorated class with the logger attribute.
        If `cls` is `None`, returns a wrapper function that expects a class and
        performs the decoration.

    Raises
    ------
    ValueError
        If `logger_name` is `None` and the class has no ``__module__`` attribute.

    Warns
    -----
    UserWarning
        If the class already has an attribute named `attr_name`.
    """

    def wrap(cls: Type[T]) -> Type[T]:
        if attr_name in cls.__dict__:
            import warnings
            warnings.warn(
                f"Class {cls.__name__} already defines attribute '{attr_name}'. Overwriting.",
                UserWarning,
                stacklevel=2
            )

        if logger_name is not None:
            name = logger_name
        else:
            if cls.__module__ is None:
                raise ValueError(
                    f"Class {cls.__qualname__} has no __module__ attribute")
            name = f"{cls.__module__}.{cls.__qualname__}"

        setattr(cls, attr_name, ContextAdapter(logging.getLogger(name), {}))
        return cls

    if cls is None:
        return wrap
    return wrap(cls)