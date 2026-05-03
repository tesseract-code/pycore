import logging
import sys
from typing import Optional, Callable


class ReplacementExceptHook:
    """
    Exception handler that logs exceptions reaching the top level instead of exiting.

    This class provides a safer exception handling mechanism for GUI applications
    by logging uncaught exceptions rather than terminating the application.
    """

    def __init__(
            self,
            logger: Optional[logging.Logger] = None,
            old_excepthook: Optional[Callable] = None
    ) -> None:
        """
        Initialize the exception hook.

        Args:
            logger: Logger instance to use (defaults to root logger)
            old_excepthook: Previous exception hook to chain to
        """
        self.logger = logger if logger is not None else logging.getLogger()
        self.old_excepthook = old_excepthook

    def __call__(self, etype: type, evalue: BaseException, tb) -> None:
        """
        Handle an uncaught exception.

        Args:
            etype: Exception type
            evalue: Exception value
            tb: Traceback object
        """
        try:
            exc_info = (etype, evalue, tb)
            self.logger.error('Exception raised to toplevel', exc_info=exc_info)
        except Exception as e:
            # Last resort error handling
            print(f"Error in exception hook: {e}", file=sys.stderr)
            import traceback
            traceback.print_exception(etype, evalue, tb)

        if self.old_excepthook is not None:
            try:
                self.old_excepthook(etype, evalue, tb)
            except Exception as e:
                print(f"Error in old exception hook: {e}", file=sys.stderr)


def replace_excepthook(
        logger: Optional[logging.Logger] = None,
        passthrough: bool = True
) -> ReplacementExceptHook:
    """
    Replace the system exception hook with a logging exception hook.

    Args:
        logger: Logger instance to use (defaults to root logger)
        passthrough: Whether to chain to the original exception hook

    Returns:
        The installed ReplacementExceptHook instance
    """
    old = sys.excepthook if passthrough else None
    replacement = ReplacementExceptHook(logger=logger, old_excepthook=old)
    sys.excepthook = replacement
    return replacement
