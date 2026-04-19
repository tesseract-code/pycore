import logging

import psutil

from pycore.platform import IS_MACOS, IS_WINDOWS

logger = logging.getLogger(__name__)


def set_high_priority(process_name: str = "Process", pid: int | None = None) -> bool:
    """
    Safely raises the process priority.

    Behavior:
    - Linux/macOS: Sets 'nice' value to -10 (High Priority).
    - Windows: Sets priority class to HIGH_PRIORITY_CLASS.

    Safe:
    - Catches permission errors (if not running as sudo/admin).
    - Does not crash the application if it fails.
    """
    try:
        p = psutil.Process(pid) if pid is not None else psutil.Process()

        if IS_WINDOWS:
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            # Unix/macOS: Range is -20 (High) to 20 (Low). Default is 0.
            # -10 is polite but firm; -20 is usually reserved for kernel tasks.
            p.nice(-10)

        logger.info(f"[{process_name}] Priority raised to HIGH.")
        return True

    except (psutil.AccessDenied, PermissionError):
        logger.warning(
            f"[{process_name}] Priority skipped: Requires Administrator/Root privileges.")
        return False
    except Exception as e:
        logger.error(f"[{process_name}] Failed to set priority: {e}")
        return False


def set_cpu_affinity(
    cpu_ids: list[int],
    process_name: str = "Process",
    pid: int | None = None,
) -> bool:
    """
    Safely pins the process to specific CPU cores.

    Behavior:
    - Linux/Windows: Pins process to the requested cores.
    - macOS: Ignored (returns False, logs warning) to prevent fighting the scheduler.

    Safe:
    - Validates that requested cores actually exist on the hardware.
    - Prevents empty list assignment (which can error out).
    """
    if IS_MACOS:
        logger.debug(
            f"[{process_name}] Affinity ignored: Not recommended on macOS "
            "(Apple Silicon manages this).")
        return False

    if not cpu_ids:
        logger.warning(f"[{process_name}] Affinity ignored: No CPU IDs provided.")
        return False

    # Reject negative core indices before they reach psutil.
    if any(c < 0 for c in cpu_ids):
        logger.error(
            f"[{process_name}] Affinity failed: Negative core indices are invalid: "
            f"{[c for c in cpu_ids if c < 0]}")
        return False

    try:
        p = psutil.Process(pid) if pid is not None else psutil.Process()

        if not hasattr(p, "cpu_affinity"):
            logger.info(f"[{process_name}] Affinity ignored: OS does not support pinning.")
            return False

        total_cpus = psutil.cpu_count(logical=True)
        if total_cpus is None:
            logger.warning(
                f"[{process_name}] Affinity failed: Could not determine CPU count "
                "(running in a restricted container?).")
            return False

        valid_cpus = [c for c in cpu_ids if c < total_cpus]

        if not valid_cpus:
            logger.error(
                f"[{process_name}] Affinity failed: Requested cores {cpu_ids} are "
                f"invalid for system with {total_cpus} CPUs.")
            return False

        dropped = sorted(set(cpu_ids) - set(valid_cpus))
        if dropped:
            logger.warning(
                f"[{process_name}] Dropped out-of-range cores: {dropped} "
                f"(system has {total_cpus} CPUs).")

        p.cpu_affinity(valid_cpus)
        logger.info(f"[{process_name}] CPU affinity pinned to cores: {valid_cpus}")
        return True

    except (psutil.AccessDenied, PermissionError):
        logger.warning(f"[{process_name}] Affinity failed: Access Denied.")
        return False
    except Exception as e:
        logger.error(f"[{process_name}] Affinity error: {e}")
        return False