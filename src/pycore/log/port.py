import json
import socket
from pathlib import Path
from typing import Optional


class PortManager:
    """
    Manages port allocation for TCP logging across processes.

    Uses a lock file mechanism to ensure only one process binds to a port.
    """

    DEFAULT_PORT = 9020
    PORT_RANGE = range(9020, 9030)  # Try 10 ports

    @staticmethod
    def get_lock_file(port: int) -> Path:
        """Get the lock file path for a given port."""
        import tempfile
        return Path(tempfile.gettempdir()) / f"tcp_log_server_{port}.lock"

    @staticmethod
    def is_port_available(port: int) -> bool:
        """Check if a port is available for binding."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            return True
        except OSError:
            return False

    @classmethod
    def acquire_port(cls, preferred_port: Optional[int] = None) -> Optional[
        int]:
        """
        Acquire an available port, creating a lock file.

        Args:
            preferred_port: Preferred port number, or None to auto-select

        Returns:
            Port number if successful, None otherwise
        """
        ports_to_try = [preferred_port] if preferred_port else cls.PORT_RANGE

        for port in ports_to_try:
            if not cls.is_port_available(port):
                continue

            lock_file = cls.get_lock_file(port)
            try:
                # Try to create lock file exclusively
                if not lock_file.exists():
                    lock_file.write_text(str(port))
                    return port
            except (IOError, OSError):
                continue

        return None

    @classmethod
    def release_port(cls, port: int) -> None:
        """Release a port by removing its lock file."""
        print("Releasing port: ", port)
        lock_file = cls.get_lock_file(port)
        try:
            if lock_file.exists():
                lock_file.unlink()
        except (IOError, OSError):
            raise

    @classmethod
    def is_server_running(cls, port: int) -> bool:
        """
        Check if a TCP log server is running on a specific port.

        Args:
            port: Port number to check

        Returns:
            True if server is running and responding, False otherwise
        """
        # Check if port is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            sock.connect(('localhost', port))
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    @classmethod
    def find_active_server_port(cls) -> Optional[int]:
        """Find the port of an active TCP log server."""
        for port in cls.PORT_RANGE:
            lock_file = cls.get_lock_file(port)
            if lock_file.exists():
                # Verify the server is actually running
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.settimeout(0.5)
                    sock.connect(('localhost', port))
                    sock.close()
                    return port
                except (socket.timeout, ConnectionRefusedError, OSError):
                    # Lock file exists but server isn't running - clean up
                    cls.release_port(port)
        return None

    def clean_stale_locks(self, output_json: bool = False):
        """Remove stale lock files."""
        from pathlib import Path
        import tempfile

        lock_dir = Path(tempfile.gettempdir())
        lock_files = list(lock_dir.glob("tcp_log_server_*.lock"))

        removed = []
        failed = []

        for lock_file in lock_files:
            try:
                port = int(lock_file.read_text().strip())
                if not PortManager.is_server_running(port):
                    lock_file.unlink()
                    removed.append({'file': str(lock_file), 'port': port})
            except Exception as e:
                failed.append({'file': str(lock_file), 'error': str(e)})

        if output_json:
            result = {
                'removed': removed,
                'failed': failed,
                'removed_count': len(removed),
                'failed_count': len(failed)
            }
            print(json.dumps(result, indent=2))
        else:
            if removed:
                print(f"✓ Removed {len(removed)} stale lock file(s):")
                for info in removed:
                    print(f"  • Port {info['port']}: {info['file']}")
            else:
                print("No stale lock files to remove")

            if failed:
                print(f"\n✗ Failed to remove {len(failed)} file(s):")
                for info in failed:
                    print(f"  • {info['file']}: {info['error']}")
