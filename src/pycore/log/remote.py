import multiprocessing
from typing import Optional

from pycore.log.port import PortManager
from qtcore.log.mngr import setup_tcp_server_logging
from qtgui.log.widget import LogWidget


class RemoteLogServerProcess:
    """
    Manages a log server running in a separate process.

    This allows you to spawn a log viewer in another process and
    connect to it from your main application.
    """

    def __init__(self, port: Optional[int] = None, max_lines: int = 10000,
                 font_size: int = 10):
        """
        Initialize remote log server configuration.

        Args:
            port: TCP port (None for auto-select)
            max_lines: Maximum lines in log widget
            font_size: Font size for log display
        """
        self.port = port
        self.max_lines = max_lines
        self.font_size = font_size
        self._process: Optional[multiprocessing.Process] = None
        self._actual_port: Optional[int] = None
        self._port_queue = multiprocessing.Queue()

    def _server_worker(self):
        """Worker function that runs the log server."""
        try:
            from qtcore.app import Application
            import sys

            # Create Qt application
            app = Application(argv=sys.argv)
            log_widget = LogWidget(max_lines=self.max_lines,
                                   font_size=self.font_size)

            # Setup TCP server
            widget, actual_port = setup_tcp_server_logging(
                app, log_widget, port=self.port, auto_show=True
            )

            log_widget.setWindowTitle(f"Remote Log Server")

            # Send actual port back to parent process
            self._port_queue.put(actual_port)

            # Run event loop
            sys.exit(app.exec())

        except Exception as e:
            self._port_queue.put(None)
            print(f"Error in server worker: {e}")
            import traceback
            traceback.print_exc()

    def start(self) -> int:
        """
        Start the log server in a separate process.

        Returns:
            The actual port number the server is listening on

        Raises:
            RuntimeError: If server fails to start
        """
        if self._process and self._process.is_alive():
            raise RuntimeError("Server process already running")

        # Start server process
        self._process = multiprocessing.Process(
            target=self._server_worker,
            name="RemoteLogServer",
            daemon=False
        )
        self._process.start()

        # Wait for server to start and report its port
        try:
            self._actual_port = self._port_queue.get(timeout=10.0)
            if self._actual_port is None:
                raise RuntimeError("Server failed to start")
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to start log server: {e}") from e

        # Verify server is actually running
        import time
        max_retries = 10
        for i in range(max_retries):
            if PortManager.is_server_running(self._actual_port):
                return self._actual_port
            time.sleep(0.5)

        self.stop()
        raise RuntimeError("Server started but not responding")

    def stop(self):
        """Stop the log server process."""
        if self._process:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=2.0)
            self._process = None

        # Clean up port
        if self._actual_port:
            PortManager.release_port(self._actual_port)
            self._actual_port = None

    @property
    def is_running(self) -> bool:
        """Check if server process is running."""
        return (self._process is not None and
                self._process.is_alive() and
                self._actual_port is not None and
                PortManager.is_server_running(self._actual_port))

    @property
    def actual_port(self) -> Optional[int]:
        """Get the actual port the server is running on."""
        return self._actual_port

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
