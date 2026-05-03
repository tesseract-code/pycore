"""
TCP logging module with JSON serialization.
"""

import logging
import logging.handlers
import socketserver
import ssl
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List

from pycore.log.constants import DEFAULT_MAX_BYTES, \
    DEFAULT_BACKUP_COUNT, SERVER_START_TIMEOUT, \
    THREAD_JOIN_TIMEOUT
from pycore.log.handler import JSONSocketHandler, JSONLogRecordStreamHandler


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """
    Threaded TCP server with proper resource management.
    """

    allow_reuse_address = True
    daemon_threads = True  # Daemon threads for clean shutdown

    def __init__(self, server_address, request_handler_class):
        super().__init__(server_address, request_handler_class)
        self._received_records: List[logging.LogRecord] = []
        self._records_lock = threading.RLock()

    def handle_log_record(self, record: logging.LogRecord) -> None:
        """Thread-safe log record handling."""
        with self._records_lock:
            self._received_records.append(record)

    def get_received_messages(self) -> List[str]:
        """Get all received log messages (thread-safe)."""
        with self._records_lock:
            return [record.getMessage() for record in self._received_records]

    def get_received_records(self) -> List[logging.LogRecord]:
        """Get copy of received records (thread-safe)."""
        with self._records_lock:
            return self._received_records.copy()

    def clear_records(self) -> None:
        """Clear all received records (thread-safe)."""
        with self._records_lock:
            self._received_records.clear()


class TCPLogServer:
    """
    Main TCP log server with SSL support and proper lifecycle management.
    """

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 0,
            ssl_certfile: Optional[str] = None,
            ssl_keyfile: Optional[str] = None,
            ssl_cafile: Optional[str] = None
    ):
        self._validate_config(host, port, ssl_certfile, ssl_keyfile, ssl_cafile)

        self.host = host
        self.port = port
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_cafile = ssl_cafile

        self._ssl_context: Optional[ssl.SSLContext] = None
        self._server: Optional[ThreadedTCPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._started_event = threading.Event()
        self._running = False
        self._logger = logging.getLogger(f"{__name__}.TCPLogServer")

        if ssl_certfile and ssl_keyfile:
            self._setup_ssl_context()

    @staticmethod
    def _validate_config(host: str, port: int, ssl_certfile: Optional[str],
                         ssl_keyfile: Optional[str],
                         ssl_cafile: Optional[str]) -> None:
        """Validate server configuration."""
        if not host or not isinstance(host, str):
            raise ValueError("Host must be a non-empty string")

        if not isinstance(port, int) or port < 0 or port > 65535:
            raise ValueError("Port must be an integer between 0 and 65535")

        if ssl_certfile and not Path(ssl_certfile).exists():
            raise FileNotFoundError(f"SSL cert file not found: {ssl_certfile}")

        if ssl_keyfile and not Path(ssl_keyfile).exists():
            raise FileNotFoundError(f"SSL key file not found: {ssl_keyfile}")

        if ssl_cafile and not Path(ssl_cafile).exists():
            raise FileNotFoundError(f"SSL CA file not found: {ssl_cafile}")

        if bool(ssl_certfile) != bool(ssl_keyfile):
            raise ValueError(
                "Both ssl_certfile and ssl_keyfile must be provided together")

    def _setup_ssl_context(self) -> None:
        """Setup SSL context for the server."""
        try:
            self._ssl_context = ssl.create_default_context(
                ssl.Purpose.CLIENT_AUTH)
            self._ssl_context.load_cert_chain(
                certfile=self.ssl_certfile,
                keyfile=self.ssl_keyfile
            )

            if self.ssl_cafile:
                self._ssl_context.load_verify_locations(cafile=self.ssl_cafile)
                self._ssl_context.verify_mode = ssl.CERT_REQUIRED
                self._logger.info(
                    "SSL enabled with client certificate verification")
            else:
                self._ssl_context.verify_mode = ssl.CERT_NONE
                self._logger.warning(
                    "SSL enabled WITHOUT client certificate verification")

        except Exception as e:
            raise RuntimeError(f"Failed to setup SSL context: {e}") from e

    def start(self) -> None:
        """Start the TCP log server."""
        if self._running:
            raise RuntimeError("Server is already running")

        try:
            # Create the server
            self._server = ThreadedTCPServer(
                (self.host, self.port),
                JSONLogRecordStreamHandler
            )

            # Wrap with SSL if configured
            if self._ssl_context:
                self._server.socket = self._ssl_context.wrap_socket(
                    self._server.socket,
                    server_side=True
                )

            # Get actual port
            self.port = self._server.server_address[1]

            # Start server thread
            self._running = True
            self._started_event.clear()
            self._server_thread = threading.Thread(
                target=self._run_server,
                name="TCPLogServerThread",
                daemon=False  # Non-daemon for proper cleanup
            )
            self._server_thread.start()

            # Wait for server to start
            if not self._started_event.wait(timeout=SERVER_START_TIMEOUT):
                raise RuntimeError("Server failed to start within timeout")

            self._logger.info(
                f"TCP log server started on {self.host}:{self.port} "
                f"(SSL: {bool(self._ssl_context)})"
            )

        except Exception as e:
            self._running = False
            if self._server:
                self._server.server_close()
                self._server = None
            raise RuntimeError(f"Failed to start TCP log server: {e}") from e

    def _run_server(self) -> None:
        """Run the server main loop."""
        try:
            self._started_event.set()
            if self._server:
                self._server.serve_forever()
        except Exception as e:
            if self._running:
                self._logger.error(f"Server error: {e}")
        finally:
            self._started_event.set()  # In case we failed early

    def stop(self) -> None:
        """Stop the TCP log server with proper cleanup."""
        if not self._running:
            return

        self._running = False

        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as e:
                self._logger.error(f"Error shutting down server: {e}")

        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            if self._server_thread.is_alive():
                self._logger.warning(
                    "Server thread did not stop within timeout")

        self._server = None
        self._server_thread = None
        self._logger.info("TCP log server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running and self._server is not None

    def get_received_messages(self) -> List[str]:
        """Get received log messages."""
        if self._server:
            return self._server.get_received_messages()
        return []

    def get_received_records(self) -> List[logging.LogRecord]:
        """Get received log records."""
        if self._server:
            return self._server.get_received_records()
        return []

    def clear_records(self) -> None:
        """Clear received records."""
        if self._server:
            self._server.clear_records()

    def __enter__(self):
        """Context manager entry - starts the server."""
        if not self.is_running:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the server."""
        self.stop()
        return False


def setup_basic_logging(
        level: int = logging.INFO,
        console: bool = True,
        log_file: Optional[str] = None
) -> None:
    """
    Setup basic logging configuration.

    Args:
        level: Logging level
        console: Enable console logging
        log_file: Optional file path for file logging
    """
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(level)

    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT
        )
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)


def setup_tcp_logging(
        host: str,
        port: int,
        level: int = logging.INFO,
        use_ssl: bool = True,
        ssl_cafile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None
) -> JSONSocketHandler:
    """
    Setup TCP logging with JSON serialization.

    Args:
        host: Server host
        port: Server port
        level: Logging level
        use_ssl: Enable SSL
        ssl_cafile: CA certificate file
        ssl_certfile: Client certificate file
        ssl_keyfile: Client key file

    Returns:
        Configured handler
    """
    handler = JSONSocketHandler(
        host=host,
        port=port,
        use_ssl=use_ssl,
        ssl_cafile=ssl_cafile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile
    )
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    return handler


@contextmanager
def tcp_log_server_context(host: str = 'localhost', port: int = 0, **kwargs):
    """
    Context manager for TCP log server that auto-starts and stops.

    Example:
        with tcp_log_server_context('localhost', 9020) as server:
            logger = logging.getLogger('test')
            handler = setup_tcp_logging('localhost', server.port, use_ssl=False)
            logger.info("Test message")
    """
    server = TCPLogServer(host, port, **kwargs)
    server.start()
    try:
        yield server
    finally:
        server.stop()
