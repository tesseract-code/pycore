import json
import logging
import socket
import socketserver
import ssl
import struct
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from pycore.circuit import CircuitBreaker
from pycore.log.constants import DEFAULT_TIMEOUT, MAX_MESSAGE_SIZE
from pycore.log.record import LogRecordData
from pycore.retry import retry


class JSONSocketHandler(logging.Handler):
    """
    A logging handler that sends log records as JSON over a TCP socket.

    The handler establishes a persistent TCP connection (optionally secured with TLS)
    to a remote server and transmits each log record as a JSON object preceded by
    a 4‑byte big-endian length prefix. It includes a `CircuitBreaker` to suspend
    sending after consecutive failures and transparent retry logic for socket creation.

    Parameters
    ----------
    host : str
        Hostname or IP address of the log server.
    port : int
        TCP port of the log server (1-65535).
    use_ssl : bool, optional
        If `True`, wrap the socket in a TLS connection (default `True`).
    ssl_cafile : str or None, optional
        Path to a CA certificate file for verifying the server. If `None`, the
        default system CA bundle is used. Ignored when `use_ssl` is `False`.
    ssl_certfile : str or None, optional
        Path to a client certificate file (PEM). Used together with `ssl_keyfile`
        for mutual TLS. Ignored when `use_ssl` is `False`.
    ssl_keyfile : str or None, optional
        Path to the private key file for the client certificate. Must be provided
        together with `ssl_certfile`. Ignored when `use_ssl` is `False`.
    timeout : float, optional
        Socket timeout in seconds (default defined by `DEFAULT_TIMEOUT`).

    Raises
    ------
    ValueError
        If `host` is empty or `port` is not in the valid range.
    FileNotFoundError
        If any of the SSL certificate/key files do not exist.
    RuntimeError
        If the SSL context cannot be initialized for any other reason.

    Attributes
    ----------
    host : str
        Target hostname or IP.
    port : int
        Target TCP port.
    use_ssl : bool
        Whether TLS is enabled.
    timeout : float
        Socket timeout in seconds.
    _breaker : CircuitBreaker
        Circuit breaker that controls sending attempts.
    _sock : socket.socket or None
        The underlying connected socket, or `None` if not connected.
    """

    def __init__(
            self,
            host: str,
            port: int,
            use_ssl: bool = True,
            ssl_cafile: Optional[str] = None,
            ssl_certfile: Optional[str] = None,
            ssl_keyfile: Optional[str] = None,
            timeout: float = DEFAULT_TIMEOUT
    ):
        super().__init__()

        # Validate inputs
        self._validate_host(host)
        self._validate_port(port)

        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.timeout = timeout
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile

        self._sock: Optional[socket.socket] = None
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._lock = threading.RLock()
        self._breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=1.0,
            half_open_max_calls=2,
            rolling_window_size=50,
            use_time_based_decay=True,
            decay_factor=0.5  # Reduce failures by half on each success
        )

        if self.use_ssl:
            self._setup_ssl_context()

    @staticmethod
    def _validate_host(host: str) -> None:
        """
        Validate the `host` parameter.

        Parameters
        ----------
        host : str
            Hostname or IP address.

        Raises
        ------
        ValueError
            If `host` is empty or not a string.
        """
        if not host or not isinstance(host, str):
            raise ValueError("Host must be a non-empty string")

    @staticmethod
    def _validate_port(port: int) -> None:
        """
        Validate the `port` parameter.

        Parameters
        ----------
        port : int
            TCP port number.

        Raises
        ------
        ValueError
            If `port` is not an integer or outside the range 1-65535.
        """
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError("Port must be an integer between 1 and 65535")

    def _setup_ssl_context(self) -> None:
        """
        Configure the SSL context with certificate verification.

        Creates a default SSL context requiring hostname verification.
        Optionally loads a custom CA bundle and client certificate/key pair.

        Raises
        ------
        FileNotFoundError
            If any specified certificate or key file does not exist.
        RuntimeError
            If the SSL context cannot be initialized.
        """
        try:
            self._ssl_context = ssl.create_default_context()

            self._ssl_context.check_hostname = True
            self._ssl_context.verify_mode = ssl.CERT_REQUIRED

            if self.ssl_cafile:
                if not Path(self.ssl_cafile).exists():
                    raise FileNotFoundError(
                        f"CA file not found: {self.ssl_cafile}")
                self._ssl_context.load_verify_locations(self.ssl_cafile)

            if self.ssl_certfile and self.ssl_keyfile:
                if not Path(self.ssl_certfile).exists():
                    raise FileNotFoundError(
                        f"Cert file not found: {self.ssl_certfile}")
                if not Path(self.ssl_keyfile).exists():
                    raise FileNotFoundError(
                        f"Key file not found: {self.ssl_keyfile}")
                self._ssl_context.load_cert_chain(
                    self.ssl_certfile,
                    self.ssl_keyfile
                )
        except Exception as e:
            raise RuntimeError(f"Failed to setup SSL context: {e}") from e

    def _make_socket(self) -> socket.socket:
        """
        Create a new socket and connect to the server, wrapping with SSL if configured.

        Returns
        -------
        socket.socket
            The connected (and optionally TLS-wrapped) socket.

        Raises
        ------
        ConnectionError
            If the connection cannot be established.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect((self.host, self.port))

            if self.use_ssl and self._ssl_context:
                sock = self._ssl_context.wrap_socket(
                    sock,
                    server_hostname=self.host
                )

            return sock
        except Exception:
            sock.close()
            raise

    def _get_socket(self) -> Optional[socket.socket]:
        """
        Return the current socket, creating a new one if needed and the circuit breaker allows.

        This method is thread-safe and respects the circuit breaker state.

        Returns
        -------
        socket.socket or None
            The open socket, or `None` if connection failed or the circuit is open.
        """
        with self._lock:
            if self._sock is None and self._breaker.can_execute():
                try:
                    self._sock = self._make_socket()
                except Exception:
                    # Connection failed, don't raise
                    return None
            return self._sock

    def _close_socket(self) -> None:
        """
        Close the socket connection and clear the internal reference.

        This method is thread-safe and ignores errors during closing.
        """
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                finally:
                    self._sock = None

    def emit(self, record: logging.LogRecord) -> None:
        """
        Transmit a log record over the network as JSON.

        Converts the `record` to a `LogRecordData` instance and serializes it
        to a JSON byte string with a 4-byte big-endian length prefix. If the
        serialized size exceeds `MAX_MESSAGE_SIZE`, the record is dropped.
        The method respects the circuit breaker state and uses a retry
        wrapper for socket creation.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to emit.

        Notes
        -----
        If sending fails or the socket becomes unusable, the socket is closed
        and the failure recorded in the circuit breaker. Unhandled exceptions
        are passed to `handleError`.
        """
        # Check circuit breaker first - avoid unnecessary work
        if not self._breaker.can_execute():
            return

        try:
            # Convert to JSON-safe format
            log_data = LogRecordData.from_log_record(record)
            json_str = json.dumps(asdict(log_data))
            json_bytes = json_str.encode('utf-8')

            # Check size limit
            if len(json_bytes) > MAX_MESSAGE_SIZE:
                raise ValueError(
                    f"Log message exceeds max size of {MAX_MESSAGE_SIZE} bytes")

            # Send with length prefix (same protocol as SocketHandler)
            data = struct.pack('>L', len(json_bytes)) + json_bytes

            decorated = retry(max_attempts=2)(self._get_socket)
            sock = decorated()

            if sock:
                try:
                    with self._lock:
                        sock.sendall(data)
                    self._breaker.record_success()  # Outside lock
                except (BrokenPipeError, ConnectionResetError, OSError):
                    self._close_socket()
                    self._breaker.record_failure()
                    # Don't re-raise - let handleError deal with it below
            else:
                # Socket creation failed after retries
                self._breaker.record_failure()

        except Exception as e:
            self._close_socket()
            self.handleError(record)

    def close(self) -> None:
        """
        Close the handler and release all resources.

        Closes the underlying socket and then calls the parent `close()` method.
        """
        self._close_socket()
        super().close()

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        Returns
        -------
        JSONSocketHandler
            The handler instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and close the handler.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred, else `None`.
        exc_val : BaseException or None
            Exception instance if an exception occurred, else `None`.
        exc_tb : traceback or None
            Traceback object if an exception occurred, else `None`.

        Returns
        -------
        bool
            `False` so that any exception is re-raised after cleanup.
        """
        self.close()
        return False


class JSONLogRecordStreamHandler(socketserver.StreamRequestHandler):
    """
    A stream request handler that reads JSON-encoded log records from a TCP connection.

    Designed to be used with `socketserver.TCPServer` or `socketserver.ThreadingTCPServer`.
    Each incoming connection is expected to send a stream of length-prefixed JSON
    messages (4‑byte big-endian prefix). Each message is deserialized into a
    `LogRecordData` object and passed to the server's `handle_log_record` method
    if it exists.
    """

    def handle(self) -> None:
        """
        Process all log records on this connection until the stream ends.

        The method runs a loop that reads the length prefix, then the JSON payload.
        Invalid data or connection errors cause the loop to terminate. Each valid
        record is forwarded to `_process_json_record`.

        Notes
        -----
        The loop is designed to handle multiple records in a single connection,
        enabling persistent log channels.
        """
        while True:
            try:
                # Read the length prefix
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break

                msg_len = struct.unpack('>L', chunk)[0]

                # Validate message size
                if msg_len > MAX_MESSAGE_SIZE:
                    logging.getLogger(__name__).error(
                        f"Message size {msg_len} exceeds max {MAX_MESSAGE_SIZE}"
                    )
                    break

                # Read the JSON data
                data = b''
                while len(data) < msg_len:
                    chunk = self.connection.recv(min(msg_len - len(data), 8192))
                    if not chunk:
                        raise ConnectionError(
                            "Connection closed while reading data")
                    data += chunk

                # Safely deserialize JSON
                self._process_json_record(data)

            except (EOFError, ConnectionResetError, ConnectionError):
                break
            except json.JSONDecodeError as e:
                logging.getLogger(__name__).error(f"Invalid JSON received: {e}")
                break
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Error handling log record: {e}")
                break

    def _process_json_record(self, data: bytes) -> None:
        """
        Deserialize a JSON byte string into a log record and pass it to the
        server.

        Parameters
        ----------
        data : bytes
            Raw JSON bytes representing a `LogRecordData` object.

        Raises
        ------
        json.JSONDecodeError
            If the data cannot be decoded as JSON.
        TypeError, ValueError
            If the decoded data does not match the `LogRecordData` schema.
        """
        try:
            json_data = json.loads(data.decode('utf-8'))
            log_data = LogRecordData(**json_data)
            record = log_data.to_log_record()

            # Pass to server for handling
            if hasattr(self.server, 'handle_log_record'):
                self.server.handle_log_record(record)
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to process log record: {e}")