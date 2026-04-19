#!/usr/bin/env python3
"""
network_connect.py - Production-grade module for TCP/UDP connections.

This module provides synchronous and asynchronous utilities to connect,
send, receive, and check the availability of remote services over TCP or UDP.
All operations are fully typed, use context managers, and support configurable
timeouts and retries.
"""

import asyncio
import logging
import socket
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Protocol",
    "ConnectionConfig",
    "Connection",
    "AsyncConnection",
    "is_port_alive",
    "tcp_send_receive",
    "udp_send_receive",
    "async_is_port_alive",
    "async_tcp_send_receive",
    "async_udp_send_receive",
]

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: float = 5.0       # Default socket timeout (seconds)
DEFAULT_RETRIES: int = 3           # Number of connection attempts
DEFAULT_BUFFER_SIZE: int = 4096    # Receive buffer size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class Protocol(Enum):
    """Supported transport protocols."""
    TCP = auto()
    UDP = auto()


@dataclass(frozen=True)
class ConnectionConfig:
    """
    Immutable configuration for a network connection.

    Attributes:
        host:           Remote hostname or IP address.
        port:           Remote port number (1-65535).
        protocol:       Protocol to use (TCP or UDP).
        timeout:        Socket timeout in seconds.
        retries:        Number of connection attempts (minimum 1).
        buffer_size:    Size of the receive buffer.
        address_family: Socket address family (AF_INET or AF_INET6).
                        Defaults to AF_INET; pass AF_INET6 for IPv6 hosts.
    """
    host: str
    port: int
    protocol: Protocol = Protocol.TCP
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    buffer_size: int = DEFAULT_BUFFER_SIZE
    # FIX #8: expose address_family so callers can opt into IPv6
    address_family: socket.AddressFamily = field(default=socket.AF_INET)

    def __post_init__(self) -> None:
        # FIX #3: clamp retries to at least 1 so connect() always tries once
        if self.retries < 1:
            # frozen=True means we must use object.__setattr__
            object.__setattr__(self, "retries", 1)


# ---------------------------------------------------------------------------
# Synchronous Connection Class
# ---------------------------------------------------------------------------

class Connection:
    """
    Synchronous network connection (TCP/UDP) with automatic resource management.

    Implements the context manager protocol so it can be used with ``with``.
    Supports sending and receiving data, reconnection, and liveness checks.

    Example::

        with Connection("example.com", 80) as conn:
            conn.send(b"GET / HTTP/1.0\\r\\n\\r\\n")
            data = conn.receive()
            print(data)

    For UDP, ``connect()`` associates the socket with the remote address so
    that ``send()`` / ``receive()`` work without specifying the peer each time.
    """

    def __init__(
        self,
        host: str,
        port: int,
        protocol: Protocol = Protocol.TCP,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        address_family: socket.AddressFamily = socket.AF_INET,
    ) -> None:
        self.config = ConnectionConfig(
            host, port, protocol, timeout, retries, buffer_size, address_family
        )
        self._socket: Optional[socket.socket] = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish the network connection with retries.

        Raises:
            ConnectionError: If all connection attempts fail.
        """
        if self._connected:
            return

        last_exception: Optional[Exception] = None
        sock_type = (
            socket.SOCK_STREAM
            if self.config.protocol == Protocol.TCP
            else socket.SOCK_DGRAM
        )

        for attempt in range(1, self.config.retries + 1):
            try:
                sock = socket.socket(self.config.address_family, sock_type)
                sock.settimeout(self.config.timeout)
                # FIX #9: the TCP and UDP branches were identical; one call suffices.
                # For TCP this establishes a connection; for UDP it merely
                # associates the socket with the remote address so send()/recv()
                # can be used without specifying the peer on every call.
                sock.connect((self.config.host, self.config.port))
                self._socket = sock
                self._connected = True
                logger.info(
                    "Connected to %s:%d via %s",
                    self.config.host,
                    self.config.port,
                    self.config.protocol.name,
                )
                return

            except (socket.timeout, socket.error) as exc:
                last_exception = exc
                logger.warning(
                    "Connection attempt %d/%d failed: %s",
                    attempt,
                    self.config.retries,
                    exc,
                )

        raise ConnectionError(
            f"Failed to connect to {self.config.host}:{self.config.port} "
            f"after {self.config.retries} attempt(s)"
        ) from last_exception

    def disconnect(self) -> None:
        """Close the socket and reset the connection state."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception as exc:
                logger.error("Error closing socket: %s", exc)
            finally:
                self._socket = None
                self._connected = False
                logger.debug("Disconnected")

    def is_alive(self) -> bool:
        """
        Check whether the connection is still alive (TCP only).

        For UDP, always returns ``True`` because UDP is connectionless and
        there is no reliable way to detect peer closure without a heartbeat.

        Returns:
            ``True`` if the connection appears to be alive, ``False`` otherwise.
        """
        if not self._connected or self._socket is None:
            return False
        if self.config.protocol == Protocol.UDP:
            return True

        # FIX #2 & #7: capture original_timeout *before* the try block so it
        # is always defined in finally; use settimeout(0) for non-blocking peek
        # instead of MSG_DONTWAIT (Linux-only).  MSG_PEEK is cross-platform.
        original_timeout = self._socket.gettimeout()
        try:
            self._socket.settimeout(0.0)
            self._socket.recv(1, socket.MSG_PEEK)
            return True
        except (BlockingIOError, socket.timeout):
            # No data pending but socket is alive
            return True
        except Exception:
            return False
        finally:
            try:
                self._socket.settimeout(original_timeout)
            except Exception:
                pass

    def send(self, data: bytes) -> None:
        """
        Send *all* bytes over the connection.

        FIX #1 & #5: uses ``sendall()`` (guaranteed delivery of all bytes)
        and returns ``None`` to match the async counterpart.

        Args:
            data: Bytes to send.

        Raises:
            ConnectionError: If not connected or sending fails.
        """
        if not self._connected or self._socket is None:
            raise ConnectionError("Not connected")
        try:
            self._socket.sendall(data)
        except Exception as exc:
            raise ConnectionError(f"Send failed: {exc}") from exc

    def receive(self, max_bytes: Optional[int] = None) -> bytes:
        """
        Receive data from the connection.

        FIX #6: an empty-bytes result from the OS means the peer has closed
        the connection; this is now surfaced as ``ConnectionError`` rather than
        silently returning an empty buffer.

        Args:
            max_bytes: Maximum number of bytes to receive.
                       Defaults to ``config.buffer_size``.

        Returns:
            Received bytes.

        Raises:
            ConnectionError: If not connected, receive fails, or peer closed.
            TimeoutError:    If the receive times out.
        """
        if not self._connected or self._socket is None:
            raise ConnectionError("Not connected")
        buf = max_bytes if max_bytes is not None else self.config.buffer_size
        try:
            data = self._socket.recv(buf)
        except socket.timeout as exc:
            raise TimeoutError(
                f"Receive timed out after {self.config.timeout}s"
            ) from exc
        except Exception as exc:
            raise ConnectionError(f"Receive failed: {exc}") from exc

        if not data:
            raise ConnectionError("Connection closed by peer")
        return data

    def reconnect(self) -> None:
        """Force a reconnection: disconnect then connect."""
        self.disconnect()
        self.connect()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Connection":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        state = "connected" if self._connected else "disconnected"
        return (
            f"<Connection {self.config.protocol.name} "
            f"{self.config.host}:{self.config.port} [{state}]>"
        )


# ---------------------------------------------------------------------------
# Synchronous Helper Functions
# ---------------------------------------------------------------------------

def is_port_alive(
    host: str,
    port: int,
    timeout: float = 0.1,
    address_family: socket.AddressFamily = socket.AF_INET,
) -> bool:
    """
    Quick check whether a remote TCP port is accepting connections.

    Args:
        host:           Remote hostname or IP address.
        port:           Remote port number.
        timeout:        Connection timeout in seconds.
        address_family: ``AF_INET`` (default) or ``AF_INET6``.

    Returns:
        ``True`` if a TCP connection can be established, ``False`` otherwise.
    """
    try:
        with socket.socket(address_family, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0
    except Exception:
        return False


def tcp_send_receive(
    host: str,
    port: int,
    data: bytes,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> bytes:
    """
    One-shot TCP request/response.

    Opens a TCP connection, sends *data*, and returns the response.

    Args:
        host:    Remote hostname or IP address.
        port:    Remote port number.
        data:    Bytes to send.
        timeout: Socket timeout in seconds.
        retries: Number of connection attempts.

    Returns:
        Received bytes.

    Raises:
        ConnectionError: If connection or communication fails.
    """
    with Connection(host, port, Protocol.TCP, timeout, retries) as conn:
        conn.send(data)
        return conn.receive()


def udp_send_receive(
    host: str,
    port: int,
    data: bytes,
    timeout: float = DEFAULT_TIMEOUT,
) -> bytes:
    """
    One-shot UDP request/response.

    Sends a datagram and waits for a reply from the same peer.

    Args:
        host:    Remote hostname or IP address.
        port:    Remote port number.
        data:    Bytes to send.
        timeout: Socket timeout in seconds.

    Returns:
        Received bytes.

    Raises:
        ConnectionError: If sending or receiving fails.
    """
    with Connection(host, port, Protocol.UDP, timeout, retries=1) as conn:
        conn.send(data)
        return conn.receive()


# ---------------------------------------------------------------------------
# Asynchronous Implementation (asyncio)
# ---------------------------------------------------------------------------

class AsyncConnection:
    """
    Asynchronous version of :class:`Connection` using ``asyncio``.

    Provides the same interface but with ``async``/``await`` methods.
    Uses exponential backoff between retry attempts.

    Example::

        config = ConnectionConfig("example.com", 80)
        async with AsyncConnection(config) as conn:
            await conn.send(b"GET / HTTP/1.0\\r\\n\\r\\n")
            data = await conn.receive()
    """

    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        # FIX #4: typed as the concrete subclass so get_datagram() is visible
        self._udp_protocol: Optional[_UDPClientProtocol] = None
        self._connected: bool = False

    async def connect(self) -> None:
        """Establish the connection asynchronously with exponential backoff."""
        if self._connected:
            return

        last_exception: Optional[Exception] = None
        for attempt in range(1, self.config.retries + 1):
            try:
                if self.config.protocol == Protocol.TCP:
                    self._reader, self._writer = await asyncio.wait_for(
                        asyncio.open_connection(self.config.host, self.config.port),
                        timeout=self.config.timeout,
                    )
                else:  # UDP
                    loop = asyncio.get_running_loop()
                    transport, protocol = await loop.create_datagram_endpoint(
                        _UDPClientProtocol,
                        remote_addr=(self.config.host, self.config.port),
                    )
                    self._transport = transport
                    self._udp_protocol = protocol  # type: ignore[assignment]
                self._connected = True
                logger.info(
                    "Async connected to %s:%d via %s",
                    self.config.host,
                    self.config.port,
                    self.config.protocol.name,
                )
                return

            except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
                last_exception = exc
                logger.warning(
                    "Async attempt %d/%d failed: %s",
                    attempt,
                    self.config.retries,
                    exc,
                )
                if attempt < self.config.retries:
                    # FIX #11: exponential backoff with a 30 s ceiling
                    backoff = min(0.5 * (2 ** (attempt - 1)), 30.0)
                    await asyncio.sleep(backoff)

        raise ConnectionError(
            f"Async connection failed to {self.config.host}:{self.config.port} "
            f"after {self.config.retries} attempt(s)"
        ) from last_exception

    async def disconnect(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None
        if self._transport is not None:
            self._transport.close()
            self._transport = None
            self._udp_protocol = None
        self._connected = False

    async def send(self, data: bytes) -> None:
        """
        Send data asynchronously.

        FIX #5: returns ``None`` (matching sync ``Connection.send()``).

        Args:
            data: Bytes to send.

        Raises:
            ConnectionError: If not connected or sending fails.
        """
        if not self._connected:
            raise ConnectionError("Not connected")
        try:
            if self._writer is not None:  # TCP
                self._writer.write(data)
                await self._writer.drain()
            elif self._transport is not None:  # UDP
                self._transport.sendto(data)
            else:
                raise ConnectionError("No transport available")
        except ConnectionError:
            raise
        except Exception as exc:
            raise ConnectionError(f"Async send failed: {exc}") from exc

    async def receive(self, max_bytes: Optional[int] = None) -> bytes:
        """
        Receive data asynchronously.

        FIX #6: raises ``ConnectionError`` when the peer closes the connection
        (empty-bytes result) instead of returning empty bytes silently.

        Args:
            max_bytes: Maximum bytes to read. Defaults to ``config.buffer_size``.

        Returns:
            Received bytes.

        Raises:
            ConnectionError: If not connected, receive fails, or peer closed.
            TimeoutError:    If the receive times out.
        """
        if not self._connected:
            raise ConnectionError("Not connected")
        buf = max_bytes if max_bytes is not None else self.config.buffer_size
        try:
            if self._reader is not None:  # TCP
                data = await asyncio.wait_for(
                    self._reader.read(buf), timeout=self.config.timeout
                )
            elif self._udp_protocol is not None:  # UDP
                data = await asyncio.wait_for(
                    self._udp_protocol.get_datagram(),
                    timeout=self.config.timeout,
                )
            else:
                raise ConnectionError("No reader/protocol available")
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Async receive timed out after {self.config.timeout}s"
            ) from exc
        except ConnectionError:
            raise
        except Exception as exc:
            raise ConnectionError(f"Async receive failed: {exc}") from exc

        if not data:
            raise ConnectionError("Connection closed by peer")
        return data

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncConnection":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    def __repr__(self) -> str:
        state = "connected" if self._connected else "disconnected"
        return (
            f"<AsyncConnection {self.config.protocol.name} "
            f"{self.config.host}:{self.config.port} [{state}]>"
        )


class _UDPClientProtocol(asyncio.DatagramProtocol):
    """Internal helper to collect a single datagram from a UDP server."""

    def __init__(self) -> None:
        self._datagram_future: Optional[asyncio.Future] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._datagram_future = asyncio.get_running_loop().create_future()

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        if self._datagram_future and not self._datagram_future.done():
            self._datagram_future.set_result(data)

    def get_datagram(self) -> asyncio.Future:
        if self._datagram_future is None:
            raise RuntimeError("Protocol not yet initialized (connection_made not called)")
        return self._datagram_future


# ---------------------------------------------------------------------------
# Asynchronous Helper Functions
# ---------------------------------------------------------------------------

async def async_is_port_alive(host: str, port: int, timeout: float = 0.1) -> bool:
    """Asynchronously check whether a TCP port is accepting connections."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def async_tcp_send_receive(
    host: str,
    port: int,
    data: bytes,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> bytes:
    """One-shot asynchronous TCP request/response."""
    config = ConnectionConfig(host, port, Protocol.TCP, timeout, retries)
    async with AsyncConnection(config) as conn:
        await conn.send(data)
        return await conn.receive()


async def async_udp_send_receive(
    host: str,
    port: int,
    data: bytes,
    timeout: float = DEFAULT_TIMEOUT,
) -> bytes:
    """One-shot asynchronous UDP request/response."""
    config = ConnectionConfig(host, port, Protocol.UDP, timeout, retries=1)
    async with AsyncConnection(config) as conn:
        await conn.send(data)
        return await conn.receive()