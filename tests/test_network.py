"""
test_network.py - Comprehensive pytest suite for network_connect.py.

Coverage strategy
-----------------
- All public classes and functions are tested.
- Both happy-path and error-path branches are exercised.
- Sockets and asyncio streams are fully mocked so the tests are hermetic
  (no real network calls).
- Each original review finding gets an explicit regression test labelled
  FIX-N matching the review document.
"""

import asyncio
import socket
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from pycore.network import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    AsyncConnection,
    Connection,
    ConnectionConfig,
    Protocol,
    _UDPClientProtocol,
    async_is_port_alive,
    async_tcp_send_receive,
    async_udp_send_receive,
    is_port_alive,
    tcp_send_receive,
    udp_send_receive,
)


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

def _make_socket(
        recv_data: bytes = b"hello",
        send_error: Optional[Exception] = None,
        recv_error: Optional[Exception] = None,
        connect_error: Optional[Exception] = None,
        gettimeout_return: float = 5.0,
) -> MagicMock:
    """Return a fully-configured mock socket."""
    sock = MagicMock(spec=socket.socket)
    sock.gettimeout.return_value = gettimeout_return
    if connect_error:
        sock.connect.side_effect = connect_error
    if send_error:
        sock.sendall.side_effect = send_error
    else:
        sock.sendall.return_value = None
    if recv_error:
        sock.recv.side_effect = recv_error
    else:
        sock.recv.return_value = recv_data
    return sock


@pytest.fixture
def tcp_conn() -> Connection:
    return Connection("127.0.0.1", 9000, Protocol.TCP)


@pytest.fixture
def udp_conn() -> Connection:
    return Connection("127.0.0.1", 9000, Protocol.UDP)


# ===========================================================================
# ConnectionConfig
# ===========================================================================

class TestConnectionConfig:
    def test_defaults(self):
        cfg = ConnectionConfig("localhost", 80)
        assert cfg.protocol is Protocol.TCP
        assert cfg.timeout == DEFAULT_TIMEOUT
        assert cfg.retries == DEFAULT_RETRIES
        assert cfg.buffer_size == DEFAULT_BUFFER_SIZE
        assert cfg.address_family == socket.AF_INET

    def test_immutable(self):
        cfg = ConnectionConfig("localhost", 80)
        with pytest.raises((AttributeError, TypeError)):
            cfg.host = "other"  # type: ignore[misc]

    # FIX-3: retries=0 must be clamped to 1
    def test_retries_zero_clamped_to_one(self):
        cfg = ConnectionConfig("localhost", 80, retries=0)
        assert cfg.retries == 1

    def test_retries_negative_clamped_to_one(self):
        cfg = ConnectionConfig("localhost", 80, retries=-5)
        assert cfg.retries == 1

    def test_ipv6_address_family(self):
        cfg = ConnectionConfig("::1", 80, address_family=socket.AF_INET6)
        assert cfg.address_family == socket.AF_INET6


# ===========================================================================
# Connection – construction & repr
# ===========================================================================

class TestConnectionInit:
    def test_not_connected_on_init(self, tcp_conn):
        assert not tcp_conn._connected
        assert tcp_conn._socket is None

    def test_repr_disconnected(self, tcp_conn):
        r = repr(tcp_conn)
        assert "TCP" in r
        assert "disconnected" in r
        assert "127.0.0.1:9000" in r

    def test_repr_connected(self, tcp_conn):
        sock = _make_socket()
        tcp_conn._socket = sock
        tcp_conn._connected = True
        assert "connected" in repr(tcp_conn)


# ===========================================================================
# Connection.connect()
# ===========================================================================

class TestConnectionConnect:
    def test_tcp_connect_success(self, tcp_conn):
        sock = _make_socket()
        with patch("socket.socket", return_value=sock):
            tcp_conn.connect()
        assert tcp_conn._connected
        sock.connect.assert_called_once_with(("127.0.0.1", 9000))

    def test_udp_connect_success(self, udp_conn):
        sock = _make_socket()
        with patch("socket.socket", return_value=sock):
            udp_conn.connect()
        assert udp_conn._connected

    def test_connect_is_idempotent(self, tcp_conn):
        sock = _make_socket()
        with patch("socket.socket", return_value=sock):
            tcp_conn.connect()
            tcp_conn.connect()  # second call is a no-op
        assert sock.connect.call_count == 1

    def test_connect_retries_on_failure_then_succeeds(self, tcp_conn):
        good_sock = _make_socket()
        bad_sock = _make_socket(connect_error=socket.error("refused"))
        with patch("socket.socket", side_effect=[bad_sock, good_sock]):
            tcp_conn.connect()
        assert tcp_conn._connected

    def test_connect_raises_after_all_retries_exhausted(self):
        conn = Connection("127.0.0.1", 9000, retries=2)
        bad_sock = _make_socket(connect_error=socket.error("refused"))
        with patch("socket.socket", return_value=bad_sock):
            with pytest.raises(ConnectionError, match="after 2 attempt"):
                conn.connect()

    # FIX-3: retries clamped means we never skip the loop body
    def test_connect_with_retries_zero_still_attempts_once(self):
        conn = Connection("127.0.0.1", 9000, retries=0)
        assert conn.config.retries == 1
        sock = _make_socket()
        with patch("socket.socket", return_value=sock):
            conn.connect()
        assert conn._connected

    def test_sets_timeout_on_socket(self, tcp_conn):
        sock = _make_socket()
        with patch("socket.socket", return_value=sock):
            tcp_conn.connect()
        sock.settimeout.assert_called_with(DEFAULT_TIMEOUT)

    # FIX-8: address_family forwarded to socket.socket()
    def test_ipv6_address_family_passed_to_socket(self):
        conn = Connection("::1", 80, address_family=socket.AF_INET6)
        sock = _make_socket()
        with patch("socket.socket", return_value=sock) as mock_socket:
            conn.connect()
        mock_socket.assert_called_once_with(socket.AF_INET6, socket.SOCK_STREAM)


# ===========================================================================
# Connection.disconnect()
# ===========================================================================

class TestConnectionDisconnect:
    def test_disconnect_clears_state(self, tcp_conn):
        sock = _make_socket()
        tcp_conn._socket = sock
        tcp_conn._connected = True
        tcp_conn.disconnect()
        assert not tcp_conn._connected
        assert tcp_conn._socket is None
        sock.close.assert_called_once()

    def test_disconnect_when_not_connected_is_safe(self, tcp_conn):
        tcp_conn.disconnect()  # should not raise

    def test_disconnect_swallows_close_error(self, tcp_conn):
        sock = _make_socket()
        sock.close.side_effect = OSError("boom")
        tcp_conn._socket = sock
        tcp_conn._connected = True
        tcp_conn.disconnect()  # must not propagate
        assert not tcp_conn._connected


# ===========================================================================
# Connection.send()
# ===========================================================================

class TestConnectionSend:
    # FIX-1: sendall() is called, not send()
    def test_uses_sendall_not_send(self, tcp_conn):
        sock = _make_socket()
        tcp_conn._socket = sock
        tcp_conn._connected = True
        tcp_conn.send(b"data")
        sock.sendall.assert_called_once_with(b"data")
        sock.send.assert_not_called()

    # FIX-5: send() returns None
    def test_returns_none(self, tcp_conn):
        sock = _make_socket()
        tcp_conn._socket = sock
        tcp_conn._connected = True
        result = tcp_conn.send(b"x")
        assert result is None

    def test_raises_when_not_connected(self, tcp_conn):
        with pytest.raises(ConnectionError, match="Not connected"):
            tcp_conn.send(b"x")

    def test_wraps_socket_error(self, tcp_conn):
        sock = _make_socket(send_error=OSError("net fail"))
        tcp_conn._socket = sock
        tcp_conn._connected = True
        with pytest.raises(ConnectionError, match="Send failed"):
            tcp_conn.send(b"x")


# ===========================================================================
# Connection.receive()
# ===========================================================================

class TestConnectionReceive:
    def test_happy_path(self, tcp_conn):
        sock = _make_socket(recv_data=b"response")
        tcp_conn._socket = sock
        tcp_conn._connected = True
        assert tcp_conn.receive() == b"response"

    def test_uses_buffer_size_by_default(self, tcp_conn):
        sock = _make_socket(recv_data=b"x")
        tcp_conn._socket = sock
        tcp_conn._connected = True
        tcp_conn.receive()
        sock.recv.assert_called_once_with(DEFAULT_BUFFER_SIZE)

    def test_custom_max_bytes(self, tcp_conn):
        sock = _make_socket(recv_data=b"x")
        tcp_conn._socket = sock
        tcp_conn._connected = True
        tcp_conn.receive(max_bytes=128)
        sock.recv.assert_called_once_with(128)

    # FIX-6: empty bytes → ConnectionError
    def test_empty_bytes_raises_connection_error(self, tcp_conn):
        sock = _make_socket(recv_data=b"")
        tcp_conn._socket = sock
        tcp_conn._connected = True
        with pytest.raises(ConnectionError, match="closed by peer"):
            tcp_conn.receive()

    def test_timeout_raises_timeout_error(self, tcp_conn):
        sock = _make_socket(recv_error=socket.timeout("timed out"))
        tcp_conn._socket = sock
        tcp_conn._connected = True
        with pytest.raises(TimeoutError):
            tcp_conn.receive()

    def test_raises_when_not_connected(self, tcp_conn):
        with pytest.raises(ConnectionError, match="Not connected"):
            tcp_conn.receive()

    def test_other_error_raises_connection_error(self, tcp_conn):
        sock = _make_socket(recv_error=OSError("broken pipe"))
        tcp_conn._socket = sock
        tcp_conn._connected = True
        with pytest.raises(ConnectionError, match="Receive failed"):
            tcp_conn.receive()


# ===========================================================================
# Connection.is_alive()
# ===========================================================================

class TestConnectionIsAlive:
    def test_not_connected_returns_false(self, tcp_conn):
        assert not tcp_conn.is_alive()

    def test_udp_always_alive_when_connected(self, udp_conn):
        udp_conn._connected = True
        udp_conn._socket = _make_socket()
        assert udp_conn.is_alive()

    def test_alive_when_no_data_pending(self, tcp_conn):
        sock = _make_socket()
        sock.gettimeout.return_value = 5.0
        sock.recv.side_effect = BlockingIOError
        tcp_conn._socket = sock
        tcp_conn._connected = True
        assert tcp_conn.is_alive()

    def test_alive_on_socket_timeout_peek(self, tcp_conn):
        sock = _make_socket()
        sock.gettimeout.return_value = 5.0
        sock.recv.side_effect = socket.timeout
        tcp_conn._socket = sock
        tcp_conn._connected = True
        assert tcp_conn.is_alive()

    def test_not_alive_on_unexpected_error(self, tcp_conn):
        sock = _make_socket()
        sock.gettimeout.return_value = 5.0
        sock.recv.side_effect = OSError("reset by peer")
        tcp_conn._socket = sock
        tcp_conn._connected = True
        assert not tcp_conn.is_alive()

    # FIX-2: original_timeout must always be restored
    def test_timeout_restored_on_alive_check(self, tcp_conn):
        sock = _make_socket()
        sock.gettimeout.return_value = 7.0
        sock.recv.side_effect = BlockingIOError
        tcp_conn._socket = sock
        tcp_conn._connected = True
        tcp_conn.is_alive()
        # settimeout called: first to 0.0, then restored to 7.0
        calls = sock.settimeout.call_args_list
        assert call(0.0) in calls
        assert call(7.0) in calls

    # FIX-7: MSG_DONTWAIT must NOT be used
    def test_no_msg_dontwait(self, tcp_conn):
        sock = _make_socket(recv_data=b"x")
        sock.gettimeout.return_value = 5.0
        tcp_conn._socket = sock
        tcp_conn._connected = True
        tcp_conn.is_alive()
        # Verify recv was called with MSG_PEEK but not MSG_DONTWAIT
        recv_flags = sock.recv.call_args[0][1]
        assert recv_flags == socket.MSG_PEEK
        assert recv_flags & getattr(socket, "MSG_DONTWAIT", 0) == 0

    def test_alive_when_data_available(self, tcp_conn):
        sock = _make_socket(recv_data=b"x")
        sock.gettimeout.return_value = 5.0
        tcp_conn._socket = sock
        tcp_conn._connected = True
        assert tcp_conn.is_alive()


# ===========================================================================
# Connection.reconnect()
# ===========================================================================

class TestConnectionReconnect:
    def test_reconnect_calls_disconnect_then_connect(self, tcp_conn):
        with patch.object(tcp_conn, "disconnect") as mock_dc, \
                patch.object(tcp_conn, "connect") as mock_c:
            tcp_conn.reconnect()
        mock_dc.assert_called_once()
        mock_c.assert_called_once()


# ===========================================================================
# Context manager
# ===========================================================================

class TestConnectionContextManager:
    def test_enter_calls_connect(self, tcp_conn):
        with patch.object(tcp_conn, "connect") as mock_c, \
                patch.object(tcp_conn, "disconnect"):
            with tcp_conn:
                mock_c.assert_called_once()

    def test_exit_calls_disconnect_on_success(self, tcp_conn):
        with patch.object(tcp_conn, "connect"), \
                patch.object(tcp_conn, "disconnect") as mock_dc:
            with tcp_conn:
                pass
        mock_dc.assert_called_once()

    def test_exit_calls_disconnect_on_exception(self, tcp_conn):
        with patch.object(tcp_conn, "connect"), \
                patch.object(tcp_conn, "disconnect") as mock_dc:
            with pytest.raises(ValueError):
                with tcp_conn:
                    raise ValueError("oops")
        mock_dc.assert_called_once()


# ===========================================================================
# is_port_alive()
# ===========================================================================

class TestIsPortAlive:
    def test_returns_true_when_port_open(self):
        sock = MagicMock()
        sock.__enter__ = lambda s: s
        sock.__exit__ = MagicMock(return_value=False)
        sock.connect_ex.return_value = 0
        with patch("socket.socket", return_value=sock):
            assert is_port_alive("127.0.0.1", 80)

    def test_returns_false_when_port_closed(self):
        sock = MagicMock()
        sock.__enter__ = lambda s: s
        sock.__exit__ = MagicMock(return_value=False)
        sock.connect_ex.return_value = 111  # ECONNREFUSED
        with patch("socket.socket", return_value=sock):
            assert not is_port_alive("127.0.0.1", 80)

    def test_returns_false_on_exception(self):
        with patch("socket.socket", side_effect=OSError("fail")):
            assert not is_port_alive("127.0.0.1", 80)

    # FIX-8: address_family param forwarded
    def test_ipv6_family_forwarded(self):
        sock = MagicMock()
        sock.__enter__ = lambda s: s
        sock.__exit__ = MagicMock(return_value=False)
        sock.connect_ex.return_value = 0
        with patch("socket.socket", return_value=sock) as mock_socket:
            is_port_alive("::1", 80, address_family=socket.AF_INET6)
        mock_socket.assert_called_once_with(socket.AF_INET6, socket.SOCK_STREAM)


# ===========================================================================
# tcp_send_receive() / udp_send_receive()
# ===========================================================================

class TestSendReceiveHelpers:
    def test_tcp_send_receive(self):
        with patch.object(Connection, "connect"), \
                patch.object(Connection, "disconnect"), \
                patch.object(Connection, "send"), \
                patch.object(Connection, "receive", return_value=b"pong"):
            result = tcp_send_receive("127.0.0.1", 9000, b"ping")
        assert result == b"pong"

    def test_udp_send_receive(self):
        with patch.object(Connection, "connect"), \
                patch.object(Connection, "disconnect"), \
                patch.object(Connection, "send"), \
                patch.object(Connection, "receive", return_value=b"pong"):
            result = udp_send_receive("127.0.0.1", 9000, b"ping")
        assert result == b"pong"

    def test_tcp_send_receive_propagates_error(self):
        with patch.object(Connection, "connect",
                          side_effect=ConnectionError("fail")):
            with pytest.raises(ConnectionError):
                tcp_send_receive("127.0.0.1", 9000, b"x")


# ===========================================================================
# AsyncConnection
# ===========================================================================

def _make_async_stream(recv_data: bytes = b"hello"):
    """Return mock (StreamReader, StreamWriter) pair."""
    reader = AsyncMock(spec=asyncio.StreamReader)
    writer = AsyncMock(spec=asyncio.StreamWriter)
    reader.read.return_value = recv_data
    writer.drain = AsyncMock()
    writer.wait_closed = AsyncMock()
    return reader, writer


class TestAsyncConnectionInit:
    def test_repr_disconnected(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        assert "disconnected" in repr(conn)
        assert "TCP" in repr(conn)

    def test_repr_connected(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._connected = True
        assert "connected" in repr(conn)


class TestAsyncConnectionConnect:
    @pytest.mark.asyncio
    async def test_tcp_connect_success(self):
        reader, writer = _make_async_stream()
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        with patch("asyncio.open_connection", return_value=(reader, writer)), \
                patch("asyncio.wait_for",
                      new=AsyncMock(return_value=(reader, writer))):
            await conn.connect()
        assert conn._connected

    @pytest.mark.asyncio
    async def test_idempotent_connect(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._connected = True
        # Should return immediately without touching open_connection
        with patch("asyncio.open_connection") as mock_oc:
            await conn.connect()
        mock_oc.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_after_all_retries(self):
        cfg = ConnectionConfig("127.0.0.1", 9000, retries=2)
        conn = AsyncConnection(cfg)
        with patch("asyncio.wait_for",
                   new=AsyncMock(side_effect=OSError("refused"))), \
                patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ConnectionError, match="after 2 attempt"):
                await conn.connect()

    # FIX-11: exponential backoff (not linear)
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        cfg = ConnectionConfig("127.0.0.1", 9000, retries=3)
        conn = AsyncConnection(cfg)
        sleep_calls = []

        async def fake_sleep(n):
            sleep_calls.append(n)

        with patch("asyncio.wait_for",
                   new=AsyncMock(side_effect=OSError("refused"))), \
                patch("asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(ConnectionError):
                await conn.connect()

        # Two sleeps expected between attempts 1→2 and 2→3
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 0.5  # 0.5 * 2^0
        assert sleep_calls[1] == 1.0  # 0.5 * 2^1


class TestAsyncConnectionDisconnect:
    @pytest.mark.asyncio
    async def test_tcp_disconnect_closes_writer(self):
        reader, writer = _make_async_stream()
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._reader = reader
        conn._writer = writer
        conn._connected = True
        await conn.disconnect()
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        assert not conn._connected

    @pytest.mark.asyncio
    async def test_disconnect_safe_when_not_connected(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        await conn.disconnect()  # should not raise


class TestAsyncConnectionSend:
    @pytest.mark.asyncio
    async def test_tcp_send(self):
        reader, writer = _make_async_stream()
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._reader = reader
        conn._writer = writer
        conn._connected = True
        result = await conn.send(b"hello")
        assert result is None
        writer.write.assert_called_once_with(b"hello")
        writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_not_connected(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        with pytest.raises(ConnectionError, match="Not connected"):
            await conn.send(b"x")

    @pytest.mark.asyncio
    async def test_wraps_write_error(self):
        reader, writer = _make_async_stream()
        writer.write.side_effect = OSError("broken")
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._reader = reader
        conn._writer = writer
        conn._connected = True
        with pytest.raises(ConnectionError, match="Async send failed"):
            await conn.send(b"x")


class TestAsyncConnectionReceive:
    @pytest.mark.asyncio
    async def test_tcp_receive(self):
        reader, writer = _make_async_stream(recv_data=b"world")
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._reader = reader
        conn._writer = writer
        conn._connected = True
        with patch("asyncio.wait_for", new=AsyncMock(return_value=b"world")):
            data = await conn.receive()
        assert data == b"world"

    # FIX-6: empty bytes → ConnectionError
    @pytest.mark.asyncio
    async def test_empty_bytes_raises(self):
        reader, writer = _make_async_stream(recv_data=b"")
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._reader = reader
        conn._writer = writer
        conn._connected = True
        with patch("asyncio.wait_for", new=AsyncMock(return_value=b"")):
            with pytest.raises(ConnectionError, match="closed by peer"):
                await conn.receive()

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self):
        reader, writer = _make_async_stream()
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._reader = reader
        conn._writer = writer
        conn._connected = True
        with patch("asyncio.wait_for",
                   new=AsyncMock(side_effect=asyncio.TimeoutError)):
            with pytest.raises(TimeoutError):
                await conn.receive()

    @pytest.mark.asyncio
    async def test_raises_when_not_connected(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        with pytest.raises(ConnectionError, match="Not connected"):
            await conn.receive()

    @pytest.mark.asyncio
    async def test_no_reader_or_protocol_raises(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        conn._connected = True
        with pytest.raises(ConnectionError, match="No reader"):
            await conn.receive()


class TestAsyncConnectionContextManager:
    @pytest.mark.asyncio
    async def test_aenter_aexit(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        with patch.object(conn, "connect", new=AsyncMock()) as mock_c, \
                patch.object(conn, "disconnect", new=AsyncMock()) as mock_dc:
            async with conn:
                mock_c.assert_awaited_once()
        mock_dc.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_called_on_exception(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        conn = AsyncConnection(cfg)
        with patch.object(conn, "connect", new=AsyncMock()), \
                patch.object(conn, "disconnect", new=AsyncMock()) as mock_dc:
            with pytest.raises(RuntimeError):
                async with conn:
                    raise RuntimeError("boom")
        mock_dc.assert_awaited_once()


# ===========================================================================
# _UDPClientProtocol
# ===========================================================================

class TestUDPClientProtocol:
    def test_get_datagram_before_connection_made_raises(self):
        proto = _UDPClientProtocol()
        with pytest.raises(RuntimeError, match="not yet initialized"):
            proto.get_datagram()

    def test_datagram_received_sets_future(self):
        async def _run():
            loop = asyncio.get_running_loop()
            proto = _UDPClientProtocol()
            transport = MagicMock()
            proto.connection_made(transport)
            proto.datagram_received(b"data", ("127.0.0.1", 9000))
            result = await proto.get_datagram()
            assert result == b"data"

        asyncio.run(_run())

    def test_second_datagram_ignored(self):
        async def _run():
            proto = _UDPClientProtocol()
            proto.connection_made(MagicMock())
            proto.datagram_received(b"first", ("127.0.0.1", 9000))
            proto.datagram_received(b"second", ("127.0.0.1", 9000))
            result = await proto.get_datagram()
            assert result == b"first"

        asyncio.run(_run())


# ===========================================================================
# Async helper functions
# ===========================================================================

class TestAsyncIsPortAlive:
    @pytest.mark.asyncio
    async def test_returns_true_when_connectable(self):
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)
        writer.wait_closed = AsyncMock()
        with patch("asyncio.wait_for",
                   new=AsyncMock(return_value=(reader, writer))):
            assert await async_is_port_alive("127.0.0.1", 80)

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self):
        with patch("asyncio.wait_for",
                   new=AsyncMock(side_effect=OSError("refused"))):
            assert not await async_is_port_alive("127.0.0.1", 80)


class TestAsyncTcpSendReceive:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        cfg = ConnectionConfig("127.0.0.1", 9000)
        with patch.object(AsyncConnection, "connect", new=AsyncMock()), \
                patch.object(AsyncConnection, "disconnect", new=AsyncMock()), \
                patch.object(AsyncConnection, "send", new=AsyncMock()), \
                patch.object(AsyncConnection, "receive",
                             new=AsyncMock(return_value=b"pong")):
            result = await async_tcp_send_receive("127.0.0.1", 9000, b"ping")
        assert result == b"pong"


class TestAsyncUdpSendReceive:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        with patch.object(AsyncConnection, "connect", new=AsyncMock()), \
                patch.object(AsyncConnection, "disconnect", new=AsyncMock()), \
                patch.object(AsyncConnection, "send", new=AsyncMock()), \
                patch.object(AsyncConnection, "receive",
                             new=AsyncMock(return_value=b"pong")):
            result = await async_udp_send_receive("127.0.0.1", 9000, b"ping")
        assert result == b"pong"


# ===========================================================================
# __all__ coverage
# ===========================================================================

class TestPublicAPI:
    def test_all_exports_importable(self):
        import pycore.network as nc
        for name in nc.__all__:
            assert hasattr(nc,
                           name), f"{name!r} listed in __all__ but not importable"

    def test_private_class_not_in_all(self):
        import pycore.network as nc
        assert "_UDPClientProtocol" not in nc.__all__
