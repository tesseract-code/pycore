"""
Subsystem RPC Framework
=======================
Zero-boilerplate remote procedure calls with multiple transport backends.

Similar to: Celery, Ray, PyRO, multiprocessing.managers
Optimized for: Low latency, fault tolerance, transparency
"""

import asyncio
import inspect
import logging
import multiprocessing
import multiprocessing as mp
import pickle
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Type

import zmq
import zmq.asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SubsystemRPC")


# =============================================================================
# Core Protocol
# =============================================================================
class Transport(Enum):
    """Communication backend types."""
    ZMQ = "zmq"
    PIPE = "pipe"
    QUEUE = "queue"
    SHARED_MEMORY = "shm"


@dataclass
class RPCRequest:
    """Standardized RPC request message."""
    subsystem: str
    method: str
    args: tuple
    kwargs: dict
    request_id: str = field(default_factory=lambda: str(id(object())))


@dataclass
class RPCResponse:
    """Standardized RPC response message."""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None


def private(func):
    """Decorator to mark methods as private (not exposed via RPC)."""
    func._rpc_private = True
    return func


# =============================================================================
# Transport Abstraction Layer
# =============================================================================
class BaseTransport(ABC):
    """Abstract base for transport backends."""

    @abstractmethod
    def send(self, data: bytes) -> None:
        """Send serialized data."""
        pass

    @abstractmethod
    def recv(self, timeout: Optional[float] = None) -> bytes:
        """Receive serialized data with optional timeout."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanup resources."""
        pass


class ZMQTransport(BaseTransport):
    """ZMQ REQ/REP transport (lowest latency for network IPC)."""

    def __init__(self, endpoint: str, is_server: bool = False):
        self.ctx = zmq.Context()
        if is_server:
            self.sock = self.ctx.socket(zmq.REP)
            self.sock.bind(endpoint)
        else:
            self.sock = self.ctx.socket(zmq.REQ)
            self.sock.connect(endpoint)
        self.sock.setsockopt(zmq.LINGER, 0)

    def send(self, data: bytes) -> None:
        self.sock.send(data)

    def recv(self, timeout: Optional[float] = None) -> bytes:
        if timeout:
            if self.sock.poll(int(timeout * 1000)):
                return self.sock.recv()
            raise TimeoutError("Receive timeout")
        return self.sock.recv()

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


class QueueTransport(BaseTransport):
    """Thread-safe queue transport (best for threads)."""

    def __init__(self, send_queue: multiprocessing.Queue,
                 recv_queue: multiprocessing.Queue):
        self.send_q = send_queue
        self.recv_q = recv_queue

    def send(self, data: bytes) -> None:
        self.send_q.put(data)

    def recv(self, timeout: Optional[float] = None) -> bytes:
        try:
            return self.recv_q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("Receive timeout")

    def close(self) -> None:
        pass  # Queues don't need explicit cleanup


class PipeTransport(BaseTransport):
    """Multiprocessing pipe transport (lowest latency for local IPC)."""

    def __init__(self, conn: Connection):
        self.conn = conn

    def send(self, data: bytes) -> None:
        self.conn.send_bytes(data)

    def recv(self, timeout: Optional[float] = None) -> bytes:
        if timeout:
            if self.conn.poll(timeout):
                return self.conn.recv_bytes()
            raise TimeoutError("Receive timeout")
        return self.conn.recv_bytes()

    def close(self) -> None:
        self.conn.close()


# =============================================================================
# Serialization Layer
# =============================================================================
class Serializer:
    """High-performance serialization with protocol 5 (Python 3.8+)."""

    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Serialize with highest protocol for speed."""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize with safety checks."""
        return pickle.loads(data)


# =============================================================================
# Proxy Generation (Client Side)
# =============================================================================
class SubsystemProxy:
    """Dynamic proxy that intercepts method calls and forwards via RPC."""

    def __init__(self, subsystem_name: str, transport: BaseTransport,
                 timeout: float = 5.0):
        self._subsystem_name = subsystem_name
        self._transport = transport
        self._timeout = timeout

    def __getattr__(self, name: str):
        """Intercept method calls and create RPC callable."""

        def rpc_call(*args, **kwargs):
            # Create request
            request = RPCRequest(
                subsystem=self._subsystem_name,
                method=name,
                args=args,
                kwargs=kwargs
            )

            # Serialize and send
            data = Serializer.serialize(request)
            self._transport.send(data)

            # Wait for response
            try:
                resp_data = self._transport.recv(timeout=self._timeout)
                response: RPCResponse = Serializer.deserialize(resp_data)

                if response.success:
                    return response.result
                else:
                    raise RuntimeError(f"Remote error: {response.error}")

            except TimeoutError:
                raise TimeoutError(
                    f"RPC timeout for {self._subsystem_name}.{name}"
                )

        return rpc_call

    def close(self):
        """Close transport connection."""
        self._transport.close()


class AsyncSubsystemProxy:
    """Async version of SubsystemProxy for async subsystems."""

    def __init__(self, subsystem_name: str, endpoint: str,
                 timeout: float = 5.0):
        self._subsystem_name = subsystem_name
        self._endpoint = endpoint
        self._timeout = timeout
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(endpoint)
        self._sock.setsockopt(zmq.LINGER, 0)

    def __getattr__(self, name: str):
        """Intercept async method calls."""

        async def async_rpc_call(*args, **kwargs):
            request = RPCRequest(
                subsystem=self._subsystem_name,
                method=name,
                args=args,
                kwargs=kwargs
            )

            data = Serializer.serialize(request)
            await self._sock.send(data)

            if await self._sock.poll(int(self._timeout * 1000)):
                resp_data = await self._sock.recv()
                response: RPCResponse = Serializer.deserialize(resp_data)

                if response.success:
                    return response.result
                else:
                    raise RuntimeError(f"Remote error: {response.error}")
            else:
                raise TimeoutError(f"Async RPC timeout for {name}")

        return async_rpc_call

    async def close(self):
        self._sock.close()
        self._ctx.term()


# =============================================================================
# Subsystem Manager (Server Side)
# =============================================================================
class SubsystemManager:
    """Manages subsystem instances and handles RPC requests."""

    def __init__(self, subsystem_instance: Any, transport: BaseTransport):
        self._instance = subsystem_instance
        self._transport = transport
        self._running = False

        # Extract public methods (not starting with _ and not marked private)
        self._methods = {}
        for name, method in inspect.getmembers(subsystem_instance,
                                               predicate=inspect.ismethod):
            if not name.startswith('_') and not getattr(method,
                                                        '_rpc_private', False):
                self._methods[name] = method

        logger.info(f"Manager: Exposed methods: {list(self._methods.keys())}")

    def _handle_request(self, request: RPCRequest) -> RPCResponse:
        """Process a single RPC request."""
        try:
            method = self._methods.get(request.method)
            if not method:
                raise AttributeError(
                    f"Method '{request.method}' not found or not exposed"
                )

            # Call the actual method
            result = method(*request.args, **request.kwargs)

            return RPCResponse(
                request_id=request.request_id,
                success=True,
                result=result
            )

        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
            return RPCResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )

    def run(self):
        """Main server loop (blocking)."""
        self._running = True
        logger.info(f"Manager running for {self._instance.__class__.__name__}")

        while self._running:
            try:
                # Receive request
                req_data = self._transport.recv(timeout=0.5)
                request: RPCRequest = Serializer.deserialize(req_data)

                # Process and respond
                response = self._handle_request(request)
                resp_data = Serializer.serialize(response)
                self._transport.send(resp_data)

            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Manager error: {e}", exc_info=True)

    def stop(self):
        """Stop the manager loop."""
        self._running = False
        self._transport.close()


class AsyncSubsystemManager:
    """Async version for async subsystems."""

    def __init__(self, subsystem_instance: Any, endpoint: str):
        self._instance = subsystem_instance
        self._endpoint = endpoint
        self._running = False

        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(endpoint)
        self._sock.setsockopt(zmq.LINGER, 0)

        # Extract async methods
        self._methods = {}
        for name, method in inspect.getmembers(subsystem_instance):
            if (asyncio.iscoroutinefunction(method) and
                    not name.startswith('_') and
                    not getattr(method, '_rpc_private', False)):
                self._methods[name] = method

        logger.info(
            f"AsyncManager: Exposed methods: {list(self._methods.keys())}")

    async def _handle_request(self, request: RPCRequest) -> RPCResponse:
        """Process async RPC request."""
        try:
            method = self._methods.get(request.method)
            if not method:
                raise AttributeError(f"Method '{request.method}' not found")

            result = await method(*request.args, **request.kwargs)

            return RPCResponse(
                request_id=request.request_id,
                success=True,
                result=result
            )

        except Exception as e:
            logger.error(f"Async handler error: {e}", exc_info=True)
            return RPCResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )

    async def run(self):
        """Async server loop."""
        self._running = True
        logger.info("AsyncManager running")

        while self._running:
            try:
                if await self._sock.poll(500):
                    req_data = await self._sock.recv()
                    request: RPCRequest = Serializer.deserialize(req_data)

                    response = await self._handle_request(request)
                    resp_data = Serializer.serialize(response)
                    await self._sock.send(resp_data)

            except Exception as e:
                logger.error(f"AsyncManager error: {e}", exc_info=True)

    async def stop(self):
        self._running = False
        self._sock.close()
        self._ctx.term()


# =============================================================================
# Factory Functions
# =============================================================================
def create_subsystem_proxy(subsystem_name: str, transport_type: Transport,
                           endpoint: str, **kwargs) -> SubsystemProxy:
    """Factory: Create a client proxy."""

    if transport_type == Transport.ZMQ:
        transport = ZMQTransport(endpoint, is_server=False)
    else:
        raise NotImplementedError(f"Transport {transport_type} not supported")

    return SubsystemProxy(subsystem_name, transport, **kwargs)


def create_subsystem_manager(subsystem_instance: Any, transport_type: Transport,
                             endpoint: str) -> SubsystemManager:
    """Factory: Create a server manager."""

    if transport_type == Transport.ZMQ:
        transport = ZMQTransport(endpoint, is_server=True)
    else:
        raise NotImplementedError(f"Transport {transport_type} not supported")

    return SubsystemManager(subsystem_instance, transport)


# =============================================================================
# Process/Thread Runners
# =============================================================================
def _process_manager_target(subsystem_class: Type, transport_type: Transport,
                            endpoint: str, init_args: tuple, init_kwargs: dict):
    """Top-level function for process target (must be picklable)."""
    instance = subsystem_class(*init_args, **init_kwargs)
    manager = create_subsystem_manager(instance, transport_type, endpoint)
    manager.run()


def run_manager_in_process(subsystem_class: Type, transport_type: Transport,
                           endpoint: str, *args, **kwargs) -> mp.Process:
    """Run manager in separate process."""
    proc = mp.Process(
        target=_process_manager_target,
        args=(subsystem_class, transport_type, endpoint, args, kwargs),
        daemon=True
    )
    proc.start()
    return proc


def run_manager_in_thread(subsystem_instance: Any, transport_type: Transport,
                          endpoint: str) -> threading.Thread:
    """Run manager in separate thread."""

    manager = create_subsystem_manager(subsystem_instance, transport_type,
                                       endpoint)

    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    return thread


# =============================================================================
# Decorator for Zero-Boilerplate Registration
# =============================================================================
_subsystem_registry: Dict[str, Type] = {}


def subsystem(name: Optional[str] = None):
    """Decorator to register a class as a subsystem."""

    def decorator(cls: Type) -> Type:
        subsystem_name = name or cls.__name__
        _subsystem_registry[subsystem_name] = cls
        cls._subsystem_name = subsystem_name
        logger.info(f"Registered subsystem: {subsystem_name}")
        return cls

    return decorator


def get_subsystem(name: str) -> Type:
    """Retrieve registered subsystem by name."""
    return _subsystem_registry[name]


# =============================================================================
# Example Usage
# =============================================================================

# IMPORTANT: Classes must be defined at module level (not in __main__)
# to be picklable across processes
@subsystem(name="Calculator")
class Calculator:
    """Example subsystem for demonstration."""

    def __init__(self):
        self.history = []

    def add(self, a: int, b: int) -> int:
        result = a + b
        self.history.append(f"add({a}, {b}) = {result}")
        return result

    def multiply(self, a: int, b: int) -> int:
        return a * b

    @private
    def clear_history(self):
        """This method won't be exposed via RPC."""
        self.history.clear()

    def get_history(self) -> list:
        return self.history


def run_example():
    """Run the example (call this from main)."""
    import time

    # Start manager in separate process
    endpoint = "tcp://127.0.0.1:9999"
    proc = run_manager_in_process(Calculator, Transport.ZMQ, endpoint)

    time.sleep(0.5)  # Let server start

    # Create proxy and use it
    calc_proxy = create_subsystem_proxy("Calculator", Transport.ZMQ, endpoint)

    print("Testing Calculator subsystem via RPC:")
    print(f"add(5, 3) = {calc_proxy.add(5, 3)}")
    print(f"multiply(4, 7) = {calc_proxy.multiply(4, 7)}")
    print(f"add(10, 20) = {calc_proxy.add(10, 20)}")
    print(f"History: {calc_proxy.get_history()}")

    # Try calling private method (should fail)
    try:
        calc_proxy.clear_history()
    except RuntimeError as e:
        print(f"Expected error: {e}")

    calc_proxy.close()
    proc.terminate()
    proc.join()

    print("Done!")


if __name__ == "__main__":
    # Required for Windows compatibility
    mp.set_start_method('spawn', force=True)
    run_example()
