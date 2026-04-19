"""
High-performance Event-Driven Architecture with Auto-Pickling Interface

This module provides a scalable, asyncio-based event-driven system with:
- EventDriven interface for event publishing/handling
- AutoPickle interface for automatic serialization
- Multiple transport mechanisms (callbacks, queues, sockets, etc.)
- Strong separation of concerns and minimal code debt
"""

import asyncio
import logging
import pickle
import time
import weakref
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any, Dict, List, Optional, Union, Callable, Awaitable,
    TypeVar, Protocol, runtime_checkable
)

from pycore.autopickle import AutoPickle

# Type definitions
EventHandler = Union[Callable[[Any], None], Callable[[Any], Awaitable[None]]]
T = TypeVar('T')


class TransportType(Enum):
    """Enumeration of supported transport mechanisms."""
    CALLBACK = "callback"
    QUEUE = "queue"
    PIPE = "pipe"
    SOCKET = "socket"
    LOAD_BALANCER = "load_balancer"


@dataclass
class EventPayload:
    """Standardized event payload for job lifecycle events."""
    event_type: str
    job_id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@runtime_checkable
class EventTransport(Protocol):
    """Protocol defining the interface for event transport mechanisms."""

    async def send(self, payload: EventPayload) -> None:
        """Send an event payload through this transport."""
        ...

    async def close(self) -> None:
        """Close the transport and cleanup resources."""
        ...


class CallbackTransport:
    """Transport that delivers events via direct callback invocation."""

    def __init__(self, callback: EventHandler):
        self.callback = callback
        self._is_coroutine = asyncio.iscoroutinefunction(callback)

    async def send(self, payload: EventPayload) -> None:
        """Send event to callback."""
        try:
            if self._is_coroutine:
                await self.callback(payload)
            else:
                self.callback(payload)
        except Exception as e:
            logging.error(f"Callback transport error: {e}")

    async def close(self) -> None:
        """No cleanup needed for callbacks."""
        pass


class QueueTransport:
    """Transport that delivers events via asyncio Queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def send(self, payload: EventPayload) -> None:
        """Send event to queue."""
        try:
            await self.queue.put(payload)
        except Exception as e:
            logging.error(f"Queue transport error: {e}")

    async def close(self) -> None:
        """No cleanup needed for queues."""
        pass


class SocketTransport:
    """Transport that delivers events via network sockets."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._connected = False

    async def connect(self) -> None:
        """Establish socket connection."""
        if not self._connected:
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self.host, self.port
                )
                self._connected = True
            except Exception as e:
                logging.error(f"Socket connection failed: {e}")
                raise

    async def send(self, payload: EventPayload) -> None:
        """Send event over socket."""
        if not self._connected:
            await self.connect()

        try:
            data = pickle.dumps(payload)
            # Send length prefix followed by data
            length = len(data)
            self._writer.write(length.to_bytes(4, byteorder='big'))
            self._writer.write(data)
            await self._writer.drain()
        except Exception as e:
            logging.error(f"Socket transport error: {e}")
            self._connected = False

    async def close(self) -> None:
        """Close socket connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False


class EventDriven(ABC):
    """
    Asyncio-based event-driven interface.

    Provides event publishing and handling with multiple
    transport mechanisms and strong separation of concerns.
    """

    def __init__(self):
        self._event_handlers: Dict[str, List[EventTransport]] = defaultdict(
            list)
        self._global_handlers: List[EventTransport] = []
        self._transport_registry: Dict[str, EventTransport] = {}
        self._task_registry: weakref.WeakSet = weakref.WeakSet()
        self._closed = False

    def register_handler(
            self,
            event_type: str,
            handler: Union[EventHandler, EventTransport],
            transport_type: TransportType = TransportType.CALLBACK
    ) -> str:
        """
        Register an event handler for specific event type.

        Args:
            event_type: Type of event to handle
            handler: Handler function or transport
            transport_type: Type of transport mechanism

        Returns:
            Handler ID for later removal
        """
        transport = self._create_transport(handler, transport_type)
        handler_id = f"{event_type}_{id(transport)}"

        self._event_handlers[event_type].append(transport)
        self._transport_registry[handler_id] = transport

        return handler_id

    def register_global_handler(
            self,
            handler: Union[EventHandler, EventTransport],
            transport_type: TransportType = TransportType.CALLBACK
    ) -> str:
        """
        Register a global event handler for all events.

        Args:
            handler: Handler function or transport
            transport_type: Type of transport mechanism

        Returns:
            Handler ID for later removal
        """
        transport = self._create_transport(handler, transport_type)
        handler_id = f"global_{id(transport)}"

        self._global_handlers.append(transport)
        self._transport_registry[handler_id] = transport

        return handler_id

    @staticmethod
    def _create_transport(
            handler: Union[EventHandler, EventTransport],
            transport_type: TransportType
    ) -> EventTransport:
        """Create appropriate transport based on type and handler."""
        if isinstance(handler, EventTransport):
            return handler

        if transport_type == TransportType.CALLBACK:
            return CallbackTransport(handler)
        elif transport_type == TransportType.QUEUE:
            if not isinstance(handler, asyncio.Queue):
                raise ValueError("Queue transport requires asyncio.Queue")
            return QueueTransport(handler)
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

    async def publish_event(
            self,
            event_type: str,
            data: Any,
            metadata: Optional[Dict[str, Any]] = None,
            correlation_id: Optional[str] = None
    ) -> None:
        """
        Publish an event to all registered handlers.

        Args:
            event_type: Type of event being published
            data: Event data payload
            metadata: Optional metadata dictionary
            correlation_id: Optional correlation ID for tracking
        """
        if self._closed:
            raise RuntimeError("EventDriven instance is closed")

        payload = EventPayload(

            event_type=event_type,
            data=data,
            metadata=metadata or {},
            job_id=correlation_id
        )

        # Collect all relevant handlers
        handlers = []
        handlers.extend(self._event_handlers.get(event_type, []))
        handlers.extend(self._global_handlers)

        if handlers:
            # Create tasks for concurrent event delivery
            tasks = [self._deliver_event(handler, payload) for handler in
                     handlers]

            # Store tasks in registry for cleanup
            for task in tasks:
                self._task_registry.add(task)

            # Fire and forget - don't wait for completion
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver_event(self, transport: EventTransport,
                             payload: EventPayload) -> None:
        """Deliver event to a specific transport."""
        try:
            await transport.send(payload)
        except Exception as e:
            logging.error(f"Event delivery failed: {e}")
            await self.on_delivery_error(transport, payload, e)

    async def on_delivery_error(
            self,
            transport: EventTransport,
            payload: EventPayload,
            error: Exception
    ) -> None:
        """
        Hook for handling delivery errors. Override in subclasses.

        Args:
            transport: Transport that failed
            payload: Event payload that failed to deliver
            error: Exception that occurred
        """
        pass

    def unregister_handler(self, handler_id: str) -> bool:
        """
        Unregister an event handler.

        Args:
            handler_id: ID returned from register_handler

        Returns:
            True if handler was found and removed
        """
        if handler_id not in self._transport_registry:
            return False

        transport = self._transport_registry.pop(handler_id)

        # Remove from event handlers
        for handlers in self._event_handlers.values():
            if transport in handlers:
                handlers.remove(transport)

        # Remove from global handlers
        if transport in self._global_handlers:
            self._global_handlers.remove(transport)

        # Close transport
        asyncio.create_task(transport.close())

        return True

    async def close(self) -> None:
        """Close all transports and cleanup resources."""
        if self._closed:
            return

        self._closed = True

        # Close all registered transports
        close_tasks = [
            transport.close()
            for transport in self._transport_registry.values()
        ]

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Wait for any pending event delivery tasks
        pending_tasks = [task for task in self._task_registry if
                         not task.done()]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Clear all registries
        self._event_handlers.clear()
        self._global_handlers.clear()
        self._transport_registry.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def event_driven(cls):
    """
    Class decorator to make any class event-driven.

    Adds EventDriven capabilities to existing classes without inheritance.
    """
    if not hasattr(cls, '__init__'):
        raise ValueError("Decorated class must have __init__ method")

    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Initialize EventDriven capabilities
        EventDriven.__init__(self)
        # Call original init
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    # Add EventDriven methods to the class
    for name, method in EventDriven.__dict__.items():
        if not name.startswith('_') and callable(method):
            setattr(cls, name, method)

    # Add async context manager support
    cls.__aenter__ = EventDriven.__aenter__
    cls.__aexit__ = EventDriven.__aexit__

    return cls
