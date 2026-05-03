"""
Centralized Settings Management System (Enhanced with Advanced ZMQ Patterns)
=============================================================================
Enterprise-grade configuration server using async ZMQ patterns.

Features:
- Async ROUTER/DEALER architecture (non-blocking concurrent requests)
- Proper graceful shutdown with context.destroy(linger=0)
- Clone pattern for state synchronization
- Thread and process safe
- Schema validation and audit trail
- High availability ready

Based on: ZMQ Guide patterns (Clone, Binary Star, LVC, Suicidal Snail)
"""

import asyncio
import logging
import pickle
from datetime import datetime
from typing import Any

import zmq
import zmq.asyncio

from pycore.settings.accessor import SettingsAccessor
from pycore.settings.history import SettingsHistory, ChangeMetadata, \
    ChangeNotification
from pycore.settings.msg import OperationType, SettingsRequest, \
    SettingsResponse
from pycore.settings.validator import SettingsValidator
from pycore.zmq.actor import ZMQActor

logger = logging.getLogger(__name__)


class SettingsServer(ZMQActor):
    """
    Async settings server using:
    - ROUTER for concurrent request handling
    - PUB for change notifications (Clone pattern)
    - Sequence numbers for consistency checking
    """

    def __init__(self,
                 initial_settings: Any,
                 router_endpoint: str = "tcp://127.0.0.1:6000",
                 pub_endpoint: str = "tcp://127.0.0.1:6001"):
        super().__init__("SettingsServer")

        self.settings = initial_settings
        self.router_endpoint = router_endpoint
        self.pub_endpoint = pub_endpoint

        # Components
        self.validator = SettingsValidator()
        self.history = SettingsHistory()
        self.accessor = SettingsAccessor()

        # Sequence number for Clone pattern
        self.sequence = 0

        # Sockets (created in start())
        self.router = None
        self.pub = None

    async def start(self):
        """Initialize and start server."""
        await super().start()

        # Create ROUTER socket for requests
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind(self.router_endpoint)

        # Create PUB socket for notifications
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(self.pub_endpoint)

        self.logger.info(
            f"Bound ROUTER={self.router_endpoint}, PUB={self.pub_endpoint}")

        # Start request handler
        self._tasks.append(asyncio.create_task(self._request_loop()))

    async def _request_loop(self):
        """Main loop handling requests asynchronously."""
        while self._running:
            try:
                # ROUTER format when receiving from DEALER: [Identity, Data]
                # (DEALER doesn't send empty delimiter like REQ does)
                msg = await self.router.recv_multipart()

                if len(msg) == 2:
                    # DEALER format: [Identity, Data]
                    identity, data = msg[0], msg[1]
                elif len(msg) == 3:
                    # REQ format: [Identity, Empty, Data]
                    identity, _, data = msg[0], msg[1], msg[2]
                else:
                    self.logger.error(
                        f"Unexpected message format: {len(msg)} frames")
                    continue

                # Handle request in background (non-blocking)
                asyncio.create_task(self._handle_request(identity, data))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Request loop error: {e}", exc_info=True)

    async def _handle_request(self, identity: bytes, data: bytes):
        """Process a single request."""
        try:
            request: SettingsRequest = pickle.loads(data)

            # Dispatch based on operation
            if request.operation == OperationType.GET:
                response = await self._handle_get(request)
            elif request.operation == OperationType.SET:
                response = await self._handle_set(request)
            elif request.operation == OperationType.VALIDATE:
                response = await self._handle_validate(request)
            elif request.operation == OperationType.SNAPSHOT:
                response = await self._handle_snapshot(request)
            else:
                response = SettingsResponse(
                    success=False,
                    error=f"Unknown operation: {request.operation}"
                )

            # Send response back to client
            # DEALER expects: [Data] (no empty frame)
            resp_data = pickle.dumps(response)
            await self.router.send_multipart([identity, resp_data])

        except Exception as e:
            self.logger.error(f"Request handling error: {e}", exc_info=True)
            try:
                err_resp = SettingsResponse(success=False, error=str(e))
                await self.router.send_multipart(
                    [identity, pickle.dumps(err_resp)])
            except Exception:
                self.logger.error("Could not send error response")

    async def _handle_get(self, request: SettingsRequest) -> SettingsResponse:
        """Handle GET operation."""
        try:
            if request.field_path:
                data = self.accessor.get_nested(self.settings,
                                                request.field_path)
            else:
                data = self.settings

            return SettingsResponse(
                success=True,
                data=data,
                request_id=request.request_id,
                sequence=self.sequence
            )
        except Exception as e:
            return SettingsResponse(
                success=False,
                error=str(e),
                request_id=request.request_id
            )

    async def _handle_set(self, request: SettingsRequest) -> SettingsResponse:
        """Handle SET operation with validation and notification."""
        if not request.field_path:
            return SettingsResponse(success=False, error="field_path required")

        try:
            old_value = self.accessor.get_nested(self.settings,
                                                 request.field_path)

            # Validate
            is_valid, error = self.validator.validate(request.field_path,
                                                      request.value)
            if not is_valid:
                return SettingsResponse(success=False, error=error)

            # Update settings
            self.settings = self.accessor.set_nested(
                self.settings, request.field_path, request.value
            )

            # Increment sequence
            self.sequence += 1

            # Create metadata
            metadata = ChangeMetadata(
                timestamp=datetime.now(),
                operation=OperationType.SET,
                field_path=request.field_path,
                old_value=old_value,
                new_value=request.value,
                changed_by=request.metadata.get('changed_by', 'unknown'),
                reason=request.metadata.get('reason'),
                sequence=self.sequence
            )

            # Record in history
            await self.history.add(metadata)

            # Publish notification
            await self._publish_change(metadata)

            return SettingsResponse(
                success=True,
                data=request.value,
                request_id=request.request_id,
                sequence=self.sequence
            )

        except Exception as e:
            self.logger.error(f"SET error: {e}", exc_info=True)
            return SettingsResponse(success=False, error=str(e))

    async def _handle_validate(self,
                               request: SettingsRequest) -> SettingsResponse:
        """Handle VALIDATE operation."""
        try:
            is_valid = self.accessor.validate_path(self.settings,
                                                   request.field_path)
            return SettingsResponse(success=True, data=is_valid)
        except Exception as e:
            return SettingsResponse(success=False, error=str(e))

    async def _handle_snapshot(self,
                               request: SettingsRequest) -> SettingsResponse:
        """Handle SNAPSHOT operation (Clone pattern)."""
        try:
            snapshot = {
                'settings': self.settings,
                'sequence': self.sequence
            }
            return SettingsResponse(success=True, data=snapshot)
        except Exception as e:
            return SettingsResponse(success=False, error=str(e))

    async def _publish_change(self, metadata: ChangeMetadata):
        """Publish change notification (Clone pattern)."""
        try:
            notification = ChangeNotification(
                metadata=metadata,
                current_settings=self.settings
            )
            topic = f"settings.{metadata.field_path}"
            data = pickle.dumps(notification)
            await self.pub.send_multipart([topic.encode(), data])
            self.logger.debug(f"Published: {topic} (seq={metadata.sequence})")
        except Exception as e:
            self.logger.error(f"Publish error: {e}")
