import pickle
from typing import Optional, Any, Dict

import zmq

from pycore.settings.msg import OperationType, SettingsRequest, SettingsResponse
from pycore.zmq.actor import ZMQActor


class SettingsClient(ZMQActor):
    """
    Async client for settings server using DEALER socket.
    Supports concurrent requests without blocking.
    """

    def __init__(self,
                 router_endpoint: str = "tcp://127.0.0.1:6000",
                 timeout: float = 5.0):
        super().__init__("SettingsClient")

        self.router_endpoint = router_endpoint
        self.timeout = timeout
        self.dealer = None
        self.last_sequence = -1

    async def start(self):
        """Initialize client."""
        await super().start()

        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        self.dealer.connect(self.router_endpoint)

        self.logger.info(f"Connected to {self.router_endpoint}")

    async def get(self, field_path: Optional[str] = None) -> Any:
        """Get settings value."""
        request = SettingsRequest(
            operation=OperationType.GET,
            field_path=field_path
        )

        response = await self._send_request(request)
        if response.success:
            self.last_sequence = response.sequence
            return response.data
        else:
            raise RuntimeError(f"Get failed: {response.error}")

    async def set(self, field_path: str, value: Any,
                  changed_by: str = None, reason: str = None) -> bool:
        """Set settings value with audit metadata."""
        if changed_by is None:
            changed_by = f"client-{id(self)}"

        request = SettingsRequest(
            operation=OperationType.SET,
            field_path=field_path,
            value=value,
            metadata={
                'changed_by': changed_by,
                'reason': reason
            }
        )

        response = await self._send_request(request)
        if response.success:
            self.last_sequence = response.sequence
            return True
        else:
            raise RuntimeError(f"Set failed: {response.error}")

    async def get_snapshot(self) -> Dict[str, Any]:
        """Get complete snapshot (Clone pattern)."""
        request = SettingsRequest(operation=OperationType.SNAPSHOT)
        response = await self._send_request(request)

        if response.success:
            return response.data
        else:
            raise RuntimeError(f"Snapshot failed: {response.error}")

    async def _send_request(self, request: SettingsRequest) -> SettingsResponse:
        """Send request and wait for response."""
        req_data = pickle.dumps(request)
        await self.dealer.send(req_data)

        # DEALER receives from ROUTER: [Data] (single frame)
        resp_data = await self.dealer.recv()
        response: SettingsResponse = pickle.loads(resp_data)
        return response
