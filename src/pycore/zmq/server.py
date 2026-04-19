import asyncio
import random

import zmq
import zmq.asyncio

from pycore.zmq_utils.actor import ZMQActor


class AsyncServer(ZMQActor):
    """
    Uses a ROUTER socket to handle requests asynchronously.
    Unlike REP, ROUTER gives us the client Identity, allowing us to
    process multiple requests in parallel and reply out-of-order.
    """

    def __init__(self, endpoint: str):
        super().__init__("AsyncServer")
        self.endpoint = endpoint
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)

    async def start(self):
        await super().start()
        self.socket.bind(self.endpoint)
        self._tasks.append(asyncio.create_task(self._recv_loop()))

    async def _recv_loop(self):
        while self._running:
            try:
                # ROUTER frame structure: [Identity, Empty, Data]
                msg = await self.socket.recv_multipart()
                identity, _, data = msg[0], msg[1], msg[2]

                # Offload processing to a task (simulating work) without blocking recv
                asyncio.create_task(self._handle_request(identity, data))
            except asyncio.CancelledError:
                break

    async def _handle_request(self, identity, data):
        # Simulate processing time
        request = data.decode()
        await asyncio.sleep(random.random() * 0.5)

        response = f"Echo: {request}"
        if self._running:
            # Send back to specific identity
            await self.socket.send_multipart([identity, b"", response.encode()])
