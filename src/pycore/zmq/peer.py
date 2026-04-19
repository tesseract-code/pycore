import asyncio
import time
from enum import Enum

import zmq

from pycore.zmq_utils.actor import ZMQActor


class PeerState(Enum):
    PRIMARY = 1
    BACKUP = 2
    PASSIVE = 3
    ACTIVE = 4


class BinaryStarPeer(ZMQActor):
    """
    Implements a simplified Binary Star Pattern.
    Two servers (Primary/Backup) heartbeat each other.
    If Primary vanishes, Backup takes over.
    """

    def __init__(self, name: str, local_pub: str, peer_sub: str,
                 is_primary=False):
        super().__init__(name)
        self.state = PeerState.PRIMARY if is_primary else PeerState.BACKUP
        self.local_pub_addr = local_pub
        self.peer_sub_addr = peer_sub

        # Sockets
        self.pub = self.ctx.socket(zmq.PUB)
        self.sub = self.ctx.socket(zmq.SUB)

        # Heartbeat config
        self.heartbeat_at = time.time()
        self.peer_expiry = time.time() + 2.0
        self.interval = 0.5  # Send HB every 500ms

    async def start(self):
        await super().start()
        # Bind our HB publisher
        self.pub.bind(self.local_pub_addr)
        # Subscribe to peer's HB
        self.sub.connect(self.peer_sub_addr)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"HB")

        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._listen_loop()))

    async def _heartbeat_loop(self):
        while self._running:
            # Send Heartbeat
            await self.pub.send_multipart(
                [b"HB", str(self.state.value).encode()])
            await asyncio.sleep(self.interval)

    async def _listen_loop(self):
        while self._running:
            try:
                # Check for peer heartbeats
                try:
                    msg = await asyncio.wait_for(self.sub.recv_multipart(),
                                                 timeout=0.1)
                    # We got a heartbeat, update expiry
                    self.peer_expiry = time.time() + 2.0
                except asyncio.TimeoutError:
                    pass  # Check logic below

                # Failover Logic
                if time.time() > self.peer_expiry:
                    if self.state == PeerState.BACKUP:
                        self.logger.warning("Peer died! Promoting to ACTIVE.")
                        self.state = PeerState.ACTIVE
                else:
                    # Peer came back? (Simplified logic)
                    if self.state == PeerState.ACTIVE:
                        self.logger.info("Peer returned. Demoting to BACKUP.")
                        self.state = PeerState.BACKUP

            except asyncio.CancelledError:
                break
