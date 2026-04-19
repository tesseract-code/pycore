import asyncio
import pickle
import random
from typing import Dict, Any

import zmq

from pycore.zmq_utils.actor import ZMQActor


class CloneServer(ZMQActor):
    """
    Maintains shared state (KV Store).
    1. Updates are broadcast via PUB (updates).
    2. Snapshots are sent via ROUTER when requested (state recovery).
    """

    def __init__(self, pub_endpoint: str, snap_endpoint: str):
        super().__init__("CloneServer")
        self.kv_store: Dict[str, Any] = {}
        self.sequence = 0

        # Publisher for updates
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(pub_endpoint)
        self.snapshot = self.ctx.socket(zmq.ROUTER)
        self.snapshot.bind(snap_endpoint)

    async def start(self):
        await super().start()
        self._tasks.append(asyncio.create_task(self._snapshot_loop()))
        self._tasks.append(asyncio.create_task(self._update_generator()))

    async def _update_generator(self):
        """Simulates internal state changes."""
        while self._running:
            await asyncio.sleep(1)
            key = random.choice(["status", "load", "users"])
            value = random.randint(1, 100)

            # Update internal state
            self.sequence += 1
            self.kv_store[key] = value

            # Publish update: [Topic, Sequence, Key, Value]
            update = {"seq": self.sequence, "key": key, "val": value}
            self.logger.debug(f"Publishing update: {update}")
            await self.pub.send_multipart([b"KV", pickle.dumps(update)])

    async def _snapshot_loop(self):
        """Responds to Snapshot requests (LVC)."""
        while self._running:
            try:
                # Wait for request: [Identity, empty, "ICANHAZ?"]
                msg = await self.snapshot.recv_multipart()
                identity = msg[0]

                self.logger.info(f"Sending Snapshot to {identity}")

                # Send state
                payload = pickle.dumps({
                    "seq": self.sequence,
                    "store": self.kv_store
                })
                await self.snapshot.send_multipart([identity, b"", payload])

            except asyncio.CancelledError:
                break


class CloneClient(ZMQActor):
    """
    Subscribes to CloneServer.
    1. Connects SUB.
    2. Request Snapshot via DEALER.
    3. Merges Snapshot.
    4. Applies buffered updates (catching up).
    5. Checks sequence numbers (Suicidal Snail logic).
    """

    def __init__(self, pub_endpoint: str, snap_endpoint: str):
        super().__init__("CloneClient")
        self.kv_store = {}
        self.last_sequence = -1

        # Use the helper
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(pub_endpoint)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"KV")
        self.sub.setsockopt(zmq.RCVHWM, 100)

        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.connect(snap_endpoint)

    async def start(self):
        await super().start()
        await self._sync_state()
        self._tasks.append(asyncio.create_task(self._watch_updates()))

    async def _sync_state(self):
        """Handshake to get initial state (Clone Pattern)."""
        self.logger.info("Requesting snapshot...")
        # Send simple request
        await self.dealer.send(b"ICANHAZ?")

        # FIX: Use recv_multipart to handle the envelope [Empty, Payload]
        # The ROUTER sends: [Identity, b"", Payload]
        # The DEALER receives: [b"", Payload]
        frames = await self.dealer.recv_multipart()

        if len(frames) < 2:
            # Handle unexpected frame format
            self.logger.error(f"Unexpected snapshot format: {frames}")
            return

        # The payload is the last frame
        payload = frames[-1]

        try:
            data = pickle.loads(payload)
            self.kv_store = data["store"]
            self.last_sequence = data["seq"]
            self.logger.info(
                f"Snapshot received. Seq: {self.last_sequence} Data: {self.kv_store}")
        except Exception as e:
            self.logger.error(f"Failed to decode snapshot: {e}")

    async def _watch_updates(self):
        while self._running:
            try:
                msg = await self.sub.recv_multipart()
                update = pickle.loads(msg[1])
                seq = update["seq"]

                # SUICIDAL SNAIL LOGIC
                # If the incoming sequence is essentially greater than last + 1,
                # we missed messages due to Slow Subscriber (HWM dropped them).
                if seq > self.last_sequence + 1:
                    self.logger.critical(
                        f"SNAIL ALERT: Expected seq {self.last_sequence + 1}, got {seq}. We are too slow!")
                    self.logger.error("Committing suicide (Stopping Client)...")
                    self._running = False  # Stop loop
                    return

                # Normal update
                self.kv_store[update["key"]] = update["val"]
                self.last_sequence = seq
                self.logger.info(f"Updated: {self.kv_store} (Seq: {seq})")

            except asyncio.CancelledError:
                break
