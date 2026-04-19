import asyncio
import logging

import zmq


class ZMQActor:
    """
    Base Actor using asyncio as the Reactor.
    Handles context creation and graceful shutdown.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.ctx = zmq.asyncio.Context()
        self.loop = asyncio.get_running_loop()
        self._running = False
        self._tasks = []

    async def start(self):
        self._running = True
        self.logger.info("Starting...")

    async def stop(self):
        """
        Graceful shutdown that handles:
        1. Task cancellation
        2. Socket cleanup (even untracked ones)
        3. Idempotency (safe to call multiple times)
        """
        if not self.ctx or self.ctx.closed:
            return

        self._running = False
        self.logger.info("Stopping...")

        # 1. Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # 2. Wait for tasks to finish cancelling
        # return_exceptions=True prevents crash if a task raises CancelledError
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # 3. DESTROY context
        # destroy(linger=0) is the magic bullet:
        # - It forces all attached sockets to close immediately (linger=0)
        # - It terminates the context
        # - It prevents the "hanging on term()" issue
        if not self.ctx.closed:
            self.ctx.destroy(linger=0)

        self.logger.info("Stopped.")
