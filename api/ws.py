"""WebSocket fan-out shared by live events and notifications.

Worker threads (live pipeline, escalation) call `publish_*`; messages are
handed to the asyncio loop with `call_soon_threadsafe`, so inference never
blocks on a slow client. Each connection has a bounded queue; when it is full
the oldest message is dropped (live overlays only need the latest state).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class Client:
    cid: int
    ws: WebSocket
    user_id: int
    role: str
    channels: set
    queue: asyncio.Queue = field(repr=False, default=None)


class Hub:
    def __init__(self, queue_size: int):
        self.queue_size = queue_size
        self._clients: Dict[int, Client] = {}
        self._ids = itertools.count(1)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def register(self, ws: WebSocket, user_id: int, role: str, channels: set) -> Client:
        c = Client(next(self._ids), ws, user_id, role, set(channels), asyncio.Queue(maxsize=self.queue_size))
        self._clients[c.cid] = c
        return c

    def unregister(self, client: Client) -> None:
        self._clients.pop(client.cid, None)

    @property
    def connected(self) -> int:
        return len(self._clients)

    def connected_users(self) -> set:
        return {c.user_id for c in self._clients.values()}

    def _put(self, client: Client, message: Dict[str, Any]) -> None:
        if client.queue.full():
            try:
                client.queue.get_nowait()
            except asyncio.QueueEmpty:
                logger.debug("queue drained concurrently for client %d", client.cid)
        client.queue.put_nowait(message)

    def _dispatch(self, channel: str, message: Dict[str, Any], user_id: Optional[int]) -> None:
        for c in list(self._clients.values()):
            if channel in c.channels and (user_id is None or c.user_id == user_id):
                self._put(c, message)

    def publish(self, channel: str, message: Dict[str, Any], user_id: Optional[int] = None) -> None:
        if self.loop is None or self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self._dispatch, channel, message, user_id)

    def publish_to_user(self, user_id: int, message: Dict[str, Any]) -> None:
        """escalation.Publisher signature."""
        self.publish("notifications", message, user_id=user_id)

    async def pump(self, client: Client) -> None:
        while True:
            msg = await client.queue.get()
            await client.ws.send_json(msg)
