"""Client for the provider mocks' control routes (/__requests, /__reset)."""

import asyncio
from typing import Callable, Optional

import httpx

from .config import MOCK_HOST, MOCK_PORTS


class Mock:
    def __init__(self, name: str):
        self.name = name
        self.url = f"http://{MOCK_HOST}:{MOCK_PORTS[name]}"
        self.http = httpx.AsyncClient(base_url=self.url, timeout=30)

    async def wait_ready(self, timeout: float = 30) -> bool:
        for _ in range(int(timeout * 2)):
            try:
                if (await self.http.get("/__requests")).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        return False

    async def reset(self) -> None:
        await self.http.post("/__reset")

    async def requests(self, match: Optional[Callable[[dict], bool]] = None) -> list:
        entries = (await self.http.get("/__requests")).json()
        return [e for e in entries if match is None or match(e)]

    async def last(self, match: Optional[Callable[[dict], bool]] = None) -> dict:
        entries = await self.requests(match)
        return entries[-1] if entries else {}

    async def close(self) -> None:
        await self.http.aclose()
