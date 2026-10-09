"""
title: E2E Workspace Tool
author: owndev
version: 0.2.0
license: Apache License 2.0
description: Test-only workspace tool (Python) for the native tool calling scenarios of tests/e2e (suites/_gemini_tools.py). It reports what Open WebUI passed to it.
"""

import base64
import json
import struct
import zlib
from typing import Optional

from pydantic import BaseModel, Field


def _png_base64() -> str:
    """Base64 of a 1x1 red PNG."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


class Tools:
    class Valves(BaseModel):
        offset: int = Field(default=0, description="added to every sum")

    def __init__(self):
        self.valves = self.Valves()

    def add_numbers(self, a: int, b: int, note: str = "") -> str:
        """
        Add two integers and report the Python types that arrived.
        :param a: first addend
        :param b: second addend
        :param note: optional free text
        """
        return json.dumps(
            {
                "sum": a + b + self.valves.offset,
                "types": [type(a).__name__, type(b).__name__, type(note).__name__],
                "note": note,
            }
        )

    async def whoami(
        self,
        label: str,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> dict:
        """
        Echo a label plus the user, the chat and whether an event emitter exists.
        :param label: any text
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"whoami {label}", "done": True},
                }
            )
        return {
            "label": label,
            "user_email": (__user__ or {}).get("email"),
            "chat_id": (__metadata__ or {}).get("chat_id"),
            "has_event_emitter": __event_emitter__ is not None,
        }

    def make_image(self, color: str = "red") -> str:
        """
        Draw a tiny image. Open WebUI passes the images of tool results to the
        model in a user message after the tool results.
        :param color: any text
        """
        return "data:image/png;base64," + _png_base64()
