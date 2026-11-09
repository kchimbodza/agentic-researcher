"""Synchronous MCP client for the agent: starts app.mcp_server over stdio once and reuses it.

The MCP SDK is async; the agent graph is synchronous. A background thread owns one event loop
and one ClientSession; call() submits a tool call to it and waits for the result.
If MCP can't start, ToolBox falls back to calling app.tools directly (and says so).
"""

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import Future

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import docs, tools


class MCPTools:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._ready: Future = Future()
        self._stop: asyncio.Event | None = None
        self.session: ClientSession | None = None
        threading.Thread(target=self._run, daemon=True, name="mcp-client").start()
        self._ready.result(timeout=60)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"],
                                       cwd=str(docs.ROOT), env={**os.environ})
        try:
            async with stdio_client(params, errlog=open(os.devnull, "w")) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session, self._stop = session, asyncio.Event()
                    self.names = [t.name for t in (await session.list_tools()).tools]
                    self._ready.set_result(True)
                    await self._stop.wait()
        except Exception as e:  # report start-up failures to the waiting constructor
            if not self._ready.done():
                self._ready.set_exception(e)

    def call(self, name: str, **args):
        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(name, args), self._loop)
        result = fut.result(timeout=120)
        if getattr(result, "is_error", False) or getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool {name} failed: {_text(result)}")
        sc = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
        if isinstance(sc, dict) and set(sc) == {"result"}:
            return sc["result"]
        if sc is not None:
            return sc
        return json.loads(_text(result))

    def close(self):
        if self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


class ToolBox:
    """What the agent uses. transport='mcp' (default) or 'direct'."""

    _shared: MCPTools | None = None

    def __init__(self, transport: str = "mcp"):
        self.transport, self.note = transport, ""
        if transport == "mcp":
            try:
                if ToolBox._shared is None:
                    ToolBox._shared = MCPTools()
            except Exception as e:
                self.transport, self.note = "direct", f"MCP unavailable ({type(e).__name__}); using tools in-process"

    def __call__(self, name: str, **args):
        if self.transport == "mcp":
            return ToolBox._shared.call(name, **args)
        return getattr(tools, name)(**args)
