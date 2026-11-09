"""Starts the real MCP server over stdio and calls its tools."""
from app.mcp_client import MCPTools


def test_mcp_server_tools():
    m = MCPTools()
    try:
        assert set(m.names) == {"search_documents", "read_page", "check_quote", "list_library"}
        assert m.call("check_quote", doc="datasheets/Demo_Router.pdf", page=1, quote="POE Budget 40W")["verified"]
        assert m.call("read_page", doc="datasheets/Demo_Router.pdf", page=1)["text"].startswith("Demo router")
        assert len(m.call("list_library")) == 3
    finally:
        m.close()
