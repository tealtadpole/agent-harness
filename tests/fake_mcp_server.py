"""A stand-in for confluence-rag: a real stdio MCP server with a canned search tool."""

import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("fake-confluence")


@server.tool()
def search_confluence(query: str, max_results: int = 6) -> str:
    """Search Confluence pages by meaning and keywords."""
    secret = os.environ.get("FAKE_RAG_SECRET", "<unset>")
    return (f"[1] Vacation policy\nURL: https://wiki.test/pages/1\n<passage>Employees get 25 days "
            f"of paid vacation.</passage>\nquery={query} secret={secret}")


if __name__ == "__main__":
    server.run("stdio")
