import os
import uvicorn
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyCustomTools")

@mcp.tool()
def calculate_uptime(days: int) -> str:
    """Calculates a fake uptime percentage based on days."""
    return f"Uptime for the last {days} days: 99.9%"

app = mcp.as_asgi()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
