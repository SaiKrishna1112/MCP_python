from mcp.server.fastmcp import FastMCP

import uvicorn
# Initialize the server
mcp = FastMCP("MyCustomTools")

# Add a custom tool
@mcp.tool()
def calculate_uptime(days: int) -> str:
    """Calculates a fake uptime percentage based on days."""
    return f"Uptime for the last {days} days: 99.9%"

if __name__ == "__main__":
    # This creates a standard web app from your MCP server
    app = mcp.as_asgi()
    uvicorn.run(app, host="0.0.0.0", port=8000)
