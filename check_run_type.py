from mcp_use import MCPAgent
import inspect
from typing import get_type_hints

sig = inspect.signature(MCPAgent.run)
print(f"Query annotation: {sig.parameters['query'].annotation}")
