from mcp_use import MCPAgent
import inspect

print("Signature:")
print(inspect.signature(MCPAgent.__init__))
print("\nDocstring:")
print(MCPAgent.__init__.__doc__)
