import asyncio
from mcp_use import MCPAgent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", api_key=api_key)
    # Mock client just to satisfy init if needed, or None
    # MCPAgent needs client or connectors.
    # We can try with None and see if it inits (it might fail without client).
    # But usually we can inspect structure without running.
    
    agent = MCPAgent(llm=llm, connectors=[], memory_enabled=True)
    
    print("Agent attributes:", dir(agent))
    if hasattr(agent, "memory"):
        print("Memory type:", type(agent.memory))
        print("Memory attributes:", dir(agent.memory))
    
    if hasattr(agent, "chat_history"):
        print("Chat History:", agent.chat_history)

if __name__ == "__main__":
    asyncio.run(main())
