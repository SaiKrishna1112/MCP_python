import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

# Load environment variables
load_dotenv()

app = FastAPI(title="OxyLoans Agent API")

# Configuration
MCP_SERVER_URL = "https://testingmcp-kulj.onrender.com/sse"
# MCP_SERVER_URL = "http://localhost:8001/mcp/sse" # Uncomment for local Docker

# System Instructions (Global Configuration)
SYSTEM_INSTRUCTION = """
SYSTEM INSTRUCTIONS:
1. You are an AI assistant for OxyLoans.
2. If a user tries to perform restricted actions (like viewing cart, adding to cart, checkout) and you do not have their identity (user_id/token):
   - Do NOT ask for "session ID", "user ID", or "token".
   - Instead, politely ask them to login. Say: "Please login to {action}. Please provide your mobile number to login."
3. If the user provides a mobile number, use the 'simple_login' tool immediately.
4. After successful login, you can proceed with their original request (e.g., showing the cart).
"""

# Global session store: Map local_session_id -> { "mcp_client": MCPClient, "history": list }
# In a real app, this should be a persistent database + connection pool
active_sessions: Dict[str, dict] = {}

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None # Helper for client tracking, but we rely on MCP session
    model: str = "gpt-4o"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    mcp_session_id: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a request using a persistent MCP connection.
    The MCP server handles the 'real' session (user login state).
    """
    # 1. Validate Env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    # 2. Get or Create Session
    # If the user sends a session_id, we try to reuse the existing MCP connection.
    # If not, we create a new one.
    
    local_session_id = request.session_id or str(os.urandom(8).hex())
    
    if local_session_id not in active_sessions:
        # --- NEW CONNECTION ---
        print(f"Creating new MCP connection for session {local_session_id}")
        config = {
            "mcpServers": {
                "default": {
                    "url": MCP_SERVER_URL,
                    "auth": {"type": "none"}
                }
            }
        }
        try:
            # Initialize Client
            client = MCPClient.from_dict(config)
            # We need to keep this client ALIVE to maintain the session
            # Note: mcp_use client might auto-connect on first use. 
            # We will store it. check if we can get the session ID?
            
            active_sessions[local_session_id] = {
                "client": client,
                "history": [],
                "agent": None # We will init agent lazily
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to MCP: {str(e)}")

    # 3. Get Session Objects
    session_data = active_sessions[local_session_id]
    client = session_data["client"]
    
    # Initialize Agent if needed (or reuse to keep conversational state?)
    # Re-creating agent is usually safe if client is persistent.
    if not session_data["agent"]:
        llm = ChatOpenAI(model=request.model, temperature=0, api_key=api_key)
        
        # We can pass previous messages if we want memory
        # mcp_use Agent might handle history internally if we keep the instance.
        # Increased max_steps to 30 to avoid recursion limits on complex auth flows
        session_data["agent"] = MCPAgent(llm=llm, client=client, max_steps=40)

    agent = session_data["agent"]

    try:
        # 4. Run Agent
        # We prepend the system instruction to ensure the model follows behavioral rules
        full_query = f"{SYSTEM_INSTRUCTION}\n\nUser Query: {request.query}"
        
        # We pass recursion_limit config if supported, otherwise max_steps in init handles it usually
        result = await agent.run(full_query)
        
        # Store interactions locally
        session_data["history"].append({"role": "user", "content": request.query})
        session_data["history"].append({"role": "assistant", "content": str(result)})

        return ChatResponse(
            response=str(result),
            session_id=local_session_id
        )

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing request: {e}")
        
        # Graceful handling for recursion limit
        if "Recursion limit" in error_msg:
             return ChatResponse(
                response="I'm having trouble processing that request in time. Please try again or provide more details.",
                session_id=local_session_id
            )
            
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run on port 8090 to avoid conflict with standard MCP server 8001
    print(f"🚀 Agent API running on http://localhost:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090)
