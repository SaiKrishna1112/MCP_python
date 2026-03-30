import os
import asyncio
import httpx
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from mcp_use import MCPAgent, MCPClient

# Load environment variables
load_dotenv()

async def scheduled_task():
    """
    Background task that runs every 13 minutes.
    """
    while True:
        try:
            print("Running scheduled task (13 mins cycle)...")
            # Example: Keep the MCP server alive
            # async with httpx.AsyncClient() as client:
            #     # This uses the global MCP_SERVER_URL defined below
            #     response = await client.get(MCP_SERVER_URL)
            #     print(f"Ping response: {response.status_code}")
        except Exception as e:
            print(f"Error in scheduled task: {e}")
        
        # Wait for 13 minutes (13 * 60 seconds)
        await asyncio.sleep(13 * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start the scheduled task
    task = asyncio.create_task(scheduled_task())
    yield
    # Shutdown: cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Askoxy.AI Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://testingmcp-kulj.onrender.com/sse")
# MCP_SERVER_URL = "http://localhost:8001/mcp/sse" # Uncomment for local Docker

# System Instructions (Global Configuration)
# SYSTEM_INSTRUCTION = """
# SYSTEM INSTRUCTIONS:
# 1. You are an AI assistant for Askoxy.AI.
# 2. If a user tries to perform restricted actions (like viewing cart, adding to cart, checkout) and you do not have their identity (user_id/token):
#    - Do NOT ask for "session ID", "user ID", or "token".
#    - Instead, politely ask them to login. Say: "Please login to {action}. Please provide your mobile number to login."
# 3. If the user provides a mobile number, use the 'send_login_otp' tool immediately.
# 4. After successful login, you can proceed with their original request (e.g., showing the cart).
# """

SYSTEM_INSTRUCTION = """
SYSTEM INSTRUCTIONS:

1. You are an AI assistant strictly for Askoxy.AI.
   - You MUST only respond to queries related to Askoxy.AI (products, orders, cart, delivery, login, offers, services).
   - If the user asks anything unrelated (general knowledge, coding, news, etc.), respond:
     "I can only assist with Askoxy.AI related queries."

2. Authentication Handling:
   - If a user tries to perform restricted actions such as:
     • View cart
     • Add to cart
     • Checkout
     • View orders
     • Manage profile
   AND you do NOT have user authentication (user_id/token):

     → DO NOT ask for session ID, token, or user ID.
     → Respond ONLY with:
       "Please login to continue. Please provide your mobile number to login."

3. Login Flow:
   - If the user provides a valid mobile number:
     → Immediately call the 'send_login_otp' tool.
     → Do NOT ask any additional questions before calling the tool.

4. Post-login Behavior:
   - Once login is successful:
     → Resume and complete the user’s original request automatically.
     → Do not ask the user to repeat the request.

5. Response Rules:
   - Keep responses concise and action-oriented.
   - Do not provide unnecessary explanations.
   - Always prioritize completing user actions over conversation.

6. Strict Boundaries:
   - Do NOT answer:
     • General knowledge questions
     • Programming or technical questions
     • Personal advice
     • Any non-Askoxy.AI topics

   - If such a request is made, respond:
     "I can only assist with Askoxy.AI related services."

"""

# Global session store: Map local_session_id -> { "client": MCPClient, "agent": MCPAgent, "history": list }
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

@app.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up a specific session"""
    if session_id in active_sessions:
        try:
            # Close client connection if possible
            client = active_sessions[session_id].get("client")
            if client and hasattr(client, "close"):
                await client.close()
        except Exception as e:
            print(f"Error closing client: {e}")
        
        del active_sessions[session_id]
        return {"message": f"Session {session_id} cleaned up"}
    return {"message": "Session not found"}

@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    return {"active_sessions": list(active_sessions.keys()), "count": len(active_sessions)}

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
            # Initialize Client and Agent once per session
            client = MCPClient.from_dict(config)
            llm = ChatOpenAI(model=request.model, temperature=0, api_key=api_key)
            agent = MCPAgent(llm=llm, client=client, max_steps=40)
            
            active_sessions[local_session_id] = {
                "client": client,
                "agent": agent,
                "history": []
            }
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"MCP module not found. Install with: pip install mcp-use")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to MCP: {str(e)}")

    # 3. Get Session Objects - Reuse existing agent to maintain conversation state
    session_data = active_sessions[local_session_id]
    agent = session_data["agent"]

    # Agent maintains its own conversation state, no need to restore history manually

    try:
        # 4. Run Agent
        # For first message in session, include system instruction
        if not session_data["history"]:
            full_query = f"{SYSTEM_INSTRUCTION}\n\nUser Query: {request.query}"
        else:
            full_query = request.query
        
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
        
        # Reset agent on error to prevent error caching
        try:
            client = session_data["client"]
            llm = ChatOpenAI(model=request.model, temperature=0, api_key=api_key)
            session_data["agent"] = MCPAgent(llm=llm, client=client, max_steps=40)
            print(f"Agent reset for session {local_session_id} due to error")
        except Exception as reset_error:
            print(f"Failed to reset agent: {reset_error}")
        
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