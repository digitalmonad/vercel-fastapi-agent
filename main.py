from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agent import run_agent

# import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Request model
class AgentRequest(BaseModel):
    """Request model for agent invocation."""

    prompt: str


# Response model
class AgentResponse(BaseModel):
    """Response model for agent invocation."""

    response: str


@app.get("/")
async def home(request: Request):
    """Serve the main HTML interface."""
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """
    Invoke the AI agent with a prompt.

    The agent can read and write text files based on natural language instructions.
    """

    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Run the agent with the user's prompt
        result = run_agent(request.prompt)

        return AgentResponse(response=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")


# uvicorn.run(app, host="0.0.0.0", port=8000)
# for development only
