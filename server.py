"""
server.py
---------
A tiny FastAPI service that exposes our Python tools as HTTP endpoints.
The agent calls these endpoints to execute tools - the LLM itself never
runs code; it only *describes* the call it wants to make.

Run it with uvicorn (the ASGI server that hosts FastAPI):

    uv run uvicorn server:app --reload --port 8000

Then visit:
    http://localhost:8000/docs        <- auto-generated Swagger UI
    http://localhost:8000/tools       <- our tool catalog
    http://localhost:8000/healthz     <- liveness probe
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from tools import TOOL_REGISTRY, TOOL_SCHEMAS, dispatch_tool


app = FastAPI(title="Agent Tool Server", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> dict[str, Any]:
    """Return the list of registered tools and their OpenAI-style schemas."""
    return {
        "names": list(TOOL_REGISTRY.keys()),
        "schemas": TOOL_SCHEMAS,
    }


@app.post("/tools/{name}")
def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with a JSON body of arguments.

    TODO (live):
      1. Try `dispatch_tool(name, arguments)`. Map failures to HTTP errors:
           - KeyError              -> 404 "unknown tool"
           - NotImplementedError   -> 501 "not implemented yet"
           - TypeError             -> 400 "bad arguments"
           - any other Exception   -> 500 with the error string
      2. On success, return {"name": name, "result": <whatever the tool returned>}.
    """
    try:
        result = dispatch_tool(name, arguments)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"name": name, "result": result}

