"""
agent.py
--------
The agent loop. This is the heart of the interview problem.

Flow:
  1. Send the user's message + tool schemas to the LLM.
  2. If the LLM responds with `tool_calls`, execute each one (by POST-ing
     to our FastAPI tool server) and append the results back to the
     conversation as `role="tool"` messages.
  3. Repeat until the LLM responds with plain content (no tool calls).
  4. Return the final assistant message.

You'll live-code most of this. The class wiring, env loading, OpenAI
client construction, and the HTTP helper are already done so you can
focus on the loop itself.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SCHEMAS


load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SERVER_URL = os.getenv("AGENT_SERVER_URL", "http://localhost:8000")
DEFAULT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
DEFAULT_STREAM = os.getenv("AGENT_STREAM", "false").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# HTTP client for our own tool server
# ---------------------------------------------------------------------------

class ToolServerClient:
    """Thin wrapper around httpx for talking to our FastAPI tool server."""

    def __init__(self, base_url: str = DEFAULT_SERVER_URL) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=30.0)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """POST /tools/{name} with `arguments` as JSON; raise on non-2xx."""
        resp = self._client.post(f"/tools/{name}", json=arguments)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class Agent:
    """A minimal tool-calling agent built directly on the OpenAI SDK."""

    def __init__(
        self,
        client: OpenAI | None = None,
        tool_client: ToolServerClient | None = None,
        model: str = DEFAULT_MODEL,
        system_prompt: str = (            
            "You are a tool-using agent driving a small finite state machine.\n"
            "\n"
            "STATE       = the running conversation (these messages).\n"
            "ACTIONS     = the tools listed below. Each one is a function in\n"
            "              the agent's *reducer table* that produces new\n"
            "              context as structured JSON.\n"
            "TRANSITIONS = the tool's return value is appended back to STATE,\n"
            "              then you decide the next action (or finish).\n"
            "\n"

            "Available actions:\n"
            "  - get_nanotime()                       -> {nanotime: int}\n"
            "  - day_of_week_from_nanotime(nanotime)  -> {day_of_week: str}\n"
            "  - translate(text, target_language)     -> {translated: str} | {error: ...}\n"
            "\n"
            "Plan: decompose the user's request into the SMALLEST sequence of\n"
            "tool calls that solves it, chain them, then return a concise\n"
            "final answer in plain text. Prefer calling tools over guessing,\n"
            "especially for dates, times, and translations. If a tool returns\n"
            "an error, read it and either retry with different arguments or\n"
            "explain the limitation honestly."
        ),
        max_steps: int = DEFAULT_MAX_STEPS,
        stream: bool = DEFAULT_STREAM,
    ) -> None:
        self.client = client or OpenAI()
        self.tools = tool_client or ToolServerClient()
        self.model = model
        self.max_steps = max_steps
        self.stream = stream
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

    # ----- Public API ------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a user message and return the assistant's final reply.

        TODO (live) - this is the agent loop:
          1. Append {"role": "user", "content": user_message} to self.messages.
          2. Loop up to self.max_steps:
             a. assistant_msg = self._call_model()
             b. self.messages.append(assistant_msg)
             c. If assistant_msg has no "tool_calls" (or it's empty):
                  return assistant_msg["content"]   # we're done
             d. Otherwise, for each tool_call:
                  tool_msg = self._execute_tool_call(tool_call)
                  self.messages.append(tool_msg)
          3. If the loop exits without a final answer, raise RuntimeError
             (the model is stuck calling tools forever).
        """

        self.messages.append({"role": "user", "content": user_message})
        for _ in range(self.max_steps):
            assistant_msg = self._call_model()
            self.messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                return assistant_msg["content"] or ""

            for tool_call in tool_calls:
                tool_msg = self._execute_tool_call(tool_call)
                self.messages.append(tool_msg)

        raise RuntimeError(
            f"Agent exceeded max_steps ({self.max_steps}) without a final answer"
        )

    # ----- Internals -------------------------------------------------------

    def _call_model(self) -> dict[str, Any]:
        """Call OpenAI with current messages + tool schemas; return assistant msg.

        Return value must be a *plain dict* that we can append to self.messages
        and re-send to the model on the next turn. When `self.stream` is enabled,
        final-answer tokens are printed to stdout as they arrive.
        """
        if self.stream:
            return self._call_model_stream()
        return self._call_model_sync()

    def _call_model_sync(self) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        return self._assistant_msg_from_sdk_message(resp.choices[0].message)

    def _call_model_stream(self) -> dict[str, Any]:
        """Stream tokens for the final answer; accumulate tool calls from deltas."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            stream=True,
        )

        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_by_index[idx]
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

        content = "".join(content_parts) or None
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls_by_index:
            assistant_msg["tool_calls"] = [
                tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
            ]

        return assistant_msg

    @staticmethod
    def _assistant_msg_from_sdk_message(msg: Any) -> dict[str, Any]:
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return assistant_msg


    def _execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Run one tool_call against our server, return a `role='tool'` message.

        A tool message looks like:
            {
                "role": "tool",
                "tool_call_id": <id>,
                "content": <stringified JSON result>,
            }

        TODO (live):
          - Pull name = tool_call["function"]["name"]
          - Pull raw_args = tool_call["function"]["arguments"]  # JSON string
          - args = json.loads(raw_args) if raw_args else {}
          - result = self.tools.call(name, args)
          - On any exception, set content to json.dumps({"error": str(e)})
            so the model can see the failure and recover.
          - Return the tool message dict (note: content must be a STRING).
        """
        name = tool_call["function"]["name"]
        raw_args = tool_call["function"]["arguments"]
        try:
            args = json.loads(raw_args) if raw_args else {}
            res = self.tools.call(name, args)
            content = json.dumps(res["result"])
        except Exception as e:
            content = json.dumps({"error": str(e)})

        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": content,
        }

