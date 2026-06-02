"""
main.py
-------
Tiny REPL for chatting with the agent from your terminal.

Prerequisites:
  1. Tool server running in another terminal:
        uv run uvicorn server:app --reload --port 8000
  2. .env file with OPENAI_API_KEY set (see .env.example).

Run:
    uv run python main.py
"""

from __future__ import annotations

import sys

from agent import Agent


def main() -> None:
    print("Agent ready. Type a message, or 'exit' to quit.\n")
    agent = Agent()
    try:
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            try:
                reply = agent.chat(user)
            except Exception as e:  # noqa: BLE001 - keep the REPL alive while learning
                print(f"[error] {e}", file=sys.stderr)
                continue
            print(f"bot> {reply}\n")
    finally:
        agent.tools.close()


if __name__ == "__main__":
    main()
