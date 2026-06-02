"""
tools.py
--------
Defines the agent's *reducer table* - a map from tool name to Python
callable. Each callable is one **step** of the state machine that the
agent runs to satisfy a user request.

The framing:
  - STATE       = the running conversation (the list of messages).
  - ACTIONS     = the tools below.
  - REDUCER     = TOOL_REGISTRY (a dict / "map object") that maps each
                  action name to the function that produces new context.
  - TRANSITIONS = the agent loop in `agent.py` appends each tool's
                  return value back to STATE, then asks the LLM what
                  to do next.

The system prompt in `agent.py` is the *spec* for this state machine;
the keys in TOOL_REGISTRY are the only legal actions.

During live coding you'll:
  - Implement `day_of_week_from_nanotime()` and `translate()`
  - Finish the TOOL_SCHEMAS list so the model knows each action's contract
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. Tool implementations (the reducer functions)
# ---------------------------------------------------------------------------

def get_nanotime() -> dict[str, int]:
    """Return the current Unix epoch time in nanoseconds (UTC).

    Fully implemented as a reference for the others.
    Note: nanoseconds, not seconds - see `time.time_ns()`.
    """
    return {"nanotime": time.time_ns()}


def day_of_week_from_nanotime(nanotime: int) -> dict[str, str]:
    """Return the English day-of-week (e.g. "Tuesday") for a ns timestamp."""
    seconds = nanotime / 1_000_000_000
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return {"day_of_week": dt.strftime("%A")}


# A small bundled dictionary so we don't need a real translation API.
# Keys are *lowercase English words/phrases*; values are the translation.
# Add more entries as you like - this is intentionally tiny.
_TRANSLATIONS: dict[str, dict[str, str]] = {
    "spanish": {
        "monday": "lunes",
        "tuesday": "martes",
        "wednesday": "miércoles",
        "thursday": "jueves",
        "friday": "viernes",
        "saturday": "sábado",
        "sunday": "domingo",
        "hello": "hola",
        "goodbye": "adiós",
        "thanks": "gracias",
    },
    "french": {
        "monday": "lundi",
        "tuesday": "mardi",
        "wednesday": "mercredi",
        "thursday": "jeudi",
        "friday": "vendredi",
        "saturday": "samedi",
        "sunday": "dimanche",
        "hello": "bonjour",
        "goodbye": "au revoir",
        "thanks": "merci",
    },
    "german": {
        "monday": "Montag",
        "tuesday": "Dienstag",
        "wednesday": "Mittwoch",
        "thursday": "Donnerstag",
        "friday": "Freitag",
        "saturday": "Samstag",
        "sunday": "Sonntag",
        "hello": "hallo",
        "goodbye": "auf Wiedersehen",
        "thanks": "danke",
    },
}


def translate(text: str, target_language: str) -> dict[str, Any]:
    """Translate a single English word/phrase using the bundled dictionary.

    Returning errors as *data* (rather than raising) is intentional - the
    model gets to see what went wrong and can recover (e.g. retry with a
    different word or admit it doesn't know).

    """
    lang = target_language.strip().lower()
    key = text.strip().lower()

    if lang not in _TRANSLATIONS:
        return {
            "error": f"unsupported language: {target_language!r}",
            "supported": sorted(_TRANSLATIONS.keys()),
        }

    if key not in _TRANSLATIONS[lang]:
        return {
            "error": f"no translation for {text!r} in {lang}",
            "known_words": sorted(_TRANSLATIONS[lang].keys()),
        }

    return {
        "text": text,
        "target_language": lang,
        "translated": _TRANSLATIONS[lang][key],
    }


# ---------------------------------------------------------------------------
# 2. The reducer table - name -> Python callable
# ---------------------------------------------------------------------------
# This is the "map object used as a reducer" from the design. The agent
# can only invoke entries that live here. To add a new action: implement
# a function above, register it here, and add its schema below.

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_nanotime": get_nanotime,
    "day_of_week_from_nanotime": day_of_week_from_nanotime,
    "translate": translate,
}


# ---------------------------------------------------------------------------
# 3. OpenAI tool schemas (the action contracts the model sees)
# ---------------------------------------------------------------------------
# Docs: https://platform.openai.com/docs/guides/function-calling

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_nanotime",
            "description": "Get the current Unix epoch time in nanoseconds (UTC).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "day_of_week_from_nanotime",
            "description": "Return the English day-of-week for a Unix epoch time in nanoseconds (UTC).",
            "parameters": {
                "type": "object",
                "properties": {
                    "nanotime": {
                        "type": "integer",
                        "description": "Unix epoch time in nanoseconds (UTC).",
                    },
                },
                "required": ["nanotime"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate a single English word or short phrase using a bundled dictionary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "English word or short phrase to translate.",
                    },
                    "target_language": {
                        "type": "string",
                        "description": 'Target language: "spanish", "french", or "german".',
                    },
                },
                "required": ["text", "target_language"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 4. Dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up `name` in the reducer table and call it with **arguments.

    Raises:
        KeyError: if the tool name is not registered.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}")
    return TOOL_REGISTRY[name](**arguments)
