"""Small persistent rolling memory for the voice conversation.

Gemini Live session resumption preserves context while the transport survives, but
an explicit process restart cannot rely on a server-side handle.  This module
keeps a compact transcript history under ``data/`` so a newly-created Live
session can be reminded of the recent conversation without storing raw audio.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ConversationMemory:
    def __init__(self, path: Path, *, max_turns: int = 40, max_chars: int = 16_000) -> None:
        self.path = path
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._turns: list[dict[str, Any]] = []
        self._load()

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).split()).strip()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return
        turns = payload.get("turns") if isinstance(payload, dict) else None
        if not isinstance(turns, list):
            return
        for item in turns:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            text = self._clean(str(item.get("text") or ""))
            if role in {"user", "assistant"} and text:
                self._turns.append({
                    "role": role,
                    "text": text,
                    "at": float(item.get("at") or 0.0),
                })
        self._trim()

    def _trim(self) -> None:
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]
        while self._turns and sum(len(item["text"]) for item in self._turns) > self.max_chars:
            self._turns.pop(0)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps({"version": 1, "turns": self._turns}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def append(self, role: str, text: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported conversation-memory role: {role}")
        cleaned = self._clean(text)
        if not cleaned:
            return
        if self._turns and self._turns[-1]["role"] == role and self._turns[-1]["text"] == cleaned:
            return
        self._turns.append({"role": role, "text": cleaned, "at": time.time()})
        self._trim()
        self._persist()

    def prompt_context(self) -> str:
        if not self._turns:
            return ""
        lines = [
            "Recent persistent voice-conversation memory follows. Treat it as prior dialogue context,",
            "not as a new user request. Continue naturally if the user refers back to it.",
        ]
        for item in self._turns:
            label = "User" if item["role"] == "user" else "ADA"
            lines.append(f"{label}: {item['text']}")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "turn_count": len(self._turns),
            "chars": sum(len(item["text"]) for item in self._turns),
            "path": str(self.path),
        }
