"""Shared sanitization, parsing, and path-sandbox helpers.

Used by retain/reflect/reason/promote/provider so every LLM prompt and
every filesystem write goes through one well-tested choke point.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_PROMPT_SPAN = 2000


def env_int(name: str, default: int, *, minimum: Optional[int] = None,
            maximum: Optional[int] = None) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid integer for %s=%r; using default %s", name, raw, default)
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_float(name: str, default: float, *, minimum: Optional[float] = None,
              maximum: Optional[float] = None) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        value = float(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in ("false", "0", "no", "off", "")


def clamp_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def prompt_escape(text: str, max_len: int = _MAX_PROMPT_SPAN) -> str:
    """JSON-escape text for embedding inside an LLM prompt.

    Strips NULs, truncates, and escapes quotes/control chars so a memory
    cannot close a string and inject a new instruction.
    """
    if not text:
        return ""
    cleaned = str(text).replace("\x00", "")[:max_len]
    return json.dumps(cleaned, ensure_ascii=False)[1:-1]


def yaml_quote(value: str) -> str:
    """Return a double-quoted YAML scalar safe for frontmatter."""
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def parse_json_object(response: str, default: dict) -> dict:
    """Extract the first valid JSON object from an LLM response, else default."""
    if not response:
        return dict(default)
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(response):
        pos = response.find("{", idx)
        if pos == -1:
            break
        try:
            result, _ = decoder.raw_decode(response, pos)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        idx = pos + 1
    logger.warning("No valid JSON object found in LLM response: %.200s...", response)
    return dict(default)


def parse_json_array(response: str) -> list:
    """Extract the first valid JSON array from an LLM response, else [].

    Handles markdown fences. Uses the JSON decoder (not naive bracket
    matching) so nested/contaminated payloads cannot smuggle extra items.
    """
    if not response:
        return []
    text = response.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    idx = 0
    while idx < len(text):
        pos = text.find("[", idx)
        if pos == -1:
            break
        try:
            result, _ = decoder.raw_decode(text, pos)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        idx = pos + 1
    logger.warning("Failed to parse LLM response as JSON array: %.200s...", text)
    return []


_INJECTION_RE = re.compile(
    r"(?i)\b("
    r"ignore (?:all |any )?(?:previous|prior|above) instructions?"
    r"|forget (?:everything|all (?:previous|prior) instructions?)"
    r"|you are now\b"
    r"|new system prompt"
    r"|override (?:these|the) (?:rules|instructions)"
    r"|bypass (?:these|the) (?:rules|instructions|filter)"
    r"|return (?:only )?json:"
    r"|extract this memory"
    r")\b"
)


def looks_like_injection(text: str) -> bool:
    """True if text looks like a prompt-injection attempt, not a real memory."""
    if not text:
        return False
    return _INJECTION_RE.search(text) is not None


def resolve_under(path: Path | str, root: Path, *, must_exist: bool = False) -> Path:
    """Resolve `path` and require it to stay inside `root`.

    Raises ValueError on traversal (absolute paths outside root, `..`
    escapes, symlink hops that leave the sandbox).
    """
    root_resolved = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Path {path!s} escapes sandbox root {root_resolved}"
        ) from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def is_under(path: Path | str, root: Path) -> bool:
    try:
        resolve_under(path, root)
        return True
    except ValueError:
        return False
