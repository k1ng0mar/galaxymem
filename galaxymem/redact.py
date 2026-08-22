"""Secret detection and redaction for text entering persistent storage.

Applied in retain.py BEFORE flagged turn text is written to the flags
table, so credentials never persist even when the LLM extraction
correctly declines to memorize them. (Pre-write filtering pattern
popularized by mnemosyne's write classifier.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_REDACTED = "[REDACTED]"

# (name, compiled pattern) — each pattern matches the secret span only.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|[A-Za-z0-9_-]{20,})\S*")),
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{3,}"
    )),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # key=value style: only when the value is long enough to be real
    ("credential_assignment", re.compile(
        r"(?i)\b(password|passwd|api[_-]?key|secret|token|auth[_-]?token)\b\s*[:=]\s*(\S{8,})"
    )),
]


@dataclass
class SecretHit:
    kind: str
    start: int
    end: int


def find_secrets(text: str) -> list[SecretHit]:
    """Return non-overlapping secret spans found in text, sorted by start."""
    if not text:
        return []
    hits: list[SecretHit] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            hits.append(SecretHit(kind=kind, start=m.start(), end=m.end()))
    return sorted(hits, key=lambda h: h.start)


def redact_secrets(text: str) -> str:
    """Replace secret spans with [REDACTED], preserving surrounding text."""
    if not text:
        return text
    out = text
    while True:
        hits = find_secrets(out)
        if not hits:
            return out
        # Merge overlapping spans (e.g. a jwt that also matches a
        # credential assignment) so we never double-replace.
        merged: list[SecretHit] = []
        for h in hits:
            if merged and h.start < merged[-1].end:
                merged[-1].end = max(merged[-1].end, h.end)
            else:
                merged.append(h)
        for h in reversed(merged):
            out = out[:h.start] + _REDACTED + out[h.end:]
