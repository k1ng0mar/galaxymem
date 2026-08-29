"""Secret detection and redaction for text entering persistent storage.

Applied at the store boundary (add_memory / add_flag / session summaries)
AND in retain.py before flagged turn text is written, so credentials never
persist even when the LLM extraction correctly declines to memorize them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_REDACTED = "[REDACTED]"
_MAX_PASSES = 32

# (name, compiled pattern) — each pattern matches the secret span only.
# Keep patterns tight: false positives poison recall; false negatives leak.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai_key", re.compile(
        r"\bsk-(?:proj-|svcacct-|ant-|none-)?[A-Za-z0-9_-]{20,}\S*"
    )),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{3,}"
    )),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b"
    )),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b")),
    ("xai_key", re.compile(r"\bxai-[A-Za-z0-9]{20,}\b")),
    ("telegram_bot", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("bearer_token", re.compile(
        r"(?i)\b(?:bearer|authorization)\s+[A-Za-z0-9._\-+=/]{16,}"
    )),
    ("connection_string", re.compile(
        r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s]{8,}"
    )),
    ("private_key_block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    )),
    ("private_key_begin", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
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
            # Never re-match a span we already replaced.
            span = text[m.start():m.end()]
            if span == _REDACTED:
                continue
            hits.append(SecretHit(kind=kind, start=m.start(), end=m.end()))
    return sorted(hits, key=lambda h: (h.start, -h.end))


def redact_secrets(text: str) -> str:
    """Replace secret spans with [REDACTED], preserving surrounding text."""
    if not text:
        return text
    out = text
    for _ in range(_MAX_PASSES):
        hits = find_secrets(out)
        if not hits:
            return out
        merged: list[SecretHit] = []
        for h in hits:
            if merged and h.start < merged[-1].end:
                merged[-1].end = max(merged[-1].end, h.end)
            else:
                merged.append(SecretHit(kind=h.kind, start=h.start, end=h.end))
        for h in reversed(merged):
            out = out[:h.start] + _REDACTED + out[h.end:]
    return out
