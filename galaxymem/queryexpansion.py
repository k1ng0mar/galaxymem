"""Query expansion for deep recall.

Uses the auxiliary LLM to rewrite a terse natural-language query into multiple
related phrasings that improve coverage in both vector and keyword search.

"api port" → "api port endpoint service listen backend port 8010"

Cost: one cheap LLM call (max 150 tokens). No caching — different queries may
expand differently even for the same raw string (context matters).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from . import config as cfg

logger = logging.getLogger(__name__)


def expand_query(query: str, llm_client) -> str:
    """Expand a query with related phrasings via the auxiliary LLM.

    Returns the original query + " " + expansions concatenated for feeding into
    embedding search. Cheap: one LLM call, ~150-250 tokens.
    """
    if not query or len(query.strip()) < 3:
        return query

    # Skip expansion for very short queries — nothing to expand usefully.
    if len(query.split()) < 3:
        return query

    prompt = (
        f'Rewrite the following search query for a memory database into multiple '
        f'related phrasings that broaden its coverage (add synonyms, specify '
        f'context, guess the user\'s underlying intent). Concatenate them with '
        f'spaces. Do NOT explain, do NOT add commentary. '
        f'Original query: "{query.strip()}"\n\n'
        f'Expanded query:'
    )
    try:
        expanded = llm_client.complete(prompt)
        if not expanded:
            return query
        # Clean up any LLM hallucination markers
        cleaned = re.sub(r'^[^\w\s]*', '', expanded)
        cleaned = re.sub(r'\s+', ' ', cleaned)[:512]
        merged = query + " " + cleaned
        logger.debug("Query expanded: '%s' → '%s'", query, merged[:100])
        return merged
    except Exception as e:
        logger.warning("Query expansion failed, using original: %s", e)
        return query


def should_expand(query: str, entity_ids: Optional[list[str]] = None) -> bool:
    """Decide whether expansion is worth the LLM cost.

    Entity-scoped queries already have a tight filter and don't need the
    extra broadening — the entity filter is doing the relevance work.
    Queries that are already long don't need expansion either.
    """
    # Entity-scoped queries: no broadening needed
    if entity_ids:
        return False
    # Single-word queries: too vague for the LLM to improve meaningfully
    if len(query.split()) < 2:
        return False
    # Already long/detailed: diminishing returns
    if len(query) > 120:
        return False
    # clean "get me the" fillers
    q = query.strip()
    if re.match(r"^(get|show|find|tell me|what|list|search|recall|remember)\b", q, re.IGNORECASE):
        return True  # these need expansion to work with actual content
    return True  # default: expand


def update_config_from_env() -> None:
    """Reload expansion config from env vars."""
    cfg.QUERY_EXPANSION_LLM = os.environ.get(
        "GALAXYMEM_QUERY_EXPANSION", "true"
    ).lower() not in ("false", "0", "no", "off")
