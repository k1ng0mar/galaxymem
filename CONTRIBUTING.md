# Contributing to GalaxyMem

Contributions welcome. This guide covers the basics.

## Setup

```bash
git clone https://github.com/k1ng0mar/galaxymem.git
cd galaxymem
pip install -e .
```

## Code style

- Python 3.10+ (uses `from __future__ import annotations`)
- Follow the existing patterns in each module
- The store has zero business logic — keep it that way
- Only `provider.py` imports Hermes internals. Everything else is standalone.
- No new dependencies without justification
- Comments explain *why*, not *what*

## Architecture constraints

Read **[ARCHITECTURE.md](ARCHITECTURE.md)** before making changes. Key rules:

1. **Module boundaries are sacred.** The store is storage only. Logic lives in
   `retain.py`, `recall.py`, `reflect.py`, `promote.py`. No upward imports.
2. **Identity links are explicit-only.** Never auto-link platform identities to
   entities. See design decision D3 in ARCHITECTURE.md.
3. **Never hard-delete memories.** `delete_memory` is a soft `archived` status
   (design decision D13).
4. **Promotion is human-gated.** GalaxyMem never writes to the user's wiki /
   Obsidian vault without explicit approval (design decision D12).
5. **No unprompted test runs.** Run at most one targeted test per change,
   right before commit. Static verification is preferred.

## Pull requests

1. Fork the repo and create a feature branch
2. Update docs alongside code (ARCHITECTURE.md, AGENTS.md, README.md)
3. Keep changes surgical — one concern per PR
4. Reference any issues in the commit message

## Reporting bugs

Include:
- GalaxyMem version (`python -c "import galaxymem; print(galaxymem.__version__)"`)
- Python version
- Whether running standalone or as a Hermes plugin
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree your contributions are licensed under the MIT
License (see [LICENSE](LICENSE)).
