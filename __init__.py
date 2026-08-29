"""Hermes Agent plugin entry point for GalaxyMem.

This file lives at the plugin root (`~/.hermes/plugins/galaxymem/__init__.py`)
so Hermes's directory-plugin loader finds `register(ctx)`. The actual package
code lives in the `galaxymem/` subdirectory.
"""


def register(ctx) -> None:
    """Plugin entry point — called by Hermes memory plugin discovery.

    Loads the inner ``galaxymem/galaxymem/`` package by file-path rather than
    by name, because Hermes may add ``~/.hermes/plugins/`` to ``sys.path``
    and ``import galaxymem`` then resolves to THIS directory — which has
    no ``provider.py``, only the inner subpackage.  Loading via a synthetic
    unique module name (``_galaxymem_inner``) avoids the ambiguity entirely.
    """
    import importlib.util
    import sys
    from pathlib import Path

    _inner = Path(__file__).parent / "galaxymem"
    _NS = "_galaxymem_inner"

    if _NS not in sys.modules:
        _init = _inner / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            _NS, str(_init),
            submodule_search_locations=[str(_inner)],
        )
        if not spec or not spec.loader:
            raise ImportError(
                f"Cannot load inner GalaxyMem package at {_inner}"
            )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_NS] = pkg
        spec.loader.exec_module(pkg)

        # Pre-register submodules so their relative imports
        # (``from .store import Store``) resolve correctly.
        for pyfile in sorted(_inner.glob("*.py")):
            if pyfile.name == "__init__.py":
                continue
            sub_name = pyfile.stem
            full_name = f"{_NS}.{sub_name}"
            if full_name not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(
                    full_name, str(pyfile)
                )
                if sub_spec and sub_spec.loader:
                    sub_mod = importlib.util.module_from_spec(sub_spec)
                    sys.modules[full_name] = sub_mod
                    try:
                        sub_spec.loader.exec_module(sub_mod)
                    except Exception as exc:
                        import logging
                        logging.getLogger("galaxymem.plugin").warning(
                            "Failed to load inner module %s: %s", sub_name, exc,
                        )
    else:
        pkg = sys.modules[_NS]

    from _galaxymem_inner.provider import GalaxyMemProvider
    ctx.register_memory_provider(GalaxyMemProvider())
