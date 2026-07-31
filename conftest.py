# Repo-root conftest — prevents pytest from walking past this directory.
# Without this, pytest's prepend import mode treats the plugin-root __init__.py
# (the Hermes entry point) as the top-level "galaxymem" package, shadowing the
# real importable package inside galaxymem/.
