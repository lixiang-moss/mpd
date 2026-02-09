"""
Allows running via `python -m dmeval ...` (or `python -m dmeval.cli ...`).

In this project the CLI lives in `dmeval.cli`. `__main__` only forwards to it so that
different invocation styles share the same code path.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
