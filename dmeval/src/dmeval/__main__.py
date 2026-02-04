"""
允许通过 `python -m dmeval.cli ...` 或 `python -m dmeval ...` 运行。

在这个项目里我们把 CLI 放在 `dmeval.cli`，`__main__` 只是做一次转发，
保证不同运行方式都能落到同一条调用链上。
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
