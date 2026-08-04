"""支持使用 ``python -m trade_game`` 启动项目。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
