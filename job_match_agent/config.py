"""旧配置模块的兼容入口。

新代码统一从 ``app.core.config`` 导入 settings。
"""

from app.core.config import (
    DEFAULT_ENV_PATH,
    PROJECT_ROOT,
    load_env_file,
    settings,
)


__all__ = [
    "DEFAULT_ENV_PATH",
    "PROJECT_ROOT",
    "load_env_file",
    "settings",
]
