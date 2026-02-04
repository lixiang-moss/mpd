"""
Adapter 层包。

Adapter 的职责：
- 面向“被测系统的产物目录”做解析（discover/load/extract）
- 输出统一字段口径的 trial 行 dict，供 DMEval 聚合/排名/绘图

当前 L1 首个适配器是 MPDAdapter；未来新增其它 planner 时，建议在此目录新增对应文件。
"""

from .mpd import MPDAdapter

__all__ = ["MPDAdapter"]
