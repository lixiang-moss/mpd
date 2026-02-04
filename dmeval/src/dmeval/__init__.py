"""
DMEval vNext（L1）包入口。

这个包实现了论文/工具规格里描述的 DMEval 管线：
- Stage I: tune（调参，输出 best_configs）
- Stage II: compare（读取 best_configs 做跨场景公平对比）

注意：DMEval **不计算指标**，只负责：
1) 串行调用被测系统（例如 MPD）执行推理；
2) 从结果目录抽取指标（Adapter）；
3) 聚合/排序/可视化并写出统一产物（CSV/plots/manifest）。
"""

__all__ = ["__version__"]

# 版本号：用于输出 manifest/论文复现记录时标识工具版本。
__version__ = "0.1.0"
