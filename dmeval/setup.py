"""
setuptools 兼容入口。

背景：
- 在某些环境里（例如用户站点包不可写/PEP660 不可用），`pip install -e` 可能失败
- 提供一个最小的 setup.py 可以让部分工具链退回到传统安装路径

注意：
- DMEval 推荐的运行方式仍然是：`PYTHONPATH=dmeval/src python -m dmeval.cli ...`
  这样不依赖安装权限，且更适合在论文复现实验里记录代码版本。
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
