"""
Setuptools compatibility entrypoint.

Background:
- In some environments (e.g., user site-packages not writable / PEP 660 not available),
  `pip install -e` may fail.
- A minimal `setup.py` allows some toolchains to fall back to the legacy installation path.

Note:
- The recommended way to run DMEval is still:
  `PYTHONPATH=dmeval/src python -m dmeval ...`
  This avoids installation permissions and is friendlier for paper/reproducibility workflows.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
