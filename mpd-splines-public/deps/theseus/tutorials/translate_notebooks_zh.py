#!/usr/bin/env python3
"""
Generate Chinese-translated copies of Theseus tutorial notebooks.

- Keeps the original English notebooks untouched.
- Writes new files next to them with suffix `_zh.ipynb`.
- Clears code-cell outputs in the Chinese copies to keep file sizes small and avoid embedding large binary outputs.

Usage:
  python3 translate_notebooks_zh.py

Notes:
  This script uses `deep-translator` (GoogleTranslator backend).
  Install:
    python3 -m pip install --user deep-translator
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


class TranslationError(RuntimeError):
    pass


def _load_translator():
    try:
        from deep_translator import GoogleTranslator  # type: ignore
    except Exception as e:  # pragma: no cover
        raise TranslationError(
            "Missing dependency `deep-translator`. Install with: "
            "python3 -m pip install --user deep-translator"
        ) from e
    return GoogleTranslator(source="en", target="zh-CN")


_H_TAG_RE = re.compile(r"(<h[1-6][^>]*>)(.*?)(</h[1-6]>)", re.IGNORECASE | re.DOTALL)
_FENCED_CODE_RE = re.compile(r"(^```.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MATH_BLOCK_RE = re.compile(r"(\$\$.*?\$\$)", re.DOTALL)
_MATH_INLINE_RE = re.compile(r"(\$[^$\n]+\$)")
_URL_RE = re.compile(r"(https?://[^\s)>\"]+)")


def _protect_patterns(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace segments we don't want to translate with stable placeholders.
    """
    replacements: Dict[str, str] = {}
    counter = 0

    def replace_with_token(match: re.Match) -> str:
        nonlocal counter
        token = f"ZXQ_TOKEN_{counter:04d}_QXZ"
        replacements[token] = match.group(0)
        counter += 1
        return token

    # Order matters: protect large blocks first.
    text = _FENCED_CODE_RE.sub(replace_with_token, text)
    text = _MATH_BLOCK_RE.sub(replace_with_token, text)
    text = _INLINE_CODE_RE.sub(replace_with_token, text)
    text = _MATH_INLINE_RE.sub(replace_with_token, text)
    text = _URL_RE.sub(replace_with_token, text)

    return text, replacements


def _restore_patterns(text: str, replacements: Dict[str, str]) -> str:
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


def _translate_text(translator, text: str) -> str:
    # Skip empty or "already non-English" chunks to reduce bad translations.
    if not text.strip():
        return text
    if not re.search(r"[A-Za-z]", text):
        return text

    # Translate paragraph-by-paragraph to avoid request size/timeout issues.
    parts = re.split(r"(\n\s*\n)", text)
    out: List[str] = []
    for part in parts:
        if part.strip() == "":
            out.append(part)
            continue
        if part.startswith("\n") and part.strip() == "":
            out.append(part)
            continue
        # Only translate pieces containing ASCII letters.
        if re.search(r"[A-Za-z]", part):
            try:
                out.append(translator.translate(part))
            except Exception:
                out.append(part)
        else:
            out.append(part)
    return "".join(out)


def translate_markdown(translator, md: str) -> str:
    # Translate inner text of <h*> tags while preserving HTML tags.
    def htag_repl(match: re.Match) -> str:
        open_tag, inner, close_tag = match.group(1), match.group(2), match.group(3)
        protected, repl = _protect_patterns(inner)
        translated = _translate_text(translator, protected)
        translated = _restore_patterns(translated, repl)
        return open_tag + translated + close_tag

    md = _H_TAG_RE.sub(htag_repl, md)

    protected, repl = _protect_patterns(md)
    translated = _translate_text(translator, protected)
    translated = _restore_patterns(translated, repl)
    return translated


def clear_outputs(nb: dict) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def add_header_cell(nb: dict, english_name: str) -> None:
    header = (
        "# Theseus 教程（中文翻译版）\n\n"
        f"- 原始英文版：`{english_name}`\n"
        "- 说明：本文件为自动翻译版本（保留代码不翻译，清空输出以减小体积）。如遇术语不一致，可优先参考英文原文。\n"
    )
    header_cell = {"cell_type": "markdown", "metadata": {}, "source": header}
    nb.setdefault("cells", [])
    nb["cells"] = [header_cell] + nb["cells"]


def main() -> int:
    root = Path(__file__).resolve().parent
    translator = _load_translator()

    ipynb_paths = sorted(root.glob("*.ipynb"))
    for p in ipynb_paths:
        if p.name.endswith("_zh.ipynb"):
            continue

        nb = json.loads(p.read_text(encoding="utf-8"))

        # Translate markdown cells.
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                src = cell.get("source", "")
                if isinstance(src, list):
                    src = "".join(src)
                cell["source"] = translate_markdown(translator, src)

        clear_outputs(nb)
        add_header_cell(nb, p.name)

        out_path = p.with_name(p.stem + "_zh.ipynb")
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
