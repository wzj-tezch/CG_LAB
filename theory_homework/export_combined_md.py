# -*- coding: utf-8 -*-
"""合并四份理论课作业 README 为一份 Markdown。"""
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "202411081003-武子杰-图形学理论作业.md")

SECTIONS = [
    ("hw1_basics/README.md", None),
    ("hw2_geometry/README.md", "hw2_geometry"),
    ("hw3_rendering/README.md", None),
    ("hw4_animation/README.md", "hw4_animation"),
]

HEADER = """# 计算机图形学理论课作业

| 学号 | 202411081003 |
|------|--------------|
| 姓名 | 武子杰 |

> 合并提交版：基础知识 · 几何 · 渲染 · 动画

## 目录

1. [作业一 · 基础概念部分](#作业一--基础概念部分)
2. [作业二 · 几何部分](#作业二--几何部分)
3. [作业三 · 渲染部分](#作业三--渲染部分)
4. [作业四 · 动画部分](#作业四--动画部分)

---

"""


def fix_images(text, img_prefix):
    if not img_prefix:
        return text

    def repl_link(m):
        path = m.group(1)
        full = f"{img_prefix}/{path}" if "/" not in path else path
        name = os.path.splitext(path)[0].replace("_", " ")
        return f"![{name}]({full})"

    text = re.sub(r"\[`([^`]+\.(?:png|jpg|gif))`\]\(\1\)", repl_link, text)
    return text


def strip_duplicate_info(text):
    lines = text.splitlines()
    out = []
    skip_table = False
    for line in lines:
        if line.startswith("| 姓名 |") or line.startswith("|------|") and "姓名" in text:
            if line.startswith("| 姓名 |"):
                skip_table = True
            continue
        if skip_table:
            if line.startswith("| 学号 |"):
                continue
            if line.strip() == "---":
                skip_table = False
                continue
            if not line.startswith("|"):
                skip_table = False
            else:
                continue
        out.append(line)
    return "\n".join(out).lstrip("\n")


def main():
    parts = [HEADER]
    for rel, img_prefix in SECTIONS:
        path = os.path.join(BASE, *rel.split("/"))
        with open(path, encoding="utf-8") as f:
            body = f.read()
        body = strip_duplicate_info(body)
        body = fix_images(body, img_prefix)
        parts.append(body)
        parts.append("\n\n---\n\n")
    content = "".join(parts).rstrip("\n- \n") + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
