# -*- coding: utf-8 -*-
"""生成语雀兼容的合并 Markdown（修复表格、图片 URL、标题层级）。"""
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "202411081003-武子杰-图形学理论作业-语雀.md")
RAW = "https://raw.githubusercontent.com/wzj-tezch/CG_LAB/main/theory_homework"

SECTIONS = [
    ("hw1_basics/README.md", "作业一 · 基础概念部分", []),
    ("hw2_geometry/README.md", "作业二 · 几何部分", [
        ("de_casteljau_reference.png", f"{RAW}/hw2_geometry/de_casteljau_reference.png", "de Casteljau 三角剖分示意图"),
    ]),
    ("hw3_rendering/README.md", "作业三 · 渲染部分", []),
    ("hw4_animation/README.md", "作业四 · 动画部分", [
        ("physics_sim_flow.png", f"{RAW}/hw4_animation/physics_sim_flow.png", "物理仿真计算流程图"),
        ("skinning_lbs.png", f"{RAW}/hw4_animation/skinning_lbs.png", "线性混合蒙皮示意图"),
    ]),
]

HEADER = """# 计算机图形学理论课作业

| 学号 | 202411081003 |
| --- | --- |
| 姓名 | 武子杰 |

> 合并提交版：基础知识 · 几何 · 渲染 · 动画
>
> 语雀导入：知识库 → 新建 → 导入 → 选择本 .md 文件。导入后使用右侧「大纲」导航。

"""


def is_separator(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-+:?", c) for c in cells if c)


def is_table_row(line):
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def fix_table_block(rows):
    if not rows:
        return []
    if len(rows) >= 2 and is_separator(rows[1]):
        fixed = []
        for i, row in enumerate(rows):
            if is_separator(row):
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                fixed.append("| " + " | ".join(["---"] * len(cells)) + " |")
            else:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if not cells[0] and len(cells) >= 3:
                    cells[0] = "项目"
                fixed.append("| " + " | ".join(cells) + " |")
        return fixed
    data_rows = [r for r in rows if not is_separator(r)]
    cols = max(len([c for c in r.strip().strip("|").split("|")]) for r in data_rows)
    fixed = []
    for i, row in enumerate(data_rows):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        while len(cells) < cols:
            cells.append("")
        if i == 0:
            fixed.append("| " + " | ".join(cells) + " |")
            fixed.append("| " + " | ".join(["---"] * cols) + " |")
        else:
            if not cells[0] and cols >= 3:
                cells[0] = "项目"
            fixed.append("| " + " | ".join(cells) + " |")
    return fixed


def process_tables(text):
    lines = text.splitlines()
    out = []
    buf = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            if buf:
                out.extend(fix_table_block(buf))
                buf = []
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if is_table_row(line):
            buf.append(line)
        else:
            if buf:
                out.extend(fix_table_block(buf))
                buf = []
            out.append(line)
    if buf:
        out.extend(fix_table_block(buf))
    return "\n".join(out)


def strip_doc_header(text):
    lines = text.splitlines()
    out = []
    skip = False
    for line in lines:
        if line.startswith("# 计算机图形学理论课作业"):
            continue
        if line.startswith("| 姓名 |"):
            skip = True
            continue
        if skip:
            if line.strip() == "---":
                skip = False
            continue
        out.append(line)
    return "\n".join(out).lstrip("\n")


def replace_images(text, images):
    for fname, url, caption in images:
        text = re.sub(
            rf"参考图：.*{re.escape(fname)}.*",
            f"\n\n![{caption}]({url})\n",
            text,
        )
    return text


def demote_headings(text):
    lines = []
    for line in text.splitlines():
        if line.startswith("#### "):
            line = "#####" + line[4:]
        elif line.startswith("### "):
            line = "####" + line[3:]
        elif line.startswith("## "):
            line = "###" + line[2:]
        elif line.startswith("# "):
            line = "##" + line[1:]
        lines.append(line)
    return "\n".join(lines)


def normalize_text(text):
    return text.replace("→", "->")


def render_section(rel, title, images):
    path = os.path.join(BASE, *rel.split("/"))
    with open(path, encoding="utf-8") as f:
        body = f.read()
    body = strip_doc_header(body)
    body = replace_images(body, images)
    body = demote_headings(body)
    body = process_tables(body)
    body = normalize_text(body)
    return f"\n\n## {title}\n\n{body.strip()}\n"


def main():
    parts = [HEADER]
    for i, (rel, title, imgs) in enumerate(SECTIONS):
        parts.append(render_section(rel, title, imgs))
        if i < len(SECTIONS) - 1:
            parts.append("\n\n***\n\n")
    content = "".join(parts).strip() + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
