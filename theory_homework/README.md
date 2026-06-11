# 理论课作业

计算机图形学课程**理论课**四次作业答案（基础知识 / 几何 / 渲染 / 动画）。

**姓名**：武子杰 | **学号**：202411081003 | **专业**：计算机科学与技术（公费师范）

## 作业目录（点击进入查看 README）

| 作业 | 目录 | 说明 |
|------|------|------|
| 作业一 | [hw1_basics](hw1_basics/) | 基础概念（选择 + 简答） |
| 作业二 | [hw2_geometry](hw2_geometry/) | 几何（判断 + 选择 + 简答） |
| 作业三 | [hw3_rendering](hw3_rendering/) | 渲染（选择 + 简答） |
| 作业四 | [hw4_animation](hw4_animation/) | 动画（简答） |

每个子目录包含：

- **README.md** — 在 GitHub 上直接预览的答案（无 LaTeX，避免乱码）
- **202411081003-武子杰-理论课作业X-....docx** — 单份 Word 提交版

## LaTeX 提交版（推荐）

按课程要求合并四份理论作业，使用 LaTeX 排版，答案参考 `cg_homework/main.pdf`（详解学习版）。

| 路径 | 说明 |
|------|------|
| [`overleaf/main.tex`](overleaf/main.tex) | LaTeX 主文档（封面、目录、四份作业） |
| [`overleaf/hw1.tex` … `hw4.tex`](overleaf/) | 各次作业（题目 + 作答） |
| [`build_submission_tex.py`](build_submission_tex.py) | 从 `cg_homework/overleaf/` 重新生成 hw*.tex |
| [`compile.bat`](compile.bat) | 一键编译 PDF |

**编译步骤：**

```bat
cd theory_homework
python build_submission_tex.py
compile.bat
```

输出 PDF：`202411081003-武子杰-图形学理论作业.pdf`

需安装 [Tectonic](https://tectonic-typesetting.github.io/)。本机常见路径：

| 类型 | 路径 |
|------|------|
| 可执行文件 | `%LOCALAPPDATA%\Temp\tectonic\bin\tectonic.exe`（运行 `download_tectonic.py` 下载） |
| 宏包缓存 | `%LOCALAPPDATA%\TectonicProject\Tectonic\`（约 136 MB，按需下载 `.sty`/字体） |

也可将 `overleaf/` 上传 Overleaf 在线编译（XeLaTeX）。

---

## 合并提交版（PDF / Word）

按课程要求，四份作业已合并为一份文档：

| 文件 | 说明 |
|------|------|
| [202411081003-武子杰-图形学理论作业.pdf](202411081003-武子杰-图形学理论作业.pdf) | **正式提交 PDF**（含封面、目录、分页排版） |
| [202411081003-武子杰-图形学理论作业.md](202411081003-武子杰-图形学理论作业.md) | 合并 Markdown 版（四份作业合一） |
| [202411081003-武子杰-图形学理论作业-语雀.md](202411081003-武子杰-图形学理论作业-语雀.md) | **语雀导入版**（表格/图片/标题已适配） |
| [202411081003-武子杰-图形学理论作业.docx](202411081003-武子杰-图形学理论作业.docx) | 合并 Word 源文件 |

重新生成：运行 `python export_combined_pdf.py`、`python export_combined_md.py` 或 `python export_yuque_md.py`。
