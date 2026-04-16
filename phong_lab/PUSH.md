# 推送到 GitHub：wzj-tezch/CG_LAB

本机已整理好 `phong_lab/` 文件夹（含 `README.md` 即原实验报告、`assets/preview.gif`、代码与 `requirements.txt`）。

在能访问 GitHub 的网络环境下执行（将路径改成你的本机路径）：

```powershell
cd D:\桌面\cg
git clone https://github.com/wzj-tezch/CG_LAB.git CG_LAB_work
Copy-Item -Recurse -Force .\phong_lab .\CG_LAB_work\phong_lab
Copy-Item -Force .\CG_LAB_README.md .\CG_LAB_work\README.md
cd CG_LAB_work
git add phong_lab README.md
git status
git commit -m "Add phong_lab: Phong lighting (Taichi ray casting + UI)"
git push origin main
```

若默认分支不是 `main`，请改为 `master` 或当前远程默认分支。

也可在已有本地克隆中直接把 `phong_lab` 整个文件夹复制进仓库根目录，再用上面的 `README.md` 覆盖仓库根 `README.md` 后 `git add` / `commit` / `push`。

SSH 远程示例：`git@github.com:wzj-tezch/CG_LAB.git`
