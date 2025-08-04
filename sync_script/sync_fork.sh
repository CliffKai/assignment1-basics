#!/bin/bash

echo "=== 同步 Fork 仓库 ==="

# 拉取上游仓库更新
git fetch upstream

# 合并上游分支到本地 main 分支
git checkout main
git merge upstream/main

# 推送合并后的代码到你 GitHub 的 fork 仓库
git push origin main

echo "✅ 同步完成"
