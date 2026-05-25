# Audit T4.4 — GPU 部署基础设施

**Auditor**: auditor
**Date**: 2026-05-25
**Commit**: 90009c0
**Verdict**: **PASS**

## 审核维度与结果

### 1. 脚本语法检查
```
bash -n scripts/sync_to_gpu.sh   → OK
bash -n scripts/sync_from_gpu.sh → OK
bash -n scripts/run_on_gpu.sh    → OK
```
全部通过,无语法错误。三个脚本均 `set -euo pipefail`,fail-fast 设置正确。

### 2. 无硬编码密码
`grep -nE "password|passwd|sshpass.*-p[ '\"]" scripts/*.sh GPU_SETUP.md` → **NO_HARDCODED_PW**

密码处理路径(sync_to_gpu.sh:48-62, run_on_gpu.sh:43-51, sync_from_gpu.sh:38-50):
- 通过 `GPU_SSH_PASS` 环境变量传入
- 经 `sshpass -e`(从 `SSHPASS` env 读取)调用,避免出现在命令行 ps 输出
- 缺失 sshpass 时降级到 ssh key / 交互式密码
- GPU_SETUP.md:108 也建议 `export GPU_SSH_PASS=...`,无明文写入

### 3. GPU_SETUP.md 顺序与可执行性
0-10 步骤顺序合理:
0 租机 → 1 SSH 登录 → 2 拉代码 → 3 装环境(A/B 双方案) → 4 上传数据 → 5 GPU 可见性测试 → 6 debug 跑通 → 7 正式训练(nohup+bg) → 8 监控(nvidia-smi/TB/log) → 9 回传 → 10 关机 checklist。
新人可照抄,关键 trap(`--delete` 危险性、OOM 调 batch、AutoDL 抢占)均在常见问题里覆盖。

### 4. requirements-gpu.txt 含 torch 2.2+cu121
```
11: --extra-index-url https://download.pytorch.org/whl/cu121
14: torch==2.2.0+cu121
15: torchvision==0.17.0+cu121
16: torchaudio==2.2.0+cu121
```
满足要求。

### 5. rsync 排除项
sync_to_gpu.sh:107-116 已排除:
`__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.git/`, `.claude/`, `.DS_Store`, `*.log`, `ckpt/`, `logs/`, `runs/`

完全覆盖审核清单(`__pycache__/`, `.git/`, `ckpt/`)且额外排除了 .claude、IDE 元数据、日志/runs。

## 加分项
- `--delete` 默认关闭,且开启时打印警告(sync_to_gpu.sh:95) — 防止误删 ckpt/logs
- `--dry-run` 一等公民,GPU_SETUP.md:122 也教用户先 dry-run
- sync_from_gpu.sh 在远端目录不存在时仅 warn 不退出,体面降级
- run_on_gpu.sh `--bg` 模式记录 PID 到 `logs/<config>.pid`,可后续 kill/监控
- 支持 `user@host:port` 写法解析(三脚本一致)

## 小建议(非阻塞,不影响 PASS)
1. sync_to_gpu.sh:50 `StrictHostKeyChecking=accept-new` 安全合理;GPU_SETUP.md 第 1 步也提醒手动接受 fingerprint。无需改。
2. run_on_gpu.sh:65 `bash -lc '${REMOTE_CMD}'` 中 REMOTE_CMD 含变量但有 `\$!`/`\$(cat ...)` 转义,需确保 config 路径不含单引号(项目当前 yaml 命名无此风险)。可在后续 T4.3 提交后复核 config 文件名。
3. requirements.txt / requirements-gpu.txt 未做版本上界 pin(只 pin torch);如未来 numpy/xarray 重大变更可能影响复现,但当前阶段不强求。

## 结论
**PASS** — 工具脚本质量超过轻量审核预期,无安全/可用性阻塞问题,可作为下游 T4.3/T4.5 的部署基础。
