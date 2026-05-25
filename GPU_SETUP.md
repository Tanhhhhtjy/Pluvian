# Pluvian GPU 机部署手册（按顺序照抄）

> 目标环境：AutoDL · A800 40GB PCIe · Ubuntu 22.04 · CUDA 12.1 · PyTorch 2.2
> 工作目录约定：`/workspace/pluvian`
>
> **照着 0 → 8 一步步执行即可，不要跳步。** 本手册中所有以 `本机$` 开头的命令在你自己的电脑上跑，
> 以 `远端$` 开头的命令在 SSH 登入 GPU 机之后跑。

---

## 0. 租机器准备（AutoDL 控制台）

- **平台**：AutoDL（https://www.autodl.com）
- **实例规格**：A800 40GB PCIe，16C 64GB RAM，200GB SSD
- **系统镜像**：选 `PyTorch 2.2 + CUDA 12.1 + Ubuntu 22.04`（如有 cu121 直接对应的镜像优先）
- **计费方式**：建议「按量计费」，首次跑通后再考虑包月
- **数据盘**：勾选数据盘 ≥ 200GB（雷达 + ERA5 + 派生量大概 ~120GB）
- 实例开机后，控制台会给一个 SSH 命令，形如：

  ```
  ssh -p 24680 root@connect.cqa1.seetacloud.com
  密码: <控制台复制>
  ```

  下文统一记为 `user@host:port`，例：`root@connect.cqa1.seetacloud.com:24680`。

---

## 1. 首次登录 + 接受 fingerprint

```bash
本机$ ssh -p 24680 root@connect.cqa1.seetacloud.com
# 第一次会问 fingerprint，输入 yes
# 然后输入控制台给的密码
```

登入后先确认在 GPU 机上：

```bash
远端$ nvidia-smi          # 应该看到 A800 40GB
远端$ pwd                 # 一般在 /root
远端$ df -h /workspace    # 确认数据盘挂上
```

---

## 2. 拉代码（GPU 机上）

```bash
远端$ mkdir -p /workspace && cd /workspace
远端$ git clone https://github.com/Tanhhhhtjy/Pluvian.git pluvian
远端$ cd pluvian
远端$ git checkout phase4/model-architecture
远端$ git log -1 --oneline    # 确认在最新提交
```

> 如果仓库是私有的：先 `ssh-keygen -t ed25519` 在 GPU 机生成 key，把 `~/.ssh/id_ed25519.pub`
> 内容贴到 GitHub → Settings → SSH keys，然后用 `git clone git@github.com:...` 的 SSH 地址。

---

## 3. 装环境

AutoDL 的官方镜像里通常已经有一个 `/opt/venv` 或 conda 环境装好了 cu121 的 PyTorch。
我们要做的是把 Pluvian 的依赖补上。

**方案 A：直接用镜像自带的 venv（推荐）**

```bash
远端$ source /opt/venv/bin/activate    # 若镜像没这个路径，跳到方案 B
远端$ which python && python --version
远端$ python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应输出: 2.2.0+cu121 True

远端$ cd /workspace/pluvian
远端$ pip install -r requirements.txt
远端$ pip install -r requirements-gpu.txt
```

**方案 B：自己建 venv**

```bash
远端$ apt update && apt install -y python3.10-venv
远端$ python3 -m venv /opt/venv
远端$ source /opt/venv/bin/activate
远端$ pip install --upgrade pip
远端$ cd /workspace/pluvian
远端$ pip install -r requirements.txt
远端$ pip install -r requirements-gpu.txt
```

装完验证一次：

```bash
远端$ python -c "import torch, xarray, metpy, einops; print('ok')"
```

---

## 4. 上传数据（在本机执行）

> 雷达 + PWV + ERA5 + derived 总量 ~120GB，按 100MB/s 算大概 20 分钟。
> 第一次同步全量，之后只动 code-only 增量。

**先在本机设置密码到环境变量**（避免每次输入）：

```bash
本机$ export GPU_SSH_PASS='控制台给的密码'
```

如果本机没装 sshpass：

```bash
本机$ sudo apt install sshpass        # Ubuntu/Debian
# 或 macOS: brew install hudochenkov/sshpass/sshpass
```

预览要传什么（强烈建议第一次先 dry-run）：

```bash
本机$ cd /Data/tanh/npj
本机$ bash scripts/sync_to_gpu.sh root@connect.cqa1.seetacloud.com:24680 --data-only --dry-run
```

确认无误后正式跑：

```bash
本机$ bash scripts/sync_to_gpu.sh root@connect.cqa1.seetacloud.com:24680 --data-only
```

代码增量（每次改完代码都跑一次）：

```bash
本机$ bash scripts/sync_to_gpu.sh root@connect.cqa1.seetacloud.com:24680 --code-only
```

> **不要**轻易加 `--delete`，会把远端多余文件全删（包括 ckpt/logs，会哭）。

---

## 5. 测试 GPU 可见性 & 数据可读

```bash
远端$ cd /workspace/pluvian
远端$ source /opt/venv/bin/activate
远端$ python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('Mem GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))"
# 期望: CUDA available: True / Device: NVIDIA A800 ... / Mem GB: 40.0

远端$ python -c "import xarray as xr; ds = xr.open_dataset('2023_05-08_ERA5_PWV.nc'); print(ds)"
# 应能正常打印 ERA5 数据集结构
```

---

## 6. 跑首组 baseline（debug 模式，确认 pipeline 通）

```bash
远端$ cd /workspace/pluvian
远端$ source /opt/venv/bin/activate
远端$ python scripts/train.py --config configs/ablation_1_radar_only.yaml --debug
```

`--debug` 应该只跑 1~2 个 epoch、小 batch，几分钟内看到 loss 在下降即可。
如果 OOM，把 config 里的 `batch_size` 减半再试。

---

## 7. 正式训练（后台跑 + 日志）

```bash
远端$ cd /workspace/pluvian
远端$ source /opt/venv/bin/activate
远端$ mkdir -p logs
远端$ nohup python scripts/train.py --config configs/ablation_5_budget_loss.yaml \
    > logs/abl5.log 2>&1 &
远端$ echo $! > logs/abl5.pid
远端$ tail -f logs/abl5.log
```

或者直接从本机一键启动（推荐）：

```bash
本机$ bash scripts/run_on_gpu.sh root@connect.cqa1.seetacloud.com:24680 \
        configs/ablation_5_budget_loss.yaml --bg
```

---

## 8. 监控

### GPU 利用率

```bash
远端$ watch -n 5 nvidia-smi
```

健康指标：

- GPU-Util > 80%
- Memory-Usage 接近 38–39 GB（不爆 40GB）
- Power 200W+

### TensorBoard

GPU 机上：

```bash
远端$ cd /workspace/pluvian
远端$ source /opt/venv/bin/activate
远端$ tensorboard --logdir runs/ --bind_all --port 6006
```

本机另开一个终端做端口转发：

```bash
本机$ ssh -p 24680 -L 6006:localhost:6006 root@connect.cqa1.seetacloud.com -N
# 然后浏览器打开 http://localhost:6006
```

### 训练日志

```bash
远端$ tail -f /workspace/pluvian/logs/abl5.log
```

---

## 9. 把产物拉回本机

训练跑完（或想中途看 ckpt），在本机执行：

```bash
本机$ cd /Data/tanh/npj
本机$ bash scripts/sync_from_gpu.sh root@connect.cqa1.seetacloud.com:24680 --all
```

只想要 ckpt：

```bash
本机$ bash scripts/sync_from_gpu.sh root@connect.cqa1.seetacloud.com:24680 --ckpt
```

---

## 10. 关机前 checklist

- [ ] `ckpt/`、`logs/`、`runs/` 都已经 sync_from_gpu 拉回本机
- [ ] 没有正在跑的训练进程：`ps -ef | grep train.py`
- [ ] AutoDL 控制台 → 关机（按量计费记得关，不然继续扣费）

---

## 常见问题

**Q: `torch.cuda.is_available()` 返回 False**
A: `nvidia-smi` 看到卡但 PyTorch 看不到 → 通常是 venv 没激活，或装错了 cpu 版 torch。
   重装：`pip uninstall torch torchvision torchaudio && pip install -r requirements-gpu.txt`

**Q: rsync 报 `sshpass: command not found`**
A: 本机 `sudo apt install sshpass`，或不用密码改用 ssh key（推荐长期方案）。

**Q: OOM (CUDA out of memory)**
A: 改 config 里的 `batch_size` / `seq_len`，或加 `--amp`（混合精度）。A800 40GB 跑雷达 2D-CNN
   一般 batch 8–16 安全。

**Q: 训练中途断了**
A: AutoDL 偶尔抢占。务必：
   - config 里开 `save_every_n_epoch: 1`
   - 用 `nohup` + `&` 后台跑（终端断了不会中断）
   - 重连后 `tail -f logs/xxx.log` 看进度
