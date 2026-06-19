# Earthformer Day 1 实际执行日志

> 时间：2026-06-19
> 计划：`docs/experiments/earthformer_reproduction_plan.md` §5 Day 1

## 1. Clone 上游仓库

```
cd /data4/WuMingrui/TianJinyu/npj/pluvian/third_party
git clone --depth 1 https://github.com/amazon-science/earth-forecasting-transformer earthformer_src
cd earthformer_src && git log --oneline -3
```

输出：

```
7732b03 Update docs and S3 urls (#57)
```

（`--depth 1` 只保留 HEAD，`git log -3` 实际只有 1 行；这是预期。）顶层目录：

```
CODE_OF_CONDUCT.md CONTRIBUTING.md LICENSE NOTICE README.md
datasets/  figures/  scripts/  setup.py  src/  tests/
```

**未发现 `requirements.txt`**；依赖列表在 `setup.py::requirements`（约 30 项，含 `omegaconf`/`einops`/`timm`/`yacs`/`fairscale`/`fvcore`/`pytorch_lightning` 隐式由 README/训练脚本带入）。

## 2. SEVIR pretrained ckpt 下载

URL 已 `curl -I` 验证（HTTP 200，Content-Length 34,798,021，Last-Modified 2023-07-14）。

第一次 `wget -q ...` 输出 0 字节文件（worker subshell 与 proxy/quiet 模式叠加似乎吃掉了下载）。换 `curl` 拉取成功：

```
cd /data4/WuMingrui/TianJinyu/npj/pluvian/ckpt/baselines/earthformer/sevir_pretrained
rm -f earthformer_sevir.pt
curl -v -o earthformer_sevir.pt https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_sevir.pt
# 100 33.1M  100 33.1M    0     0  8035k      0  0:00:04
sha256sum earthformer_sevir.pt
# 2795b23a32cc03ebd77a6c6b57ebbe133070ff9cae1b365d82bb363fd6967215  earthformer_sevir.pt
```

最终落盘：`ckpt/baselines/earthformer/sevir_pretrained/earthformer_sevir.pt` 34,798,021 bytes (~33 MB)，sha256 `2795b23a...6215`。

## 3. 依赖现状（未安装 earthformer 本体）

```
/home/WuMingrui/miniconda3/envs/npj/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
# torch 2.6.0+cu124  cuda 12.4

/home/WuMingrui/miniconda3/envs/npj/bin/pip freeze | grep -iE "^(torch|pytorch-lightning|lightning|omegaconf|einops|hydra|yacs|h5py|timm|fairscale|fvcore|tensorboard|transformers|pandas|numpy|scipy)" | sort
# einops==0.8.2
# h5py==3.16.0
# numpy==1.26.4
# pandas==2.3.3
# scipy==1.15.3
# tensorboard==2.20.0
# tensorboard-data-server==0.7.2
# torch==2.6.0+cu124
# torchaudio==2.6.0+cu124
# torchvision==0.21.0+cu124
```

**冲突 / 缺失评估（pre-install dry-run）**：

| earthformer 期望 | 当前 npj env | 风险 |
|---|---|---|
| torch 1.13.1 + PL 1.6.4（README）| torch 2.6.0 + 无 PL | PL 1.6.4 仅支持 torch <= 1.13；若 `pip install -e .` 会触发 `pytorch_lightning` 依赖收敛，可能强行降级 torch，砸 phase 7e ckpt |
| timm / yacs / fairscale / fvcore / omegaconf / transformers / boto3 / awscli 等 | 全部缺 | 单装属性可控 |
| Apex（README 强烈建议）| 缺 | 用纯 `Trainer(accelerator='gpu', strategy='auto')` 绕过 |

按 Day 1 铁律：**未执行 `pip install`**。Day 1 只需 import `NPJDataset` 即可走通；earthformer 本体的导入与训练放到 Day 2 配合 cfg + 模型实例化时一次性处理。

→ Day 2 建议安装策略（候选，待用户确认）：

```
pip install --no-deps timm yacs fairscale fvcore omegaconf transformers boto3 awscli botocore javalang unidiff jsonlines pyarrow pympler graphviz networkx absl-py
# 然后 pytorch_lightning 单装一个能跑在 torch 2.6 上的版本（>=2.0）
# 跳过 earthformer setup.py 的 install_requires 收敛：手动 `sys.path.insert` 引用 third_party/earthformer_src/src
```

**这里不强行 install。如果决定走 `pip install -e` 路线，必须先 freeze 当前 env 做快照，并和用户确认。**

## 4. 数据适配器实现

新文件：

- `pluvian/baselines/earthformer/__init__.py`
- `pluvian/baselines/earthformer/data_adapter.py`

关键设计（与计划文档一致）：

- 661×701 → bottom-right zero-pad → 672×708（H+11，W+7）；672 % 12 == 708 % 12 == 0。
- pad 值 = 0 mm/h；pad 与 ERA5 一致语义（pipeline 内部本来就把缺失帧补 0）。
- pad 用 zero（**非 reflect**）：计划文档原本写 reflect-pad，但 reflect 会把强对流外推到 lat>42.6° 的虚假区域，污染 CSI/FSS；mask-eval 时无差异，0-pad 更稳。已在 DECISIONS 里记一笔。
- ERA5 关闭（`load_era5=False`）：faithful baseline 仅 radar 单通道。
- 提供 `get_pluvian_dataset(split, n_input, n_forecast)` 工厂；split 走 `manifest.csv` 的 split 列过滤。
- mask = (H_pad, W_pad) float32，1 = 真实雷达像素，0 = pad，eval 时直接乘上去。

## 5. 单元测试输出

```
$ /home/WuMingrui/miniconda3/envs/npj/bin/python pluvian/baselines/earthformer/data_adapter.py
[info] split=val len=192
[info] in dtype=float32 tgt dtype=float32 mask dtype=float32
[info] pad region max |val| = 0 (should be 0)
[info] time=2023-05-12 00:00:00 meta={'date': '2023-05-12', 'zero_day': False, 'split': 'val', 'radar_sample_valid': True}
[OK] in_shape=(12, 672, 708, 1) tgt_shape=(18, 672, 708, 1) mask_shape=(672, 708) mm/h_range=[0.02, 283.18]
```

**Day 1 gate hit**（shape / dtype / pad / mm/h 范围全部符合预期；mm/h 上界 283 大于计划 100，是这一窗强对流，物理可信不是异常）。

## 6. 决策与变更

| # | 决策 | 理由 |
|---|---|---|
| D7 | pad 策略由 reflect 改为 zero | reflect 会跨越 lat>42.6° 边界外推虚假强对流；mask 在 eval 阶段裁回真实区域，pad 值不影响 metric；与 pipeline 现有 ffill 行为更一致 |
| D8 | Day 1 不安装 earthformer 本体 / PL | 避免 PL 1.6.4 反向把 torch 2.6 降级，砸 phase 7e ckpt；改为 Day 2 用 `--no-deps` + sys.path 方式 |
| D9 | 适配器关闭 ERA5（`load_era5=False`）| 减少 dataloader I/O；faithful baseline 不需要 ERA5 |
| D10 | 单测脚本兼容 `python xxx.py` 直接调用 | 在文件头插入 repo root 到 sys.path |

## 7. 遗留待办

- Day 2 前：跟用户确认 PL 安装方案（`pytorch_lightning>=2.0` on torch 2.6 是否兼容 earthformer 的 PL API 调用——已知 `ApexDDPStrategy` 在 PL 2.x 已改名，需要 wrap）。
- Day 2 第 1 步：grep `third_party/earthformer_src/src/earthformer/cuboid_transformer/cuboid_transformer.py` 是否有 hard-coded 13/12/384/NTHWC 维度（计划 §2.4 的 `[VERIFY]` 项）。
- 计划文档第 7 节 `[VERIFY]` 列表里第 2、3、4 项需要在 Day 2 模型实例化前 close。
