"""Phase 7d Tier-2: train the DiffCast-style residual diffusion head.

The Pluvian backbone (init from the converged p7c ab2 ckpt) is FROZEN and only
ever run under no_grad to produce the deterministic mean mu. We learn a
conditional DDPM over the per-frame residual (target - mu), conditioned on mu.
At inference a fresh residual is sampled (DDIM) and added back -> sharp cores
that the MSE mean averages out, storm still located by mu.

    python scripts/train_diffusion.py \
        --config configs/ablation_2_new_7b_p7c.yaml \
        --init-from ckpt/ablation_2_new_7b_p7c/best.pt \
        --epochs 60 --base 64 --timesteps 1000 --out ckpt/diff_ab2_p7d
"""
import argparse
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.pluvian import Pluvian
from model.diffusion_head import CondEpsUNet, GaussianDiffusion, pad_to_multiple
from scripts.train import (
    load_config, build_datasets, _collate, split_batch, _to_device,
)

RES_SCALE = 20.0   # residual dBZ -> ~unit
MU_SCALE = 50.0    # condition mu dBZ -> ~unit


def build_backbone(cfg, device):
    m = cfg["model"]
    return Pluvian(
        radar_channels=m["radar_channels"], pwv_enabled=m["pwv_enabled"],
        pwv_concat_only=m["pwv_concat_only"], era5_enabled=m["era5_enabled"],
        mfd_channel_enabled=m["mfd_channel_enabled"],
        forecast_frames=m["forecast_frames"], input_frames=m["input_frames"],
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], n_heads=m["n_heads"],
        n_era5_vars=m["n_era5_vars"], n_era5_levels=m["n_era5_levels"],
        mfd_channels=m["mfd_channels"],
        intensity_stratified=bool(m.get("intensity_stratified", False)),
        band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
        gated_fusion=bool(m.get("gated_fusion", False)),
    ).to(device)


@torch.no_grad()
def backbone_mu(model, model_in, rain_tgt):
    fused, skips = model._encode_and_fuse(model_in, return_skips=True)
    dec = model.decoder(fused, encoder_skips=skips)
    rain_pred = dec[0] if isinstance(dec, (tuple, list)) else dec
    Ht, Wt = rain_tgt.shape[-2:]
    if rain_pred.shape[-2:] != (Ht, Wt):
        rain_pred = rain_pred[..., :Ht, :Wt]
    return rain_pred.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--init-from", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--frames-per-step", type=int, default=4,
                    help="random subset of the 18 lead frames per step (661x701 is too "
                         "large to denoise all 18 at once)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg["hardware"]["precision"]]
    T_in = cfg["model"]["input_frames"]
    T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"]["mfd_channel_enabled"] else 0

    train_ds, _ = build_datasets(cfg)
    loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"]["num_workers"], collate_fn=_collate,
                        pin_memory=(device.type == "cuda"), drop_last=True)
    print(f"[data] train windows: {len(train_ds)}")

    backbone = build_backbone(cfg, device)
    state = torch.load(args.init_from, map_location=device)
    res = backbone.load_state_dict(state["model"], strict=False)
    print(f"[init-from] {args.init_from} ({len(res.missing_keys)} missing, "
          f"{len(res.unexpected_keys)} unexpected)")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    net = CondEpsUNet(base=args.base).to(device)
    diff = GaussianDiffusion(net, timesteps=args.timesteps).to(device)
    n_p = sum(p.numel() for p in net.parameters())
    print(f"[denoiser] {n_p/1e6:.3f}M  T={args.timesteps}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))

    args.out.mkdir(parents=True, exist_ok=True)
    steps = max(1, len(loader))
    for epoch in range(args.epochs):
        run = {"loss": 0.0, "n": 0}
        t0 = time.time()
        for it, batch in enumerate(loader):
            batch = _to_device(batch, device)
            model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
            mu = backbone_mu(backbone, model_in, rain_tgt)            # (B,T,H,W) dBZ
            resid = (rain_tgt - mu) / RES_SCALE
            cond = (mu / MU_SCALE)
            B, T = resid.shape[:2]
            k = min(args.frames_per_step, T)
            fidx = torch.randperm(T, device=device)[:k]
            resid = resid[:, fidx]
            cond = cond[:, fidx]
            x0 = resid.reshape(B * k, 1, *resid.shape[-2:])
            c = cond.reshape(B * k, 1, *cond.shape[-2:])
            x0, hw = pad_to_multiple(x0, 8)
            c, _ = pad_to_multiple(c, 8)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                loss = diff.p_losses(x0.float(), c.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run["loss"] += loss.item() * B
            run["n"] += B
            if it % 40 == 0:
                print(f"[ep{epoch:03d}|it{it:04d}/{steps}] mse={loss.item():.4f}", flush=True)
            if args.debug and it >= 4:
                break
        n = max(1, run["n"])
        print(f"[ep{epoch:03d}] avg mse={run['loss']/n:.5f}  ({time.time()-t0:.1f}s)", flush=True)
        torch.save({"net": net.state_dict(), "base": args.base,
                    "timesteps": args.timesteps, "res_scale": RES_SCALE,
                    "mu_scale": MU_SCALE, "epoch": epoch, "args": vars(args)},
                   args.out / "last.pt")
        if args.debug:
            break
    print(f"[done] diffusion ckpt in {args.out}")


if __name__ == "__main__":
    main()
