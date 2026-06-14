"""Train the Phase 7d Tier-1 stochastic latent head (CVAE) on a frozen backbone.

    python scripts/train_latent_head.py \
        --config configs/ablation_2_new_7b_p7c.yaml \
        --init-from ckpt/ablation_2_new_7b_p7c/best.pt \
        --epochs 25 --z-dim 64 --beta-kl 1e-3 --spectral 5.0 \
        --out ckpt/latent_head_ab2_p7d

Only the head (posterior encoder + FiLM generator) is updated; the Pluvian
encoder / fusion / decoder are frozen. Gradients flow through the frozen
decoder to reach the FiLM params but update nothing in the backbone.
"""
import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.pluvian import Pluvian
from model.latent_head import LatentSharpHead
from model.losses import SpectralLoss
from scripts.train import (
    load_config, build_datasets, _collate, split_batch, _intensity_loss,
    _to_device,
)


def build_backbone(cfg, device):
    m = cfg["model"]
    model = Pluvian(
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
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--init-from", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta-kl", type=float, default=1e-3)
    ap.add_argument("--kl-warmup-epochs", type=int, default=5)
    ap.add_argument("--spectral", type=float, default=5.0, help="spectral loss weight (0=off)")
    ap.add_argument("--spectral-highpass", type=float, default=1.0)
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
    intensity_on = bool(cfg["model"].get("intensity_stratified", False))

    train_ds, _ = build_datasets(cfg)
    loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"]["num_workers"], collate_fn=_collate,
                        pin_memory=(device.type == "cuda"), drop_last=True)
    print(f"[data] train windows: {len(train_ds)}")

    # frozen backbone
    model = build_backbone(cfg, device)
    state = torch.load(args.init_from, map_location=device)
    res = model.load_state_dict(state["model"], strict=False)
    print(f"[init-from] {args.init_from}  ({len(res.missing_keys)} missing, "
          f"{len(res.unexpected_keys)} unexpected)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    head = LatentSharpHead(n_ch=cfg["model"]["hidden_dim"], z_dim=args.z_dim,
                           t_out=T_out).to(device)
    head.train()
    n_head = sum(p.numel() for p in head.parameters())
    print(f"[head] params: {n_head/1e6:.3f}M  z_dim={args.z_dim}")

    spec = SpectralLoss(highpass=args.spectral_highpass).to(device) if args.spectral > 0 else None
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))

    args.out.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, len(loader))
    kl_warmup = args.kl_warmup_epochs * steps_per_epoch
    gstep = 0
    for epoch in range(args.epochs):
        run = {"data": 0.0, "spec": 0.0, "kl": 0.0, "n": 0}
        t0 = time.time()
        for it, batch in enumerate(loader):
            batch = _to_device(batch, device)
            model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                with torch.no_grad():
                    fused, skips = model._encode_and_fuse(model_in, return_skips=True)
                z, kl = head.sample_posterior(rain_tgt.float())
                fused2 = head.apply_film(fused, z)
                dec = model.decoder(fused2, encoder_skips=skips)
                if len(dec) == 3:
                    rain_pred, _pwv, rain_logits = dec
                else:
                    rain_pred, _pwv = dec
                    rain_logits = None
                if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
                    Ht, Wt = rain_tgt.shape[-2:]
                    rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
                    if rain_logits is not None:
                        rain_logits = rain_logits[..., :Ht, :Wt].contiguous()
                if intensity_on and rain_logits is not None:
                    loss_data, _ce, _reg = _intensity_loss(rain_logits, rain_pred, rain_tgt, cfg)
                else:
                    loss_data = ((rain_pred - rain_tgt) ** 2).mean()
                loss_data = loss_data * cfg["loss"]["data"]["weight"]
                loss_spec = spec(rain_pred, rain_tgt) * args.spectral if spec is not None else torch.zeros((), device=device)
                beta = args.beta_kl * min(1.0, gstep / max(1, kl_warmup))
                loss = loss_data + loss_spec + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            gstep += 1
            bs = rain_pred.shape[0]
            run["data"] += loss_data.item() * bs
            run["spec"] += float(loss_spec) * bs
            run["kl"] += kl.item() * bs
            run["n"] += bs
            if it % 40 == 0:
                print(f"[ep{epoch:03d}|it{it:04d}/{steps_per_epoch}] "
                      f"data={loss_data.item():.3f} spec={float(loss_spec):.3f} "
                      f"kl={kl.item():.3f} beta={beta:.1e}", flush=True)
            if args.debug and it >= 4:
                break
        n = max(1, run["n"])
        print(f"[ep{epoch:03d}] avg data={run['data']/n:.4f} spec={run['spec']/n:.4f} "
              f"kl={run['kl']/n:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        torch.save({"head": head.state_dict(), "z_dim": args.z_dim,
                    "epoch": epoch, "args": vars(args)}, args.out / "last.pt")
        if args.debug:
            break
    print(f"[done] head ckpt in {args.out}")


if __name__ == "__main__":
    main()
