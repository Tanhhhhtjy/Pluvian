"""Phase 7d Tier-1: DGMR-lite adversarial fine-tune to grow intense echo cores.

The frozen-backbone + global-FiLM latent head collapsed (samples == mean): a
per-channel affine cannot inject spatial structure. A spatial discriminator
fixes that by shaping the *output field* directly. Here:

  * generator = frozen radar/PWV encoder + fusion (no_grad) -> TRAINABLE decoder
    (init from the converged p7c ab2 ckpt). The adversarial gradient reaches the
    decoder's conv/upsample stack, which CAN place sharp local cells.
  * a small spatial PatchGAN (pix2pix NLayerDiscriminator) judges per-frame
    realism; hinge loss + feature-matching (pix2pixHD) for stability.
  * reconstruction (intensity) kept at moderate weight so storms stay located;
    adversarial weight light + ramped so D grows texture/cores without
    hallucinating speckle.

    python scripts/train_gan_ft.py \
        --config configs/ablation_2_new_7b_p7c.yaml \
        --init-from ckpt/ablation_2_new_7b_p7c/best.pt \
        --epochs 30 --lambda-adv 0.05 --lambda-fm 10 --out ckpt/gan_ft_ab2_p7d
"""
import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.pluvian import Pluvian
from model.losses import SpectralLoss
from scripts.train import (
    load_config, build_datasets, _collate, split_batch, _intensity_loss,
    _to_device,
)

DBZ_SCALE = 50.0  # normalise dBZ (~ -5..70) to ~[-0.1, 1.4] for the discriminator


def coverage_loss(pred, tgt, thresholds, temp=2.0):
    """Intensity-tail matching: the base GAN recovers texture but is timid about
    extreme cores (caps ~49 dBZ vs GT 60+). Two complementary terms, per frame:

      * soft-peak — a smooth (logsumexp) surrogate for the per-frame maximum
        intensity, L1-matched to GT. O(dBZ) magnitude, so it directly drags the
        predicted core intensity up toward the truth.
      * coverage — the fraction of pixels exceeding each dBZ threshold (smooth
        sigmoid surrogate), so the *amount* of intense echo matches, not just
        the single peak.

    pred,tgt: (B,T,H,W) dBZ. Returns a scalar."""
    B, T = pred.shape[:2]
    pf = pred.reshape(B, T, -1).float()
    tf = tgt.reshape(B, T, -1).float()
    soft_peak = F.l1_loss(torch.logsumexp(pf, dim=-1), torch.logsumexp(tf, dim=-1))
    cov = pred.new_zeros(())
    for thr in thresholds:
        cp = torch.sigmoid((pred - thr) / temp).mean(dim=(-1, -2))  # (B,T)
        ct = torch.sigmoid((tgt - thr) / temp).mean(dim=(-1, -2))
        cov = cov + F.l1_loss(cp, ct)
    return soft_peak + cov / max(1, len(thresholds))


class PatchDisc(nn.Module):
    """pix2pix NLayerDiscriminator (spatial PatchGAN). Returns (logits, feats)."""

    def __init__(self, in_ch: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        blocks = [nn.Sequential(nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True))]
        nf = ndf
        for i in range(1, n_layers):
            nf_prev, nf = nf, min(ndf * 2 ** i, 512)
            blocks.append(nn.Sequential(
                nn.Conv2d(nf_prev, nf, 4, 2, 1),
                nn.GroupNorm(min(8, nf), nf), nn.LeakyReLU(0.2, True)))
        nf_prev, nf = nf, min(ndf * 2 ** n_layers, 512)
        blocks.append(nn.Sequential(
            nn.Conv2d(nf_prev, nf, 4, 1, 1),
            nn.GroupNorm(min(8, nf), nf), nn.LeakyReLU(0.2, True)))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(nf, 1, 4, 1, 1)

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(x)
        return self.out(x), feats


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--init-from", required=True, type=Path)
    ap.add_argument("--init-disc", type=Path, default=None,
                    help="warm-start the discriminator from a prior gan ckpt's 'disc'")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--lambda-rec", type=float, default=1.0)
    ap.add_argument("--lambda-adv", type=float, default=0.05)
    ap.add_argument("--lambda-fm", type=float, default=10.0)
    ap.add_argument("--lambda-spec", type=float, default=0.0,
                    help="power-spectrum matching weight (restores high-k energy, no speckle)")
    ap.add_argument("--lambda-core", type=float, default=0.0,
                    help="intensity-tail coverage weight (grows the extreme echo cores)")
    ap.add_argument("--core-thresholds", type=str, default="35,45,55",
                    help="dBZ thresholds for the coverage loss")
    ap.add_argument("--band-map", type=str, default=None,
                    help="comma indices to expand the old rain_head into the new "
                         "band layout, e.g. '0,1,2,3,4,4' duplicates the old top "
                         "band so a 6-band (hi-core) head warm-starts from a 5-band ckpt")
    ap.add_argument("--adv-warmup-epochs", type=int, default=3)
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

    model = build_backbone(cfg, device)
    state = torch.load(args.init_from, map_location=device, weights_only=False)
    sd = dict(state["model"])
    msd = model.state_dict()
    # Hi-core warm-start: the new head has more bands than the ckpt. Re-map the
    # old rain_head conv (and its band_centers buffer slots) onto the new layout
    # by duplicating bands per --band-map, so we don't cold-start a random head.
    hk = "decoder.rain_head.weight"
    if args.band_map and hk in sd and sd[hk].shape != msd[hk].shape:
        idx = [int(i) for i in args.band_map.split(",")]
        sd["decoder.rain_head.weight"] = sd[hk][idx].clone()
        sd["decoder.rain_head.bias"] = sd["decoder.rain_head.bias"][idx].clone()
        print(f"[band-map] expanded rain_head {tuple(state['model'][hk].shape)} -> "
              f"{tuple(sd['decoder.rain_head.weight'].shape)} via {idx}")
    # drop any still-mismatched tensors (e.g. band_centers buffer) -> keep fresh
    for k in [k for k in list(sd) if k in msd and sd[k].shape != msd[k].shape]:
        sd.pop(k)
    res = model.load_state_dict(sd, strict=False)
    print(f"[init-from] {args.init_from} ({len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected)")
    # freeze encoder + fusion; train only the decoder (the spatial generator)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.decoder.train()

    disc = PatchDisc(in_ch=1, ndf=64, n_layers=3).to(device)
    if args.init_disc is not None:
        dstate = torch.load(args.init_disc, map_location=device, weights_only=False)
        if "disc" in dstate:
            disc.load_state_dict(dstate["disc"])
            print(f"[init-disc] warm-started disc from {args.init_disc}")
    disc.train()
    n_g = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    n_d = sum(p.numel() for p in disc.parameters())
    print(f"[gen=decoder] {n_g/1e6:.3f}M  [disc] {n_d/1e6:.3f}M")

    spec = SpectralLoss(highpass=1.0).to(device) if args.lambda_spec > 0 else None
    core_thr = [float(x) for x in args.core_thresholds.split(",") if x.strip()]
    opt_g = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr_g, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.9))

    args.out.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, len(loader))
    adv_warmup = args.adv_warmup_epochs * steps_per_epoch
    gstep = 0
    for epoch in range(args.epochs):
        run = {"rec": 0.0, "adv": 0.0, "fm": 0.0, "d": 0.0, "spec": 0.0, "core": 0.0, "n": 0}
        t0 = time.time()
        for it, batch in enumerate(loader):
            batch = _to_device(batch, device)
            model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
            with torch.no_grad():
                fused, skips = model._encode_and_fuse(model_in, return_skips=True)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                dec = model.decoder(fused, encoder_skips=skips)
                rain_pred, _pwv, rain_logits = (dec if len(dec) == 3 else (dec[0], dec[1], None))
                if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
                    Ht, Wt = rain_tgt.shape[-2:]
                    rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
                    if rain_logits is not None:
                        rain_logits = rain_logits[..., :Ht, :Wt].contiguous()
                if intensity_on and rain_logits is not None:
                    loss_rec, _ce, _reg = _intensity_loss(rain_logits, rain_pred, rain_tgt, cfg)
                else:
                    loss_rec = ((rain_pred - rain_tgt) ** 2).mean()

            B, T = rain_pred.shape[:2]
            fake = (rain_pred.reshape(B * T, 1, *rain_pred.shape[-2:]) / DBZ_SCALE).float()
            real = (rain_tgt.reshape(B * T, 1, *rain_tgt.shape[-2:]) / DBZ_SCALE).float()
            adv_w = args.lambda_adv * min(1.0, gstep / max(1, adv_warmup))

            # ---- discriminator step (hinge) ----
            d_logit_real, _ = disc(real)
            d_logit_fake, _ = disc(fake.detach())
            loss_d = F.relu(1.0 - d_logit_real).mean() + F.relu(1.0 + d_logit_fake).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- generator step ----
            g_logit_fake, feats_fake = disc(fake)
            with torch.no_grad():
                _, feats_real = disc(real)
            loss_adv = -g_logit_fake.mean()
            loss_fm = sum(F.l1_loss(ff.float(), fr.float()) for ff, fr in zip(feats_fake, feats_real)) / len(feats_fake)
            loss_spec = spec(rain_pred, rain_tgt) if spec is not None else rain_pred.new_zeros(())
            loss_core = coverage_loss(rain_pred, rain_tgt, core_thr) if args.lambda_core > 0 else rain_pred.new_zeros(())
            loss_g = (args.lambda_rec * loss_rec + adv_w * loss_adv + args.lambda_fm * loss_fm
                      + args.lambda_spec * loss_spec + args.lambda_core * loss_core)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            opt_g.step()
            gstep += 1

            bs = B
            run["rec"] += loss_rec.item() * bs; run["adv"] += loss_adv.item() * bs
            run["fm"] += loss_fm.item() * bs; run["d"] += loss_d.item() * bs
            run["spec"] += float(loss_spec) * bs; run["core"] += float(loss_core) * bs
            run["n"] += bs
            if it % 40 == 0:
                print(f"[ep{epoch:03d}|it{it:04d}/{steps_per_epoch}] rec={loss_rec.item():.3f} "
                      f"adv={loss_adv.item():.3f} fm={loss_fm.item():.3f} D={loss_d.item():.3f} "
                      f"spec={float(loss_spec):.3f} core={float(loss_core):.4f} advw={adv_w:.1e}", flush=True)
            if args.debug and it >= 4:
                break
        n = max(1, run["n"])
        print(f"[ep{epoch:03d}] avg rec={run['rec']/n:.4f} adv={run['adv']/n:.4f} "
              f"fm={run['fm']/n:.4f} D={run['d']/n:.4f} spec={run['spec']/n:.4f} "
              f"core={run['core']/n:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        torch.save({"model": model.state_dict(), "disc": disc.state_dict(),
                    "epoch": epoch, "args": vars(args)}, args.out / "last.pt")
        if args.debug:
            break
    print(f"[done] gan ckpt in {args.out}")


if __name__ == "__main__":
    main()
