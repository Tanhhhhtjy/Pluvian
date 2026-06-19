"""Phase 7f band_centers sanity check.

Per judge verdict (`docs/experiments/band_debate_verdict.md`) before
launching any 5-way retrain on the new band scheme:
  band_centers        = (0, 0.5, 2.5, 10.0, 30.0, 100.0)
  band_edges          = (0.1, 1.0, 5.0, 20.0, 50.0)
  band_sample_weights = (0.1, 1.0, 2.0, 4.0, 8.0, 16.0)

Verifies:
  1. softmax × centers numerically lies in [0, 100] (no cap at 50)
  2. bucketize on synthetic targets 0.5/2/5/15/35/75 mm/h falls into
     bands 1/2/3/4/5/5 (exact band indexing, `right=False` semantics)
  3. cross-entropy with band_sample_weights produces non-zero gradient
     for the top band (band 5) when fed a target ≥ 50 mm/h
  4. length consistency: len(centers) = len(weights), len(edges) = K-1
"""
from __future__ import annotations
import sys
import torch
import torch.nn.functional as F

CENTERS = torch.tensor([0.0, 0.5, 2.5, 10.0, 30.0, 100.0])
EDGES = torch.tensor([0.1, 1.0, 5.0, 20.0, 50.0])
WEIGHTS = torch.tensor([0.1, 1.0, 2.0, 4.0, 8.0, 16.0])
K = len(CENTERS)

passed = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    passed.append(cond)
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))


print("=== Phase 7f band sanity ===")
print(f"  centers = {CENTERS.tolist()}")
print(f"  edges   = {EDGES.tolist()}")
print(f"  weights = {WEIGHTS.tolist()}\n")

print("Check 0 — length consistency")
check("len(centers) == len(weights)", len(CENTERS) == len(WEIGHTS),
      f"{len(CENTERS)} vs {len(WEIGHTS)}")
check("len(edges) == K-1", len(EDGES) == K - 1,
      f"{len(EDGES)} vs {K - 1}")

print("\nCheck 1 — softmax × centers output range")
logits_top = torch.tensor([[-50.0, -50.0, -50.0, -50.0, -50.0, 0.0]])
pred_top = (F.softmax(logits_top, dim=-1) * CENTERS.view(1, -1)).sum(dim=-1)
check("top-saturated logits → pred ≈ 100",
      abs(pred_top.item() - 100.0) < 0.01,
      f"got pred={pred_top.item():.4f}")

pred_eq = (F.softmax(torch.zeros(1, K), dim=-1) * CENTERS.view(1, -1)).sum(dim=-1)
expected_eq = CENTERS.mean().item()
check("equal logits → pred = mean(centers)",
      abs(pred_eq.item() - expected_eq) < 0.001,
      f"got pred={pred_eq.item():.4f} expected={expected_eq:.4f}")

check("upper bound > 50 (no longer capped)",
      CENTERS.max().item() > 50.0,
      f"max center = {CENTERS.max().item()}")

print("\nCheck 2 — bucketize correctness (target → band index)")
test_values = torch.tensor([0.05, 0.5, 2.0, 5.0, 15.0, 35.0, 75.0, 150.0])
expected_bands = [0, 1, 2, 3, 4, 5, 5, 5]
got = torch.bucketize(test_values, EDGES).tolist()
check("bucketize: [0.05, 0.5, 2, 5, 15, 35, 75, 150] → expected",
      got == expected_bands,
      f"got {got} expected {expected_bands}")

edge_value = torch.tensor([1.0, 5.0, 20.0, 50.0])
edge_bands = torch.bucketize(edge_value, EDGES).tolist()
check("edge values land in higher band (right=False default)",
      edge_bands == [2, 3, 4, 5],
      f"got {edge_bands} expected [2, 3, 4, 5]")

print("\nCheck 3 — top-band gradient signal")
logits = torch.zeros(1, K, requires_grad=True)
target_idx = torch.tensor([5])
ce_per_sample = F.cross_entropy(logits, target_idx, reduction='none')
weighted = ce_per_sample * WEIGHTS[target_idx]
loss = weighted.sum()
loss.backward()
grad_per_band = logits.grad[0]
check("CE grad for top band non-zero",
      grad_per_band[5].abs().item() > 1e-6,
      f"grad[band 5] = {grad_per_band[5].item():.4f}")

top_grad_mag = grad_per_band[5].abs().item()
other_grad_mag = grad_per_band[:5].abs().max().item()
check("top-band grad magnitude > other bands",
      top_grad_mag >= other_grad_mag,
      f"top={top_grad_mag:.4f} max_other={other_grad_mag:.4f}")

loss_unweighted = F.cross_entropy(torch.zeros(1, K), target_idx)
weighted_factor = loss.item() / loss_unweighted.item()
expected_factor = WEIGHTS[5].item()
check("top-band weight × 16 amplifies loss",
      abs(weighted_factor - expected_factor) < 0.01,
      f"factor={weighted_factor:.4f} expected≈{expected_factor:.4f}")

print("\n" + "=" * 60)
n_pass, n_total = sum(passed), len(passed)
if n_pass == n_total:
    print(f"ALL {n_pass}/{n_total} CHECKS PASS — Phase 7f config is safe to launch")
    sys.exit(0)
else:
    print(f"FAILED {n_total - n_pass}/{n_total} CHECKS — fix before launching retrain")
    sys.exit(1)
