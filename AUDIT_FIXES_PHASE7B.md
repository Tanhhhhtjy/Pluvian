# Phase 7b audit fixes — applied on branch `phase7b/audit-fixes`

Resolves the 4 must-fix + 4 should-fix items from
`audit_phase7b_redteam.md` (2026-05-27).

## Commit map

| Audit item | Commit prefix | Status |
|------------|---------------|--------|
| M1 — `gated_fusion` not from config | `Phase 7b audit M1:` | Fixed |
| M2 — old ckpt strict=True crash | `Phase 7b audit M2:` | Fixed (strict=False + warn) |
| M3 — intensity loss silently shadows configured data loss | `Phase 7b audit M3:` | Fixed (fail-loud) |
| M4 — 24 starts/day cost regression | (no code fix) | **runbook** — see below |
| S1 — encoder runs 2× per val batch | `Phase 7b audit S1:` | Fixed (cached `_encode_and_fuse`) |
| S2 — CRPS spread compressed under intensity head | `Phase 7b audit S2:` | Fixed (`val/crps_method` tag) |
| S3 — bf16 softmax expectation precision | `Phase 7b audit S3:` | Fixed (fp32 cast) |
| S4 — `_oom_retry` dead code | `Phase 7b audit S4:` | Removed |

## M4 runbook reminder (no code change)

Before launching the 60-epoch `ab1-new` on AutoDL, the pre-relaunch checklist
in the audit (`audit_phase7b_redteam.md`, §"Pre-relaunch checklist") requires:

1. 50-iter dry-run with B=2, H×W=664×704, bf16 on the target GPU.
2. Compare iter-time to Phase 5 ab1 (~7 min/epoch ≈ 0.21 s/it at 2448
   samples / grad_accum=2). **>2× regression → stop, do not silently burn
   the ¥30+ budget.**
3. The user's `feedback_pretraining_speed_check.md` rule applies; sync with
   user if the smoke check trips.

## Verification snapshot

`gated_fusion` config plumbing — M1 verification command output captured
in the team-lead handover at task close.

All 55 unit tests pass on this branch (`pytest tests/ -q`).
