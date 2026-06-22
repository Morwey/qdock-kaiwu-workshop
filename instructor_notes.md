# Instructor notes

A ~45-minute session built around `notebooks/qdock_kaiwu_workshop.ipynb`.

## Arc

1. **Docking is sampling + scoring.** Sampling the binding geometry is the
   NP-hard part — the natural fit for an Ising machine. Scoring is left to Vina.
2. **A QUBO on Kaiwu (§1).** A 4-variable toy QUBO → Ising → solved on SA and on
   the real CIM, both hitting the brute-force minimum. Make the sign convention
   explicit: feed `qubo_matrix_to_ising_matrix(Q)` straight to `solve`.
3. **The encoding (§2).** One variable per atom↔site match; rewards + two
   penalties. GPM sites = vdW grid points, FAM sites = typed pocket atoms. The CIM
   is 8-bit, so the QUBO goes through a `PrecisionReducer`.
4. **GPM 3f3d (§3) — the central lesson.** The CIM returns ~6–10 poses per run and
   is high-variance on these dense QUBOs: per-run best RMSD swings ~1.2–5 Å. Plot
   the five runs, then pool them — the native pose (1.25 Å) is reliably in the
   pool. The bottleneck is **sample count, not precision**: this is the
   sampling-limited regime, and pooling is the fix.
5. **FAM 3d4z (§4).** Same pool on a hydrogen-bonded ligand (an iminosugar). Read
   the pooled-best pose out as polar contacts: it recovers all five of the
   crystal's hydrogen bonds. FAM *is* the H-bond picture.

## Points to land

- `precision=8` is the hardware; `truncated_precision=t>8` splits each variable
  into several spins to keep `t` bits. Keep the QUBO small (coarse grid / capped
  features) so the expanded matrix stays under ~1000 spins.
- The CIM is a stochastic sampler — **one run is a draw, not the answer.** Quote
  the per-run spread, not a single lucky number; pool runs (or raise the sample
  budget) to find the native pose.
- A strong classical annealer with many reads is the useful contrast: same
  precision, far more samples, so it is not sampling-limited.

## Setup checklist

- Python 3.10 venv + the vendored Kaiwu wheel (macOS arm64; swap for your
  platform); `conda` `chem` env for `autogrid` / `vina` / `obabel`.
- Each participant exports their own `KAIWU_USER_ID` / `KAIWU_SDK_CODE`.
- `cim_cache/` holds five runs per demo, so the notebook reproduces the spread and
  the pooled-best offline; change a `task_name` to submit a fresh hardware job.
- `notebooks/qdock_kaiwu_colab.ipynb` reproduces poses, RMSDs, H-bonds and the 3D
  figure with NumPy + Matplotlib only (the Kaiwu wheel is macOS/Windows).
