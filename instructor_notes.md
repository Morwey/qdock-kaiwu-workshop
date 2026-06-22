# Instructor notes

A ~45-minute session built around `notebooks/qdock_kaiwu_workshop.ipynb`.

## Arc

1. **Docking is sampling + scoring.** Sampling the binding geometry is the
   NP-hard part — the natural fit for an Ising machine. Scoring is left to Vina.
2. **A QUBO on Kaiwu (§1).** A 4-variable toy QUBO → Ising → solved on SA and on
   the real CIM, both hitting the brute-force minimum. Make the sign convention
   explicit: feed `qubo_matrix_to_ising_matrix(Q)` straight to `solve`.
3. **The encoding (§2).** One variable per atom↔site match; rewards + two
   penalties. GPM sites = vdW grid points, FAM sites = typed pocket atoms.
4. **GPM 3f3d (§3).** The CIM is 8-bit, so the QUBO goes through a
   `PrecisionReducer`. Sweep `truncated_precision` 8→10→12: the docking improves
   to 1.25 Å as more of the vdW reward survives quantization. This is the core
   teaching point — **precision is a resource you spend, bounded by the spin budget.**
5. **FAM 3d4z (§4).** Same sweep on a hydrogen-bonded ligand (an iminosugar).
   Read the best pose out as polar contacts: it recovers all five of the
   crystal's hydrogen bonds. FAM *is* the H-bond picture.

## Points to land

- `precision=8` is the hardware; `truncated_precision=t > 8` splits each variable
  into several spins to keep `t` bits. More bits → better, until the expanded
  matrix would exceed ~1000 spins. Both demos are sized to stay under it (GPM is
  small at 2.0 Å; FAM is capped at 24 features).
- The expansion is matrix-dependent, not a fixed multiple — measure it, don't
  extrapolate.
- The CIM returns only ~6–10 distinct poses, so it is sampling-limited: a strong
  classical annealer with the same precision but many reads is a useful contrast.

## Setup checklist

- Python 3.10 venv + the vendored Kaiwu wheel; `conda` `chem` env for
  `autogrid` / `vina` / `obabel`.
- Each participant exports their own `KAIWU_USER_ID` / `KAIWU_SDK_CODE`.
- `cim_cache/` reproduces the reference numbers offline; delete a file to submit a
  fresh hardware job.
