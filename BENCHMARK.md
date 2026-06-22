# QDock-Kaiwu benchmark — 1y6r redocking

Backend **sa** (Kaiwu), 186.4 s wall, fixed seed 42. The input ligand is the crystal pose; mRMSD is the minimum heavy-atom RMSD over the sampled poses.

## Default resolution

| encoding | grid | qubits | mRMSD | < 2.0 Å | < 1.5 Å | Vina best / crystal |
|---|---|---|---|---|---|---|
| GPM | 2.0 Å | 1128 | 0.15 Å | ✓ | ✓ | -11.35 / -11.80 |
| FAM | 1.0 Å | 1620 | 1.46 Å | ✓ | ✓ | 13.16 / — |

## GPM grid resolution: 2.0 → 1.5 Å (full dock)

| grid | qubits | discretization error | mRMSD | < 2.0 Å | < 1.5 Å |
|---|---|---|---|---|---|
| 2.0 Å | 1128 | 0.98 Å | 0.15 Å | ✓ | ✓ |
| 1.5 Å | 3827 | 0.72 Å | 2.41 Å | ✗ | ✗ |

## FAM grid resolution: 1.0 → 0.5 Å (discretization floor, geometric)

| feature spacing | feature points | discretization error |
|---|---|---|
| 1.0 Å | 408 | 0.43 Å |
| 0.5 Å | 3101 | 0.23 Å |

**Reading the resolution tables.** Refining each encoding on its own scale lowers the **discretization error** — the geometric floor the paper links to mRMSD (R² ≈ 0.93). Realizing that floor as a lower mRMSD needs a solver budget (reads, iterations, or a larger machine) that keeps pace with the finer grid's larger QUBO; at fixed budget the finer grid's mRMSD can be higher even though its floor is lower.

## Published sampling power (CASF-2016, 257 cases, < 2 Å)

| method | success @ 2 Å | avg. mRMSD |
|---|---|---|
| Glide SP | 93.4 % | 1.0 Å |
| GPM | 87.5 % | 1.1 Å |
| FAM | 67.3 % | 1.8 Å |
