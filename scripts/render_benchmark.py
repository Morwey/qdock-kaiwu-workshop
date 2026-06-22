# -*- coding: utf-8 -*-
"""Render /tmp/bench.json (from scripts/benchmark.py) into BENCHMARK.md."""
import json, os, sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench.json"))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = d["rows"]


def num(x):
    return "—" if x is None else "%.2f" % x


def tick(b):
    return "✓" if b else "✗"


def by(method, gl):
    return next((r for r in rows if r["method"] == method and r["grid_length"] == gl), None)


g2, g15, f1 = by("GPM", 2.0), by("GPM", 1.5), by("FAM", 1.0)
md = []
md.append("# QDock-Kaiwu benchmark — 1y6r redocking")
md.append("Backend **%s** (Kaiwu), %s s wall, fixed seed 42. The input ligand is "
          "the crystal pose; mRMSD is the minimum heavy-atom RMSD over the sampled "
          "poses." % (d["backend"], d["wall_seconds"]))

md.append("## Default resolution")
md.append("\n".join([
    "| encoding | grid | qubits | mRMSD | < 2.0 Å | < 1.5 Å | Vina best / crystal |",
    "|---|---|---|---|---|---|---|",
    "| GPM | 2.0 Å | %d | %s Å | %s | %s | %s / %s |" % (
        g2["n_vars"], num(g2["mRMSD"]), tick(g2["success_2A"]), tick(g2["success_1p5A"]),
        num(g2.get("vina_best")), num(g2.get("vina_crystal"))),
    "| FAM | 1.0 Å | %d | %s Å | %s | %s | %s / — |" % (
        f1["n_vars"], num(f1["mRMSD"]), tick(f1["success_2A"]), tick(f1["success_1p5A"]),
        num(f1.get("vina_best")))]))

md.append("## GPM grid resolution: 2.0 → 1.5 Å (full dock)")
md.append("\n".join(
    ["| grid | qubits | discretization error | mRMSD | < 2.0 Å | < 1.5 Å |",
     "|---|---|---|---|---|---|"] +
    ["| %.1f Å | %d | %.2f Å | %s Å | %s | %s |" % (
        r["grid_length"], r["n_vars"], r["disc_error"], num(r["mRMSD"]),
        tick(r["success_2A"]), tick(r["success_1p5A"])) for r in (g2, g15)]))

md.append("## FAM grid resolution: 1.0 → 0.5 Å (discretization floor, geometric)")
md.append("\n".join(
    ["| feature spacing | feature points | discretization error |", "|---|---|---|"] +
    ["| %.1f Å | %d | %.2f Å |" % (r["grid_length"], r["n_feature_points"], r["disc_error"])
     for r in d.get("fam_resolution", [])]))

md.append("**Reading the resolution tables.** Refining each encoding on its own "
          "scale lowers the **discretization error** — the geometric floor the paper "
          "links to mRMSD (R² ≈ 0.93). Realizing that floor as a lower mRMSD needs a "
          "solver budget (reads, iterations, or a larger machine) that keeps pace with "
          "the finer grid's larger QUBO; at fixed budget the finer grid's mRMSD can be "
          "higher even though its floor is lower.")

md.append("## Published sampling power (CASF-2016, 257 cases, < 2 Å)")
md.append("\n".join(["| method | success @ 2 Å | avg. mRMSD |", "|---|---|---|",
                     "| Glide SP | 93.4 % | 1.0 Å |", "| GPM | 87.5 % | 1.1 Å |",
                     "| FAM | 67.3 % | 1.8 Å |"]))

open(os.path.join(HERE, "BENCHMARK.md"), "w").write("\n\n".join(md) + "\n")
print("wrote BENCHMARK.md")
