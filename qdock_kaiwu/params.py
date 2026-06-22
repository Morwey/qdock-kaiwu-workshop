# -*- coding: utf-8 -*-
"""Recommended parameters for the two QDock encodings.

QUBO Lagrange multipliers (edge_cutoff = c_dist, K_dist, K_mono) are from Zha
et al., JCTC 2024 (fitted on the Astex Diversity Set). Each method has its own
docking-box grid spacing and its own finer setting for the resolution study
(GPM 2.0 -> 1.5 A; FAM 1.0 -> 0.5 A): the two are never put on the same scale,
since their default resolutions differ by 2x.

The ``sa_*`` fields are the Kaiwu simulated-annealing schedule. The initial
temperature is matched to each QUBO's energy scale (see backends.py / README).
The dense docking QUBO has high read-to-read variance, so results are reported
at a fixed ``seed`` with many reads.
"""

GPM = dict(
    grid_length=2.0,        # default grid spacing (A)
    grid_length_fine=1.5,   # finer grid for the resolution study
    cutoff=0.0,             # keep grid points with vdW energy < cutoff
    edge_cutoff=2.162,      # c_dist: geometry tolerance
    K_dist=0.405,           # distance (shape) penalty weight
    K_mono=25.090,          # monogamy (uniqueness) penalty weight
    # Kaiwu SA schedule
    sa_initial_temperature=10.0,
    sa_alpha=0.999,
    sa_cutoff_temperature=0.01,
    sa_iterations_per_t=2000,
    n_pos=300,
    seed=42,
)

FAM = dict(
    grid_length=1.0,        # default feature-point spacing (A)
    grid_length_fine=0.5,   # finer spacing for the resolution study
    edge_cutoff=1.870,
    K_dist=2.261,
    K_mono=11.479,
    max_features=60,        # cap on feature atoms at the default spacing
    sa_initial_temperature=15.0,
    sa_alpha=0.999,
    sa_cutoff_temperature=0.01,
    sa_iterations_per_t=2000,
    n_pos=300,
    seed=42,
)


def sa_schedule(p):
    """Extract the Kaiwu SA keyword arguments from a GPM/FAM parameter dict."""
    return dict(initial_temperature=p["sa_initial_temperature"],
                alpha=p["sa_alpha"], cutoff_temperature=p["sa_cutoff_temperature"],
                iterations_per_t=p["sa_iterations_per_t"])
