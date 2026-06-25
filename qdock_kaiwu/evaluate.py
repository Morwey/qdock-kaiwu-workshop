# -*- coding: utf-8 -*-
"""Docking-power metrics: RMSD to the crystal pose, success rates, and the
discretization error that bounds achievable accuracy.

Redocking convention: the input ligand IS the crystal pose, so the prepared
ligand's own coordinates are the reference. mRMSD = minimum heavy-atom RMSD over
the sampled poses (the "sampling power"); success = mRMSD < threshold
(2.0 A standard, 1.5 A strict).
"""

import numpy as np
from .geometry import heavy_atom_rmsd, dist_matrix


def pose_rmsds(poses, crystal_coords, elements):
    return np.array([heavy_atom_rmsd(p, crystal_coords, elements) for p in poses])


def mrmsd(poses, crystal_coords, elements):
    if len(poses) == 0:
        return np.inf
    return float(pose_rmsds(poses, crystal_coords, elements).min())


def discretization_error(lig_coords, site_coords, elements):
    """Mean distance from each heavy ligand atom to the nearest docking-box site.
    A ligand atom cannot match closer than its nearest site, so this is the
    geometric floor on accuracy; a finer grid lowers it (Zha et al. tie it to
    mRMSD with R^2 ~ 0.93). Solver-independent."""
    elements = np.asarray([str(e).strip().upper() for e in elements])
    keep = elements != "H"
    lc = np.asarray(lig_coords)[keep] if keep.any() else np.asarray(lig_coords)
    d = dist_matrix(np.vstack([lc, site_coords]))[:len(lc), len(lc):]
    return float(d.min(1).mean())


def evaluate_case(name, method, poses, crystal_coords, elements, vina_scores=None):
    r = pose_rmsds(poses, crystal_coords, elements) if len(poses) else np.array([])
    row = dict(case=name, method=method, n_poses=len(poses),
               mRMSD=float(r.min()) if len(r) else np.inf,
               success_2A=bool(len(r) and r.min() < 2.0),
               success_1p5A=bool(len(r) and r.min() < 1.5))
    if vina_scores is not None and len(vina_scores):
        row["vina_best"] = float(np.nanmin(vina_scores))
        if len(r):
            row["vina_at_best_rmsd"] = float(vina_scores[int(r.argmin())])
    return row
