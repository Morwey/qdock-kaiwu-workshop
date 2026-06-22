# -*- coding: utf-8 -*-
"""Rigid-body geometry: Kabsch superposition, RMSD, distance matrices."""

import numpy as np


def dist_matrix(coords):
    """Pairwise Euclidean distance matrix of an (N,3) array."""
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def kabsch(mobile, target):
    """Optimal rotation R and translation t mapping ``mobile`` onto ``target``
    (minimum RMSD, no scaling). Returns (R, t) with ``mobile @ R.T + t`` aligned."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    cm, ct = mobile.mean(0), target.mean(0)
    H = (mobile - cm).T @ (target - ct)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, ct - R @ cm


def superpose_apply(match_mobile, match_target, coords):
    """Fit the matched pairs (``match_mobile`` -> ``match_target``) and apply the
    resulting transform to a full coordinate set. This turns a QUBO matching
    into a docked pose."""
    R, t = kabsch(match_mobile, match_target)
    return np.asarray(coords, dtype=float) @ R.T + t


def rmsd(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def heavy_atom_rmsd(a, b, elements):
    """RMSD over non-hydrogen atoms (the docking-power metric)."""
    elements = np.asarray([str(e).strip().upper() for e in elements])
    keep = elements != "H"
    if not keep.any():
        keep = np.ones(len(elements), dtype=bool)
    return rmsd(np.asarray(a)[keep], np.asarray(b)[keep])
