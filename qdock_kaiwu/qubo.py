# -*- coding: utf-8 -*-
"""Native NumPy construction of the GPM and FAM QUBO matrices.

The QUBO minimised is

    H(x) = sum_v  w_v x_v
         + K_dist * sum_{p<q} [ |d_lig(p,q) - d_site(p,q)| > c ] x_p x_q
         + K_mono * sum_{p<q} [ atom(p) == atom(q) ]              x_p x_q

with x_(a,s) = 1 meaning "ligand atom a is placed at site s". The linear term
rewards a chemically favourable placement; K_dist preserves the ligand's rigid
shape; K_mono forbids one atom from occupying two sites (Eqs. 4-5 of Zha et al.,
JCTC 2024). The matrix is assembled directly in NumPy and handed to the Kaiwu
solver.
"""

import numpy as np
from .geometry import dist_matrix

# Pauling electronegativities for Feature Atom Matching.
EN = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Si': 1.9,
      'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'As': 2.18, 'Se': 2.48,
      'Br': 2.96, 'I': 2.66, 'B': 2.04}


def _assemble(weights, atom_idx, site_idx, lig_dmat, site_dmat,
              edge_cutoff, K_dist, K_mono, max_pairs=200_000_000):
    weights = np.asarray(weights, dtype=float)
    V = len(weights)
    if V == 0:
        return np.zeros((0, 0))
    if V * (V - 1) // 2 > max_pairs:
        raise MemoryError("QUBO too large: %d variables. Shrink the box, raise "
                          "the energy cutoff, or coarsen the grid." % V)
    atom_idx = np.asarray(atom_idx)
    site_idx = np.asarray(site_idx)
    Q = np.zeros((V, V), dtype=float)
    Q[np.arange(V), np.arange(V)] = weights
    dd = np.abs(lig_dmat[np.ix_(atom_idx, atom_idx)] -
                site_dmat[np.ix_(site_idx, site_idx)])
    pen = (dd > edge_cutoff).astype(float) * K_dist
    pen += (atom_idx[:, None] == atom_idx[None, :]).astype(float) * K_mono
    iu = np.triu_indices(V, k=1)
    Q[iu] += pen[iu]
    return Q


def build_gpm_qubo(lig_coords, lig_ad_types, grid_dict, box_coords,
                   edge_cutoff, K_dist, K_mono):
    """Grid Point Matching QUBO. grid_dict[ad_type] = (positions, vdW_energies);
    positions index into box_coords. Returns (Q, variables) with
    variables[k] = (ligand_atom_index, box_point_index)."""
    lig_dmat = dist_matrix(lig_coords)
    box_dmat = dist_matrix(box_coords)
    weights, atom_idx, site_idx = [], [], []
    for i, t in enumerate(lig_ad_types):
        if t not in grid_dict:
            continue
        for pos, e in zip(*grid_dict[t]):
            weights.append(float(e)); atom_idx.append(i); site_idx.append(int(pos))
    Q = _assemble(weights, atom_idx, site_idx, lig_dmat, box_dmat,
                  edge_cutoff, K_dist, K_mono)
    return Q, list(zip(atom_idx, site_idx))


def build_fam_qubo(lig_coords, lig_ad_types, feat_coords, feat_elements,
                   edge_cutoff, K_dist, K_mono):
    """Feature Atom Matching QUBO. Returns (Q, variables) with
    variables[k] = (ligand_atom_index, feature_index)."""
    lig_dmat = dist_matrix(lig_coords)
    feat_dmat = dist_matrix(feat_coords)
    weights, atom_idx, site_idx = [], [], []
    for i, t_raw in enumerate(lig_ad_types):
        t = t_raw if t_raw in EN else t_raw[:1]
        if t == "A":
            t = "C"
        if t not in EN:
            continue
        for j, fe in enumerate(feat_elements):
            fe = fe if fe in EN else fe[:1]
            if fe not in EN:
                continue
            weights.append(abs(EN[t] - EN[fe]) - 0.5)
            atom_idx.append(i); site_idx.append(j)
    Q = _assemble(weights, atom_idx, site_idx, lig_dmat, feat_dmat,
                  edge_cutoff, K_dist, K_mono)
    return Q, list(zip(atom_idx, site_idx))
