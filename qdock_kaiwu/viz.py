# -*- coding: utf-8 -*-
"""Plots for a docking run: pose-RMSD distribution, QUBO-energy vs RMSD, and the
sampled-vs-crystal pose overlay (red = sampled, blue = crystal)."""

import numpy as np

_RED, _BLUE, _GREEN = "#D1495B", "#3B5BA5", "#55A868"
_ELEMENT_COLOR = {"C": "#444444", "N": "#3B5BA5", "O": "#D1495B", "S": "#E2A100",
                  "H": "#BBBBBB", "P": "#E08A00"}


def plot_qubo_matrix(ax, Q, max_show=60):
    """Heatmap of a (small) QUBO matrix: the diagonal holds the per-match rewards,
    the off-diagonal the distance / monogamy penalties."""
    Q = np.asarray(Q, dtype=float)
    n = min(Q.shape[0], max_show)
    M = Q[:n, :n] + Q[:n, :n].T
    np.fill_diagonal(M, np.diag(Q[:n, :n]))
    vmax = np.abs(M).max() or 1.0
    im = ax.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("QUBO matrix (%d×%d)" % (n, n))
    ax.set_xlabel("variable"); ax.set_ylabel("variable")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_grid_sites(ax, grid_points, energies, ligand_coords, max_points=2500):
    """3-D view of the GPM docking box: grid points coloured by van der Waals
    energy (blue = favourable) with the ligand heavy atoms (black) on top."""
    pts = np.asarray(grid_points, dtype=float)
    en = np.asarray(energies, dtype=float)
    if len(pts) > max_points:
        idx = np.argsort(en)[:max_points]          # keep the most favourable
        pts, en = pts[idx], en[idx]
    s = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=en, cmap="viridis_r",
                   s=10, alpha=0.5, edgecolor="none")
    lig = np.asarray(ligand_coords, dtype=float)
    ax.scatter(lig[:, 0], lig[:, 1], lig[:, 2], color="black", s=35, label="ligand")
    ax.set_title("GPM grid points (%d) coloured by vdW energy" % len(pts))
    ax.figure.colorbar(s, ax=ax, fraction=0.046, pad=0.04, label="vdW energy")
    ax.legend(fontsize=8)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def plot_feature_sites(ax, feat_coords, feat_elements, ligand_coords):
    """3-D view of FAM feature atoms (coloured by type) with the ligand."""
    fc = np.asarray(feat_coords, dtype=float)
    fe = np.asarray([str(e).strip().upper()[:1] for e in feat_elements])
    for el in np.unique(fe):
        m = fe == el
        ax.scatter(fc[m, 0], fc[m, 1], fc[m, 2], color=_ELEMENT_COLOR.get(el, "#888"),
                   s=55, alpha=0.9, label="feature %s" % el)
    lig = np.asarray(ligand_coords, dtype=float)
    ax.scatter(lig[:, 0], lig[:, 1], lig[:, 2], color="black", s=20, alpha=0.6, label="ligand")
    ax.set_title("FAM feature atoms (%d)" % len(fc))
    ax.legend(fontsize=8)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def plot_resolution_bars(ax, labels, disc_errors):
    """Bar chart of the discretization floor at each grid spacing."""
    x = np.arange(len(labels))
    bars = ax.bar(x, disc_errors, color=[_BLUE, _GREEN] * len(labels), alpha=0.85)
    for b, v in zip(bars, disc_errors):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("discretization error (Å)")
    ax.set_title("Finer grid → lower accuracy floor")


def plot_rmsd_distribution(ax, rmsds):
    """Histogram of the heavy-atom RMSD of every sampled pose to the crystal."""
    rmsds = np.asarray(rmsds, dtype=float)
    ax.hist(rmsds, bins=min(30, max(8, len(rmsds) // 8)), color=_BLUE, alpha=0.85)
    for thr, c in [(2.0, "orange"), (1.5, _RED)]:
        ax.axvline(thr, ls="--", lw=1.5, color=c, label="%.1f Å cutoff" % thr)
    ax.axvline(rmsds.min(), ls="-", lw=1.5, color=_GREEN, label="best %.2f Å" % rmsds.min())
    ax.set_xlabel("heavy-atom RMSD to crystal (Å)")
    ax.set_ylabel("sampled poses")
    ax.set_title("Pose RMSD distribution (%d reads)" % len(rmsds))
    ax.legend(fontsize=8)


def plot_energy_vs_rmsd(ax, energies, rmsds):
    """QUBO energy of each pose vs its RMSD. A downward-left cloud means the
    encoding is sound: lower QUBO energy → closer to the native pose."""
    energies = np.asarray(energies, dtype=float)
    rmsds = np.asarray(rmsds, dtype=float)
    ax.scatter(rmsds, energies, s=18, alpha=0.5, color=_GREEN, edgecolor="none")
    i = int(np.argmin(rmsds))
    ax.scatter(rmsds[i], energies[i], s=90, color=_RED, zorder=5,
               edgecolor="k", linewidth=0.5, label="best RMSD pose")
    if len(rmsds) > 2:
        c = np.corrcoef(rmsds, energies)[0, 1]
        ax.set_title("QUBO energy vs RMSD  (r = %.2f)" % c)
    else:
        ax.set_title("QUBO energy vs RMSD")
    ax.set_xlabel("heavy-atom RMSD (Å)")
    ax.set_ylabel("QUBO energy")
    ax.legend(fontsize=8)


def plot_pose_overlay(ax, pose, crystal, elements):
    """3-D overlay of one sampled pose (red) on the crystal pose (blue), heavy
    atoms only, with thin lines linking corresponding atoms. ``ax`` must be a 3-D
    axis (`fig.add_subplot(..., projection='3d')`)."""
    el = np.asarray([str(e).strip().upper() for e in elements])
    keep = el != "H"
    p = np.asarray(pose, dtype=float)[keep]
    c = np.asarray(crystal, dtype=float)[keep]
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], color=_BLUE, s=45, depthshade=True, label="crystal")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=_RED, s=45, depthshade=True, label="sampled (best)")
    for a, b in zip(p, c):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="gray", lw=0.6, alpha=0.5)
    ax.set_title("Best pose vs crystal")
    ax.legend(fontsize=8)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def summary_figure(rmsds, energies, best_pose, crystal, elements, suptitle=None):
    """Build the 3-panel summary figure for a docking run and return it."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 4.3))
    plot_rmsd_distribution(fig.add_subplot(1, 3, 1), rmsds)
    plot_energy_vs_rmsd(fig.add_subplot(1, 3, 2), energies, rmsds)
    plot_pose_overlay(fig.add_subplot(1, 3, 3, projection="3d"), best_pose, crystal, elements)
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)
    fig.tight_layout()
    return fig
