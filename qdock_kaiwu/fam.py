# -*- coding: utf-8 -*-
"""Feature Atom Matching docking engine."""

import os
import numpy as np

from . import io, tools, backends
from .geometry import superpose_apply, dist_matrix
from .qubo import build_fam_qubo
from .params import FAM, sa_schedule
from .gpm import Ligand, _matches_to_poses


class FAMDock:
    """Feature Atom Matching. Each ligand atom is matched to a pocket feature
    atom; the QUBO reward is the electronegativity mismatch. Far fewer variables
    (qubits) than GPM. Uses ADFR's ``autosite`` if present, otherwise the
    built-in ``pocket_points`` substitute."""

    def __init__(self, backend="sa", workdir="run_fam",
                 max_features=FAM["max_features"]):
        self.backend = backend
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.max_features = max_features

    def make_receptor(self, receptor_pdb):
        self.receptor_name = os.path.splitext(os.path.basename(receptor_pdb))[0]
        self.receptor_pdbqt = os.path.join(self.workdir, "receptor.pdbqt")
        tools.prepare_receptor_pdbqt(receptor_pdb, self.receptor_pdbqt)
        rec = io.read_pdbqt(self.receptor_pdbqt)
        self.rec_coords = rec["coords"]
        self.rec_elements = np.array(rec["elements"])

    def make_ligand(self, ligand_paths):
        ligdir = os.path.join(self.workdir, "ligands")
        os.makedirs(ligdir, exist_ok=True)
        self.ligands = [Ligand(p, ligdir) for p in ligand_paths]

    def make_box_ligand(self, ref_ligand, center_length=8.0,
                        grid_length=FAM["grid_length"]):
        ref = io.read_pdbqt(self._as_pdbqt(ref_ligand))["coords"]
        self.box_center = ref.mean(0)
        self.box_lengths = center_length + 2 * np.abs(ref - self.box_center).max(0)
        self.grid_length = grid_length
        self._features()

    def make_box_input(self, x, y, z, dx, dy, dz, grid_length=FAM["grid_length"]):
        self.box_center = np.array([x, y, z], dtype=float)
        self.box_lengths = np.array([dx, dy, dz], dtype=float)
        self.grid_length = grid_length
        self._features()

    def _features(self):
        if tools.tool_paths().get("autosite"):
            self.feat_coords, self.feat_elements = self._autosite()
        else:
            self.feat_coords, self.feat_elements = self.pocket_points()

    def _autosite(self):
        pocs = os.path.join(self.workdir, "pocs")
        os.makedirs(pocs, exist_ok=True)
        tools.run([tools.require("autosite"), "-r", self.receptor_pdbqt,
                   "--boxcenter", "[%.3f,%.3f,%.3f]" % tuple(self.box_center),
                   "--boxdim", "[%d,%d,%d]" % tuple(self.box_lengths.astype(int)),
                   "-o", pocs], cwd=self.workdir, check=False)
        coords, elements = [], []
        for f in os.listdir(pocs):
            if "_fp_" in f:
                c, e = io.read_autosite_pocket(os.path.join(pocs, f))
                if len(c):
                    coords.append(c); elements += e
        if not coords:
            return self.pocket_points()
        return np.vstack(coords), elements

    def pocket_points(self):
        """Built-in AutoSite substitute: grid points in the receptor's first
        hydration shell (2.6-5.0 A from the nearest heavy atom), typed by the
        nearest residue atom (C/N/O), capped to ``max_features``. A KD-tree keeps
        this O(N log N) so fine grids stay tractable."""
        from scipy.spatial import cKDTree
        gl = self.grid_length
        lo = self.box_center - self.box_lengths / 2.0
        ns = (self.box_lengths / gl).astype(int) + 1
        xs, ys, zs = (lo[k] + np.arange(ns[k]) * gl for k in range(3))
        pts = np.array([[x, y, z] for z in zs for y in ys for x in xs])
        heavy = self.rec_elements != "H"
        tree = cKDTree(self.rec_coords[heavy])
        dmin, idx = tree.query(pts, k=1)
        nearest = self.rec_elements[heavy][idx]
        shell = (dmin > 2.6) & (dmin < 5.0)
        pts, nearest = pts[shell], nearest[shell]
        elements = np.where(np.isin(nearest, ["O"]), "O",
                            np.where(np.isin(nearest, ["N"]), "N", "C"))
        if len(pts) > self.max_features:
            keep = np.argsort(np.linalg.norm(pts - self.box_center, axis=1))[:self.max_features]
            pts, elements = pts[keep], elements[keep]
        return pts, list(elements)

    def feature_coords_at(self, grid_length, max_features=10**9):
        """Geometry-only helper: the feature-atom coordinates at a given spacing,
        used to show how the discretization floor scales with resolution without
        running a full dock."""
        saved_gl, saved_mf = getattr(self, "grid_length", None), self.max_features
        self.grid_length, self.max_features = grid_length, max_features
        try:
            pts, _ = self.pocket_points()
        finally:
            self.max_features = saved_mf
            if saved_gl is not None:
                self.grid_length = saved_gl
        return pts

    def dock(self, ligand, edge_cutoff=FAM["edge_cutoff"], K_dist=FAM["K_dist"],
             K_mono=FAM["K_mono"], n_pos=FAM["n_pos"], seed=FAM["seed"],
             solver_params=None, save_pose=True):
        Q, variables = build_fam_qubo(ligand.coords, ligand.ad_types,
                                      self.feat_coords, self.feat_elements,
                                      edge_cutoff, K_dist, K_mono)
        params = sa_schedule(FAM)
        params.update(solver_params or {})
        ranked = backends.solve_qubo(Q, n_pos=n_pos, backend=self.backend,
                                     seed=seed, **params)
        poses = _matches_to_poses(ligand, np.array(variables), self.feat_coords, ranked)
        if save_pose and poses:
            posedir = os.path.join(self.workdir, "poses")
            os.makedirs(posedir, exist_ok=True)
            io.write_multimodel_pdb(ligand.lines, poses,
                                    os.path.join(posedir, ligand.name + "_poses.pdb"))
        self.last_qubo = Q
        return np.array(poses) if poses else np.array([])

    def _as_pdbqt(self, path):
        if path.endswith(".pdbqt"):
            return path
        return tools.prepare_ligand_pdbqt(path, os.path.join(self.workdir, "boxref.pdbqt"))
