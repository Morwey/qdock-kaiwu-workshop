# -*- coding: utf-8 -*-
"""Grid Point Matching docking engine."""

import os
import numpy as np

from . import io, tools, backends
from .geometry import superpose_apply
from .qubo import build_gpm_qubo
from .params import GPM, sa_schedule


class Ligand:
    def __init__(self, src_path, workdir):
        self.name = os.path.splitext(os.path.basename(src_path))[0]
        self.pdbqt = os.path.join(workdir, self.name + ".pdbqt")
        tools.prepare_ligand_pdbqt(src_path, self.pdbqt)
        rec = io.read_pdbqt(self.pdbqt)
        self.coords = rec["coords"]
        self.ad_types = rec["ad_types"]
        self.elements = rec["elements"]
        self.lines = rec["lines"]
        self.n = len(self.coords)


class GPMDock:
    """Grid Point Matching. Each ligand atom is matched to a grid point whose
    van der Waals energy (from AutoGrid) is the QUBO reward."""

    def __init__(self, backend="sa", workdir="run_gpm"):
        self.backend = backend
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

    def make_receptor(self, receptor_pdb):
        self.receptor_name = os.path.splitext(os.path.basename(receptor_pdb))[0]
        self.receptor_pdbqt = os.path.join(self.workdir, "receptor.pdbqt")
        tools.prepare_receptor_pdbqt(receptor_pdb, self.receptor_pdbqt)
        self.receptor_ad_types = np.unique(io.read_pdbqt(self.receptor_pdbqt)["ad_types"])

    def use_receptor_pdbqt(self, pdbqt_path):
        """Use an already-prepared receptor .pdbqt directly (skip the PDB->PDBQT
        prep). AutoGrid reads the receptor from the work dir, so copy it in."""
        import shutil
        self.receptor_name = os.path.splitext(os.path.basename(pdbqt_path))[0]
        self.receptor_pdbqt = os.path.join(self.workdir, "receptor.pdbqt")
        shutil.copy(pdbqt_path, self.receptor_pdbqt)
        self.receptor_ad_types = np.unique(io.read_pdbqt(self.receptor_pdbqt)["ad_types"])

    def make_ligand(self, ligand_paths):
        ligdir = os.path.join(self.workdir, "ligands")
        os.makedirs(ligdir, exist_ok=True)
        self.ligands = [Ligand(p, ligdir) for p in ligand_paths]
        self.ligand_ad_types = np.unique(np.hstack([l.ad_types for l in self.ligands]))

    def make_box_ligand(self, ref_ligand, center_length=8.0,
                        grid_length=GPM["grid_length"], cutoff=GPM["cutoff"]):
        ref = io.read_pdbqt(self._as_pdbqt(ref_ligand))["coords"]
        self.box_center = ref.mean(0)
        self.box_lengths = center_length + 2 * np.abs(ref - self.box_center).max(0)
        self.grid_length = grid_length
        self._autogrid(cutoff)

    def make_box_input(self, x, y, z, dx, dy, dz,
                       grid_length=GPM["grid_length"], cutoff=GPM["cutoff"]):
        self.box_center = np.array([x, y, z], dtype=float)
        self.box_lengths = np.array([dx, dy, dz], dtype=float)
        self.grid_length = grid_length
        self._autogrid(cutoff)

    def _autogrid(self, cutoff):
        gl = self.grid_length
        dims = (self.box_lengths / gl).astype(int)
        dims = np.where(dims % 2 == 0, dims - 1, dims)
        self.dims = dims
        npts = dims - 1
        gpf = os.path.join(self.workdir, self.receptor_name + ".gpf")
        info = ("npts %d %d %d\n" % tuple(npts) +
                "gridfld %s.maps.fld\n" % self.receptor_name +
                "spacing %.4f\n" % gl +
                "receptor_types %s\n" % " ".join(self.receptor_ad_types) +
                "ligand_types %s\n" % " ".join(self.ligand_ad_types) +
                "receptor %s\n" % os.path.basename(self.receptor_pdbqt) +
                "gridcenter %.3f %.3f %.3f\nsmooth 0.5\n" % tuple(self.box_center))
        for t in self.ligand_ad_types:
            info += "map %s.%s.map\n" % (self.receptor_name, t)
        info += "elecmap %s.e.map\ndsolvmap %s.d.map\n" % (self.receptor_name,
                                                           self.receptor_name)
        with open(gpf, "w") as fh:
            fh.write(info)
        tools.run([tools.require("autogrid4"), "-p", os.path.basename(gpf),
                   "-l", self.receptor_name + ".glg"], cwd=self.workdir)
        box_st = self.box_center - ((dims - 1) / 2.0) * gl
        xs, ys, zs = (box_st[k] + np.arange(dims[k]) * gl for k in range(3))
        self.box_coords = np.array([[x, y, z] for z in zs for y in ys for x in xs])
        self.grid_dict = {t: io.read_autogrid_map(
            os.path.join(self.workdir, "%s.%s.map" % (self.receptor_name, t)),
            cutoff=cutoff) for t in self.ligand_ad_types}

    def dock(self, ligand, edge_cutoff=GPM["edge_cutoff"], K_dist=GPM["K_dist"],
             K_mono=GPM["K_mono"], n_pos=GPM["n_pos"], seed=GPM["seed"],
             solver_params=None, save_pose=True):
        """Build the GPM QUBO, solve it, and return the sampled poses
        (n_poses, n_atoms, 3)."""
        Q, variables = build_gpm_qubo(ligand.coords, ligand.ad_types,
                                      self.grid_dict, self.box_coords,
                                      edge_cutoff, K_dist, K_mono)
        params = sa_schedule(GPM)
        params.update(solver_params or {})
        ranked = backends.solve_qubo(Q, n_pos=n_pos, backend=self.backend,
                                     seed=seed, **params)
        poses, energies = _matches_to_poses(ligand, np.array(variables),
                                            self.box_coords, ranked)
        self.last_energies = np.array(energies)
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
        out = os.path.join(self.workdir, "boxref.pdbqt")
        return tools.prepare_ligand_pdbqt(path, out)


def _matches_to_poses(ligand, variables, sites, ranked):
    """Turn each QUBO solution into a pose; return (poses, energies) aligned."""
    poses, energies = [], []
    for energy, b in ranked:
        on = np.where(b == 1)[0]
        if len(on) < 3:                     # need >=3 matches to fit a rigid body
            continue
        poses.append(superpose_apply(ligand.coords[variables[on, 0]],
                                     sites[variables[on, 1]], ligand.coords))
        energies.append(energy)
    return poses, energies
