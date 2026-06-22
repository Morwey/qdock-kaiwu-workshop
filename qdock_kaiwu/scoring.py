# -*- coding: utf-8 -*-
"""Score docked poses with AutoDock Vina.

QDock samples poses (the NP-hard step, on the Ising machine); scoring is left to
a classical physics-based function. Vina is used here both to rank QDock's poses
and as an independent yardstick against the crystal pose.
"""

import os
import re
import numpy as np

from . import tools, io

_PATTERNS = [re.compile(r"Estimated Free Energy of Binding\s*[:=]\s*([-\d.]+)"),
             re.compile(r"^\s*Affinity\s*[:=]\s*([-\d.]+)", re.M),
             re.compile(r"^\s*1\s+([-\d.]+)", re.M)]


def vina_score_only(receptor_pdbqt, ligand_pdbqt, center, size):
    """``vina --score_only`` for one in-place pose. Returns affinity (kcal/mol,
    lower = better) or None. Vina 1.2 needs a grid box even to score."""
    res = tools.run([tools.require("vina"), "--receptor", receptor_pdbqt,
                     "--ligand", ligand_pdbqt, "--score_only",
                     "--center_x", "%.3f" % center[0], "--center_y", "%.3f" % center[1],
                     "--center_z", "%.3f" % center[2], "--size_x", "%.3f" % size[0],
                     "--size_y", "%.3f" % size[1], "--size_z", "%.3f" % size[2]],
                    check=False)
    out = res.stdout + "\n" + res.stderr
    for rx in _PATTERNS:
        m = rx.search(out)
        if m:
            return float(m.group(1))
    return None


def score_pose(receptor_pdbqt, template_lines, coords, workdir, tag="pose",
               padding=10.0):
    os.makedirs(workdir, exist_ok=True)
    pose = io.write_pose_pdbqt(template_lines, coords, os.path.join(workdir, tag + ".pdbqt"))
    coords = np.asarray(coords, dtype=float)
    size = np.maximum((coords.max(0) - coords.min(0)) + padding, 8.0)
    return vina_score_only(receptor_pdbqt, pose, coords.mean(0), size)


def score_poses(receptor_pdbqt, template_lines, poses, workdir, tag="pose"):
    return np.array([score_pose(receptor_pdbqt, template_lines, p, workdir,
                                "%s_%d" % (tag, k)) for k, p in enumerate(poses)],
                    dtype=float)
