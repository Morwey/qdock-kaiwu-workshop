# -*- coding: utf-8 -*-
"""Locate and drive the external command-line tools.

Required (conda-forge, no ADFR needed for GPM):
    obabel      OpenBabel        PDB/MOL2 -> PDBQT
    autogrid4   AutoGrid 4.2.x   van der Waals grids (GPM)
    vina        AutoDock Vina    pose scoring
Optional:
    autosite    ADFR suite       pocket feature atoms (FAM); a built-in
                                  substitute is used when absent.

Override any path with QDOCK_OBABEL / QDOCK_AUTOGRID / QDOCK_VINA / QDOCK_AUTOSITE.
"""

import os
import shutil
import subprocess

_ENV_BINS = [os.path.expanduser("~/miniforge3/envs/chem/bin"),
             os.path.expanduser("~/miniconda3/envs/chem/bin")]


def _resolve(name, env_var):
    p = os.environ.get(env_var)
    if p and os.path.exists(p):
        return p
    p = shutil.which(name)
    if p:
        return p
    for base in _ENV_BINS:
        cand = os.path.join(base, name)
        if os.path.exists(cand):
            return cand
    return None


def tool_paths():
    return {"obabel": _resolve("obabel", "QDOCK_OBABEL"),
            "autogrid4": _resolve("autogrid4", "QDOCK_AUTOGRID"),
            "vina": _resolve("vina", "QDOCK_VINA"),
            "autosite": _resolve("autosite", "QDOCK_AUTOSITE")}


def require(name):
    p = tool_paths().get(name)
    if not p:
        raise RuntimeError(
            "tool '%s' not found; install it or set QDOCK_%s" % (name, name.upper()))
    return p


def run(cmd, cwd=None, check=True):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError("command failed: %s\n%s\n%s"
                           % (" ".join(cmd), res.stdout, res.stderr))
    return res


def prepare_receptor_pdbqt(rec_path, out_path):
    """protein .pdb/.mol2 -> rigid receptor .pdbqt (AutoDock4 types + charges;
    AutoGrid requires charges on the receptor).

    A .mol2 receptor already carries partial charges, so we keep them: forcing
    Gasteiger on a large (10^3-atom) SYBYL protein mol2 makes OpenBabel abort
    ("0 molecules converted"). A .pdb has no charges, so we add Gasteiger."""
    cmd = [require("obabel"), rec_path, "-O", out_path, "-xr"]
    if not rec_path.lower().endswith(".mol2"):
        cmd += ["--partialcharge", "gasteiger"]
    run(cmd)
    return out_path


def prepare_ligand_pdbqt(lig_path, out_path):
    """ligand .mol2/.pdb -> .pdbqt (nonpolar H merged, AutoDock4 types)."""
    run([require("obabel"), lig_path, "-O", out_path,
         "--partialcharge", "gasteiger"])
    return out_path
