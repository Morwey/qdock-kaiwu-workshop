# -*- coding: utf-8 -*-
"""File readers/writers: PDBQT, AutoGrid maps, AutoSite pockets, pose output."""

import numpy as np

_AD_TO_ELEMENT = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "OS": "O",
    "O": "O", "SA": "S", "S": "S", "HD": "H", "HS": "H", "H": "H", "P": "P",
    "F": "F", "Cl": "Cl", "CL": "Cl", "Br": "Br", "BR": "Br", "I": "I",
    "Mg": "Mg", "MG": "Mg", "Zn": "Zn", "ZN": "Zn", "Ca": "Ca", "CA": "Ca",
    "Fe": "Fe", "FE": "Fe", "Mn": "Mn", "MN": "Mn", "Si": "Si", "B": "B",
}


def ad_type_to_element(ad_type):
    t = ad_type.strip()
    if t in _AD_TO_ELEMENT:
        return _AD_TO_ELEMENT[t]
    two = t[:2].capitalize()
    if two in ("Cl", "Br", "Si", "Ca", "Mg", "Zn", "Fe", "Mn"):
        return two
    return t[0].upper() if t else "C"


def read_pdbqt(path):
    """Return dict(coords, ad_types, elements, charges, lines) from a .pdbqt."""
    coords, ad_types, charges, lines = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if line[0:6] in ("ATOM  ", "HETATM"):
                coords.append([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
                try:
                    charges.append(float(line[66:76]))
                except ValueError:
                    charges.append(0.0)
                ad_types.append(line[77:79].strip())
                lines.append(line.rstrip("\n"))
    return dict(coords=np.array(coords, dtype=float), ad_types=ad_types,
                elements=[ad_type_to_element(t) for t in ad_types],
                charges=np.array(charges), lines=lines)


def read_autogrid_map(path, cutoff=0.0):
    """Read an AutoGrid .map (6 header lines + one vdW energy per grid point).
    Returns (positions, energies) keeping points with energy < cutoff."""
    with open(path) as fh:
        es = np.array([float(x) for x in fh.readlines()[6:]])
    if isinstance(cutoff, float):
        pos = np.argwhere(es < cutoff).flatten()
        return pos, es[pos]
    return np.arange(len(es)), es


def read_autosite_pocket(path):
    """Read an AutoSite feature-point PDB. Returns (coords, elements)."""
    coords, elements = [], []
    with open(path) as fh:
        for line in fh:
            if line[0:6] in ("ATOM  ", "HETATM"):
                coords.append([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
                el = line[76:78].strip() or line[12:16].strip()[:1]
                elements.append(el)
    return np.array(coords, dtype=float), elements


def write_pose_pdbqt(template_lines, coords, out_path):
    """Clone a ligand .pdbqt with new coordinates (keeps types/charges for Vina)."""
    out = ["%s%8.3f%8.3f%8.3f%s" % (ln[:30], x, y, z, ln[54:])
           for ln, (x, y, z) in zip(template_lines, coords)]
    with open(out_path, "w") as fh:
        fh.write("ROOT\n" + "\n".join(out) + "\nENDROOT\nTORSDOF 0\n")
    return out_path


def write_multimodel_pdb(template_lines, poses, out_path):
    """Write sampled poses as a multi-MODEL PDB (load in PyMOL/VMD)."""
    with open(out_path, "w") as fh:
        for m, coords in enumerate(poses, 1):
            fh.write("MODEL %8d\n" % m)
            for ln, (x, y, z) in zip(template_lines, coords):
                rec = ln[:30] if ln[:6] in ("ATOM  ", "HETATM") else \
                    ("ATOM  " + ln[6:30])
                fh.write("%s%8.3f%8.3f%8.3f  1.00  0.00\n" % (rec[:30], x, y, z))
            fh.write("ENDMDL\n")
    return out_path
