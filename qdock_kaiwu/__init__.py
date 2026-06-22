# -*- coding: utf-8 -*-
"""QDock-Kaiwu: molecular-docking pose sampling as a QUBO, solved on the Kaiwu
SDK (Bose Quantum). Grid Point Matching and Feature Atom Matching encodings;
classical-SA, real-CIM, and reference solvers; AutoDock Vina scoring."""

from .gpm import GPMDock, Ligand
from .fam import FAMDock
from .backends import solve_qubo, init_license, qubo_energy
from . import params, qubo, geometry, io, scoring, evaluate, tools, backends, viz

__all__ = ["GPMDock", "FAMDock", "Ligand", "solve_qubo", "init_license",
           "qubo_energy", "params", "qubo", "geometry", "io", "scoring",
           "evaluate", "tools", "backends", "viz"]
__version__ = "1.0.0"
