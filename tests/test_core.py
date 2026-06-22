# -*- coding: utf-8 -*-
"""Correctness checks that need no Kaiwu license.
Run:  conda run -n qdock python tests/test_core.py
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdock_kaiwu import geometry as G, qubo as QB
from qdock_kaiwu.backends import qubo_energy, _spins_to_binary

ok = True
def check(name, cond):
    global ok; ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name)

# Kabsch recovers a known rigid transform
rng = np.random.default_rng(0)
P = rng.normal(size=(10, 3))
th = 0.7
R0 = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
t0 = np.array([3.0, -1.0, 2.0])
T = P @ R0.T + t0
R, t = G.kabsch(P, T)
check("kabsch recovers transform", np.allclose(P @ R.T + t, T, atol=1e-9))
check("heavy_atom_rmsd ignores H",
      G.heavy_atom_rmsd(np.zeros((2, 3)), np.array([[9, 0, 0], [0, 0, 0]]), ["H", "C"]) < 1e-9)

# Kaiwu QUBO->Ising->binary mapping equivalence (license-free conversion)
try:
    import kaiwu as kw
    good = True
    for trial in range(3):
        r = np.random.default_rng(trial + 5)
        n = 6
        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = r.integers(-5, 6)
            for j in range(i + 1, n):
                if r.random() < 0.7:
                    Q[i, j] = r.integers(-5, 6)
        ising, off = kw.conversion.qubo_matrix_to_ising_matrix(Q)
        spins = np.array(list(itertools.product([-1, 1], repeat=n + 1)))
        H = kw.common.hamiltonian(ising, spins).flatten()
        qe = np.array([qubo_energy(Q, _spins_to_binary(s, n)) for s in spins])
        good &= np.allclose(H + off, qe, atol=1e-6)
    check("Kaiwu Hamiltonian+offset == QUBO(mapped binary) for ALL spins", good)
except Exception as e:
    print("SKIP kaiwu mapping check:", e)

# GPM/FAM QUBO assembly matches the definition
lig = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]], float)
ad = ["C", "O", "N", "C"]
box = np.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [0, 1.5, 0]], float)
grid = {"C": (np.array([0, 3]), np.array([-1.0, -0.5])),
        "O": (np.array([1]), np.array([-2.0])), "N": (np.array([2]), np.array([-1.5]))}
Q, variables = QB.build_gpm_qubo(lig, ad, grid, box, 1.0, 0.4, 25.0)
diag_ok = all(abs(Q[k, k] - dict(zip(*grid[ad[i]]))[pos]) < 1e-9
              for k, (i, pos) in enumerate(variables))
check("GPM diagonal == grid vdW energies", diag_ok)
faQ, favars = QB.build_fam_qubo(lig, ad, box, ["C", "O", "N", "C"], 1.0, 2.0, 11.0)
fa_ok = all(abs(faQ[k, k] - (abs(QB.EN[ad[i]] - QB.EN[["C", "O", "N", "C"][j]]) - 0.5)) < 1e-9
            for k, (i, j) in enumerate(favars))
check("FAM diagonal == |EN diff| - 0.5", fa_ok)

# Sign convention: the pipeline must return the QUBO MINIMUM, not the maximum
# (Kaiwu's raw solve maximizes sᵀMs; qubo_matrix_to_ising_matrix bakes in the
# sign, so we must NOT negate). Needs a license; skipped otherwise.
if os.environ.get("KAIWU_USER_ID") and os.environ.get("KAIWU_SDK_CODE"):
    import kaiwu as kw
    from qdock_kaiwu import solve_qubo, init_license
    init_license()
    r = np.random.default_rng(1)
    n = 6
    Qs = np.zeros((n, n))
    for i in range(n):
        Qs[i, i] = r.integers(-5, 6)
        for j in range(i + 1, n):
            if r.random() < 0.7:
                Qs[i, j] = r.integers(-5, 6)
    allb = list(itertools.product([0, 1], repeat=n))
    bmin = min(qubo_energy(Qs, np.array(b)) for b in allb)
    bmax = max(qubo_energy(Qs, np.array(b)) for b in allb)
    got = solve_qubo(Qs, n_pos=50, backend="sa")[0][0]
    check("Kaiwu SA pipeline returns the QUBO MINIMUM (not maximum)",
          abs(got - bmin) < 1e-9 and abs(got - bmax) > 1e-9)
else:
    print("SKIP sign-convention solve test (no KAIWU_USER_ID / KAIWU_SDK_CODE)")

print("\nALL CORE TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
