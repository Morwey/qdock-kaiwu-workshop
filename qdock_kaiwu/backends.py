# -*- coding: utf-8 -*-
"""QUBO solving on the Kaiwu SDK: classical simulated annealing and the real
Coherent Ising Machine.

Kaiwu's optimizers minimise an *Ising* Hamiltonian and return *spin* (+/-1)
solutions. We convert QUBO <-> Ising with Kaiwu's own routine, solve, and map
the ancilla-gauged spins back to binary:

    ising, offset = kw.conversion.qubo_matrix_to_ising_matrix(Q)
    spins         = optimizer.solve(ising)
    x_i           = (1 + s_i * s_ancilla) / 2

The mapping is exact: kw.common.hamiltonian(ising, s) + offset == x^T Q x for the
mapped binary, for every spin configuration (checked in tests/test_core.py).

Sign convention -- IMPORTANT. Kaiwu's raw `optimizer.solve(M)` *maximizes* s^T M s
(this is why Kaiwu's own MaxCut example feeds `-adjacency`). We never touch that
raw path: we always go QUBO matrix -> `kw.conversion.qubo_matrix_to_ising_matrix`
-> `solve`. That converter already bakes the sign in, so feeding its output to the
maximizer *minimizes* the QUBO. Do NOT add a manual `-` -- negating here would
return the QUBO maximum. (tests/test_core.py asserts the pipeline returns the MIN.)

Two backends, one interface (`solve_qubo`):

    backend="sa"    kaiwu.classical.SimulatedAnnealingOptimizer   (CPU annealer)
    backend="cim"   kaiwu.cim.CIMOptimizer                        (photonic CIM)
"""

import numpy as np


def init_license(user_id=None, sdk_code=None):
    """Initialise the Kaiwu license once per environment. Reads
    KAIWU_USER_ID / KAIWU_SDK_CODE from the environment if not passed."""
    import os
    import kaiwu as kw
    user_id = user_id or os.environ.get("KAIWU_USER_ID")
    sdk_code = sdk_code or os.environ.get("KAIWU_SDK_CODE")
    if not user_id or not sdk_code:
        raise ValueError("Kaiwu credentials missing; pass user_id/sdk_code or set "
                         "KAIWU_USER_ID / KAIWU_SDK_CODE (from platform.qboson.com).")
    kw.license.init(user_id=str(user_id), sdk_code=str(sdk_code))
    return True


def qubo_energy(Q, b):
    b = np.asarray(b, dtype=float)
    return float(b @ Q @ b)


def _spins_to_binary(spin_row, n):
    s = np.asarray(spin_row).astype(int)
    return ((s[:n] * s[-1]) + 1) // 2


def _rank_unique(binaries, Q):
    seen, ranked = set(), []
    for b in binaries:
        b = np.asarray(b).astype(int)
        key = b.tobytes()
        if key in seen:
            continue
        seen.add(key)
        ranked.append((qubo_energy(Q, b), b))
    ranked.sort(key=lambda t: t[0])
    return ranked


# ---------------------------------------------------------------------------
# Kaiwu classical simulated annealing
# ---------------------------------------------------------------------------
def solve_sa(Q, n_pos=300, seed=42, initial_temperature=10.0, alpha=0.999,
             cutoff_temperature=0.01, iterations_per_t=2000):
    """Solve a QUBO with kaiwu.classical.SimulatedAnnealingOptimizer.

    Kaiwu exposes the whole annealing schedule, so the initial temperature is set
    to match the QUBO's energy scale (T0 ~ 10 for these docking problems -- see
    params.py / the README). `n_pos` is the number of independent reads
    (size_limit); `seed` is rand_seed. Returns [(energy, binary), ...] best-first.
    """
    import kaiwu as kw
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if n == 0:
        return []
    ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)
    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=initial_temperature, alpha=alpha,
        cutoff_temperature=cutoff_temperature, iterations_per_t=iterations_per_t,
        size_limit=n_pos, rand_seed=seed)
    spins = np.asarray(optimizer.solve(ising))
    return _rank_unique([_spins_to_binary(s, n) for s in spins], Q)


# ---------------------------------------------------------------------------
# Real Coherent Ising Machine (photonic hardware)
# ---------------------------------------------------------------------------
def solve_cim(Q, n_pos=10, task_name="qdock", task_mode="quota", wait=True,
              interval=1, save_dir="/tmp/kaiwu_cim", **_):
    """Solve a QUBO on the real CIM with kaiwu.cim.CIMOptimizer: the Ising matrix
    is submitted to the photonic machine and the result is polled back.
    `sample_number = n_pos`. Suited to small QUBOs; a docking-scale QUBO
    (10^3 spins) needs a correspondingly large hardware quota."""
    import os
    import kaiwu as kw
    os.makedirs(save_dir, exist_ok=True)
    kw.common.CheckpointManager.save_dir = save_dir
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if n == 0:
        return []
    ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)
    optimizer = kw.cim.CIMOptimizer(task_name=task_name, wait=wait, interval=interval,
                                    task_mode=task_mode, sample_number=max(10, n_pos))
    spins = np.asarray(optimizer.solve(ising))
    return _rank_unique([_spins_to_binary(s, n) for s in spins], Q)


_BACKENDS = {"sa": solve_sa, "cim": solve_cim}


def solve_qubo(Q, n_pos=300, backend="sa", seed=42, **params):
    """Solve a QUBO matrix on Kaiwu; return [(energy, binary), ...] best-first.

        backend="sa"   -> kaiwu.classical.SimulatedAnnealingOptimizer
        backend="cim"  -> kaiwu.cim.CIMOptimizer  (real machine)
    """
    if backend not in _BACKENDS:
        raise ValueError("backend must be 'sa' or 'cim', got %r" % backend)
    if backend == "cim":
        return solve_cim(Q, n_pos=n_pos, **params)
    return solve_sa(Q, n_pos=n_pos, seed=seed,
                    **{k: v for k, v in params.items()
                       if k in ("initial_temperature", "alpha",
                                "cutoff_temperature", "iterations_per_t")})
