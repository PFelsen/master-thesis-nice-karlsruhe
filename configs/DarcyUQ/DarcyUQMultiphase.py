import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from fenics import *
from mpi4py import MPI as pyMPI

from uq.sparseGridSC import SparseGridSC
from permeability.multiphase import MultiphaseField

from fem.darcyProblem import Darcy



# MPI rank settings
comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Runtime
script_start_dt = datetime.now()
script_start_perf = time.perf_counter()
if rank == 0:
    print("=" * 80)
    print(f"Started at : {script_start_dt:%Y-%m-%d %H:%M:%S}")
    print("=" * 80)


# FEM settings
meshLevel = 5
degreeRT = 1
elementType = "RT"

# Field settings
stochDims = [12, 10, 8, 6, 4]
mu = 1.0
sigma = 0.25
corrLength = 0.1
numberOfPhases = 2
phaseValues=[2,3]


# MC/SC settings
N_ref = 1000000
mcSamples = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

sclevel = {
    14: 5,
    12: 6,
    10: 7,
    8:  8,
    6:  9,
    4: 12
}

scLevels = [1, 2, 3]#, 4, 5]
SEED = 123456789
sc_methods = [
    {
        "method": "SCClenshawCurtis",
        "gridType": "level",
        "rule": "clenshaw-curtis",
        "localDegree": 0,
    },
    {
        "method": "SCLocalDegree1",
        "gridType": "local",
        "rule": None,
        "localDegree": 1,
    },
    {
        "method": "SCLocalDegree2",
        "gridType": "local",
        "rule": None,
        "localDegree": 2,
    }
]




###########################################################
# Darcy Configuration 
###########################################################
qoiNamesDarcy = [
    "Outflow",
    "Mean flux",
    "Mean pressure box",
    "Pressure L²",
    "Flux L²",
]
qoi_indices = list(range(len(qoiNamesDarcy)))

darcyKwargs = dict(
    permeabilityField=None,
    meshLevel=meshLevel,
    degree=None,
    elementType=elementType,
    darcySolver="mumps",
    darcyPrecon=None,
    pTop=1.0,
    pBottom=0.0,
    plotSolution=False,
    femDarcyVerbose=False,
    linearSolverVerbose=False,
    estimateFemError=False,
)

def darcyProblem(field_cfg, degree):
    """
    Wrapper for Darcy problem which can be called with arbitrary field config and degree.
    """
    config = darcyKwargs.copy()
    config["degree"] = degree
    config["permeabilityField"] = field_cfg["fieldType"](**field_cfg["kWargs"])
    return Darcy(**config), config



###########################################################
# Field Configurations 
###########################################################
field_configs = []

for dim in stochDims:
    field_configs.append({
        "fieldType": MultiphaseField,
        "dim": dim,
        "kWargs": dict(
            dim=dim,
            numberOfPhases=numberOfPhases,
            phaseValues=phaseValues,
            phaseBoundaries=[np.exp(mu)],
            corrLength=corrLength,
            mu=mu,
            sigma=sigma
        )
    })


###########################################################
# MC/SC wrapper
###########################################################

def evalMC(field_cfg, degree, points, sample_sizes):
    """
    Evaluate one nested MC sequence using the largest point set once.
    """
    problem, config = darcyProblem(field_cfg, degree)

    N_max = len(points)
    qoi_dim = len(qoiNamesDarcy)

    local_sum = {N: np.zeros(qoi_dim) for N in sample_sizes}
    local_sumsq = {N: np.zeros(qoi_dim) for N in sample_sizes}
    local_count = {N: 0 for N in sample_sizes}

    start = time.perf_counter()

    localCompleted = 0
    localTotal = len(range(rank, N_max, size))

    for i in range(rank, N_max, size):
        if rank == 0 and localCompleted % 1000 == 0 and localCompleted > 0:
            elapsed = time.perf_counter() - start
            avg_per_sample = elapsed / localCompleted
            remaining = localTotal - localCompleted
            eta_seconds = remaining * avg_per_sample

            print(
                f"[MC] Completed {localCompleted}/{localTotal} samples. "
                f"Elapsed: {elapsed:.1f} s, "
                f"ETA: {eta_seconds:.1f} s."
            )

        q_full = np.asarray(problem.qoi(points[i]), dtype=float)
        q = q_full[qoi_indices]

        for N in sample_sizes:
            if i < N:
                local_sum[N] += q
                local_sumsq[N] += q**2
                local_count[N] += 1

        localCompleted += 1

    results = {}

    for N in sample_sizes:
        total_sum = comm.allreduce(local_sum[N], op=pyMPI.SUM)
        total_sumsq = comm.allreduce(local_sumsq[N], op=pyMPI.SUM)
        total_count = comm.allreduce(local_count[N], op=pyMPI.SUM)

        mean = total_sum / total_count
        second_moment = total_sumsq / total_count

        if total_count > 1:
            variance = np.maximum((total_sumsq - total_count * mean**2) / (total_count - 1), 0.0)
        else:
            variance = np.full_like(mean, np.nan)

        estimator_variance = variance / total_count
        stochastic_error = np.sqrt(estimator_variance)

        runtime = time.perf_counter() - start

        results[N] = dict(
            mean=mean,
            secondMoment=second_moment,
            variance=variance,
            std=np.sqrt(variance),
            estimatorVariance=estimator_variance,
            stochasticError=stochastic_error,
            runtime=runtime,
            N=total_count,
            costPerSample=runtime / total_count,
            config=config,
        )

    return results


def evalSC(field_cfg, degree, levels, gridType, rule=None, localDegree=0):
    """
    Run one nested multi-level SC solve.

    SparseGridSC evaluates the finest grid once and returns arrays over all
    requested levels.
    """
    problem, config = darcyProblem(field_cfg, degree)

    stochDim = field_cfg["dim"]

    sc = SparseGridSC(
        problem=problem,
        qoiNames=qoiNamesDarcy,
        stochDim=stochDim,
        level=levels,
        gridType=gridType,
        rule=rule if rule is not None else "clenshaw-curtis",
        localDegree=localDegree,
        verbose=True,
    )

    sc_result = sc.solve()


    if rank == 0:
        n_points_by_level = []
        for lev in levels:
            grid_lev = sc.makeGrid(lev)
            n_points_by_level.append(grid_lev.getNumPoints())

        n_points_by_level = np.asarray(n_points_by_level, dtype=int)
    else:
        n_points_by_level = None

    n_points_by_level = comm.bcast(n_points_by_level, root=0)

    return dict(
        levels=np.asarray(sc_result["levels"], dtype=int),
        mean=np.asarray(sc_result["mean"], dtype=float),
        secondMoment=np.asarray(sc_result["secondMoment"], dtype=float),
        variance=np.asarray(sc_result["variance"], dtype=float),
        std=np.asarray(sc_result["std"], dtype=float),
        estimatorVariance=np.full_like(sc_result["mean"], np.nan, dtype=float),
        stochasticError=np.full_like(sc_result["mean"], np.nan, dtype=float),
        runtime=sc.info["runtime"],
        N_by_level=n_points_by_level,
        costPerSampleByLevel=sc.info["runtime"] / np.maximum(n_points_by_level, 1),
        config=config,
        sc_info=sc.info,
    )


def append_result_rows(
    rows,
    result,
    fieldType,
    method,
    degree,
    N,
    scLevel,
    referenceMean=None,
    fieldCase=None,
):
    """
    Append one row per QoI for MC-like single-level result.
    """
    if rank != 0:
        return

    for qoi_id, qoi_name in enumerate(qoiNamesDarcy):
        mean = result["mean"][qoi_id]

        reference_mean = np.nan
        abs_error = np.nan
        rel_error = np.nan

        if referenceMean is not None:
            reference_mean = referenceMean[qoi_id]
            abs_error = abs(mean - reference_mean)
            rel_error = abs_error / max(abs(reference_mean), 1e-14)

        rows.append({
            "problem": "Darcy",
            "fieldType": fieldType,
            "fieldCase": fieldCase,
            "method": method,
            "degree": degree,
            "qoi_id": qoi_id,
            "qoi": qoi_name,
            "N": int(N) if not pd.isna(N) else np.nan,
            "scLevel": scLevel,
            "mean": mean,
            "referenceMean": reference_mean,
            "absError": abs_error,
            "relError": rel_error,
            "variance": result["variance"][qoi_id],
            "std": result["std"][qoi_id],
            "estimatorVariance": result["estimatorVariance"][qoi_id],
            "stochasticError": result["stochasticError"][qoi_id],
            "runtime": result["runtime"],
            "costPerSample": result["costPerSample"],
            "mpi_size": size,
            "h": 2.0 ** (-meshLevel),
            "elementType": result["config"]["elementType"],
            "solver": result["config"]["darcySolver"],
            "preconditioner": result["config"]["darcyPrecon"],
        })


def append_sc_result_rows(
    rows,
    sc_result,
    fieldType,
    method,
    degree,
    referenceMean=None,
    fieldCase=None,
):
    """
    Append one row per QoI and per SC level.
    """
    if rank != 0:
        return

    levels = sc_result["levels"]
    n_points = sc_result["N_by_level"]
    cost_per_sample = sc_result["costPerSampleByLevel"]

    for level_id, lev in enumerate(levels):
        for qoi_id, qoi_name in enumerate(qoiNamesDarcy):
            mean = sc_result["mean"][level_id, qoi_id]

            reference_mean = np.nan
            abs_error = np.nan
            rel_error = np.nan

            if referenceMean is not None:
                reference_mean = referenceMean[qoi_id]
                abs_error = abs(mean - reference_mean)
                rel_error = abs_error / max(abs(reference_mean), 1e-14)

            rows.append({
                "problem": "Darcy",
                "fieldType": fieldType,
                "fieldCase": fieldCase,
                "method": method,
                "degree": degree,
                "qoi_id": qoi_id,
                "qoi": qoi_name,
                "N": int(n_points[level_id]),
                "scLevel": int(lev),
                "mean": mean,
                "referenceMean": reference_mean,
                "absError": abs_error,
                "relError": rel_error,
                "variance": sc_result["variance"][level_id, qoi_id],
                "std": sc_result["std"][level_id, qoi_id],
                "estimatorVariance": np.nan,
                "stochasticError": np.nan,
                "runtime": sc_result["runtime"],
                "costPerSample": cost_per_sample[level_id],
                "mpi_size": size,
                "h": 2.0 ** (-meshLevel),
                "elementType": sc_result["config"]["elementType"],
                "solver": sc_result["config"]["darcySolver"],
                "preconditioner": sc_result["config"]["darcyPrecon"],
            })


# ==========================================================
# Main study
# ==========================================================

if rank == 0:
    print("\nField configurations:")
    for cfg in field_configs:
        print(
            f"{cfg['fieldType'].__name__:25s} | "            
            f"{cfg['kWargs']}"
        )

    print("\nSC methods:")
    for cfg in sc_methods:
        print(cfg)

rows_ref = []
rows = []

outer_tasks = field_configs
completed_outer = 0
outer_start = time.perf_counter()

for field_cfg in outer_tasks:
    degree = degreeRT

    stochDim = field_cfg["dim"]

    if rank == 0 and completed_outer > 0:
        elapsed = time.perf_counter() - outer_start
        avg_time = elapsed / completed_outer
        remaining = len(outer_tasks) - completed_outer
        eta_seconds = avg_time * remaining
        now = datetime.now()
        finish_time = now + timedelta(seconds=eta_seconds)

        print(
            f"[{now:%H:%M:%S}] "
            f"completed {completed_outer}/{len(outer_tasks)} "
            f"| avg={avg_time:.1f}s "
            f"| ETA={eta_seconds / 60:.1f} min "
            f"| finish ≈ {finish_time:%H:%M:%S}",
            flush=True,
        )
        print("=" * 80)

    fieldType = field_cfg["fieldType"].__name__

    if rank == 0:
        print()
        print("=" * 80)
        print(f"Field: {fieldType}, RT degree={degree}")
        print(f"Kwargs: {field_cfg['kWargs']}")
        print("=" * 80)

    if rank == 0:
        rng = np.random.default_rng(seed=SEED) 
        mc_points = rng.uniform(low=-1.0, high=1.0, size=(N_ref, stochDim))
    else:
        mc_points = None

    mc_points = comm.bcast(mc_points, root=0)

    all_mc_sizes = sorted(set(mcSamples + [N_ref]))

    if rank == 0:
        print(f"[MC] nested sequence up to N_ref={N_ref}", flush=True)

    mc_results = evalMC(
        field_cfg=field_cfg,
        degree=degree,
        points=mc_points,
        sample_sizes=all_mc_sizes,
    )

    ref_result = mc_results[N_ref]
    referenceMean = ref_result["mean"]

    append_result_rows(
        rows_ref,
        result=ref_result,
        fieldType=fieldType,
        method="MCReference",
        degree=degree,
        N=N_ref,
        scLevel=np.nan,
        referenceMean=referenceMean,
        fieldCase=stochDim,    
    )

    for N in mcSamples:
        append_result_rows(
            rows,
            result=mc_results[N],
            fieldType=fieldType,
            method="MC",
            degree=degree,
            N=N,
            scLevel=np.nan,
            referenceMean=referenceMean,
            fieldCase=stochDim,    
        )

    for sc_cfg in sc_methods:


        if rank == 0:
            print()
            print("-" * 80)
            print(f"[SC] method={sc_cfg['method']}, levels={scLevels}")
            print("-" * 80)

        sc_result = evalSC(
            field_cfg=field_cfg,
            degree=degree,
            levels=np.arange(1, sclevel[stochDim] + 1),
            gridType=sc_cfg["gridType"],
            rule=sc_cfg.get("rule"),
            localDegree=sc_cfg.get("localDegree", 0),
        )

        append_sc_result_rows(
            rows=rows,
            sc_result=sc_result,
            fieldType=fieldType,
            method=sc_cfg["method"],
            degree=degree,
            referenceMean=referenceMean,
            fieldCase=stochDim,    
        )

    completed_outer += 1


# ==========================================================
# Save
# ==========================================================

if rank == 0:
    df_ref = pd.DataFrame(rows_ref)
    df = pd.DataFrame(rows)

    output_dir = Path(__file__).resolve().parent / "pklData"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"DarcyUQMutliphaseDimRT{degreeRT}L{meshLevel}.pkl"

    save_dict = {
        "df": df,
        "df_ref": df_ref,
        "settings": {
            "stochDim": [
                {
                    "fieldType": cfg["fieldType"],
                    "dim": cfg["dim"],
                }
                for cfg in field_configs
            ],
            "mu": mu,
            "corrLength": corrLength,
            "sigma": sigma,
            "numberOfPhases": numberOfPhases,
            "field_configs": [
                {
                    "fieldType": cfg["fieldType"],
                    "kWargs": cfg["kWargs"],
                }
                for cfg in field_configs
            ],
            "meshLevel": meshLevel,
            "degreeRT": degreeRT,
            "elementType": elementType,
            "N_ref": N_ref,
            "mcSamples": mcSamples,
            "scLevels": scLevels,
            "sc_methods": sc_methods,
            "qoiNames": qoiNamesDarcy,
            "qoi_indices": qoi_indices,
            "mpi_size": size,
        },
    }

    pd.to_pickle(save_dict, output_path)

    print()
    print("=" * 80)
    print(f"Saving completed: {output_path}")
    print(f"df shape:     {df.shape}")
    print(f"df_ref shape: {df_ref.shape}")
    print("=" * 80)


# ==========================================================
# Total runtime
# ==========================================================

if rank == 0:
    script_end_dt = datetime.now()
    total_seconds = time.perf_counter() - script_start_perf

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print()
    print("=" * 80)
    print(f"Started at : {script_start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Finished at: {script_end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total runtime: {hours}h {minutes}min {seconds}s")
    print("=" * 80)