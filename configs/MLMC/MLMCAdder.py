import time
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI
from fenics import Constant, Expression

from fem.transportProblem import Transport

from permeability.lognormalField import LognormalField
from permeability.lognormalInclusion import LognormalInclusion
from permeability.movingInclusion import LognormalMovingInclusion
from permeability.multiphase import MultiphaseField
from permeability.randomShapeField import RandomShapeField


# ==========================
# MPI and output
# ==========================

comm = MPI.COMM_WORLD # Global MPI communicator containing every process started with mpirun
rank = comm.Get_rank() # MPI rank of the current process
size = comm.Get_size() # Total number of MPI processes

script_folder = Path(__file__).resolve().parent
output_folder = script_folder / "pklData"
output_file = output_folder / "mlmc_verification.pkl"


# ==========================
# MLMC settings
# ==========================

SAMPLES_PER_CONFIGURATION = 1000 # Number of samples wanted for each configuration

theta = 1.0 # Implicit Euler
T = 0.5

degrees = [0, 1]

field_types = [
    "LognormalField",
    "LognormalInclusion",
    "LognormalMovingInclusion",
    "RandomShapeField",
    "MultiphaseField",
]

qoi_names = [
    "outflow_first_moment_normalized",
    "mean_c_box_time_average",
    "L2normSquared",
]

levels = [3, 4, 5, 6, 7, 8]


# Cartesian product of permeability-field types and transport degrees.
# This creates: 5 field types × 2 degrees = 10 configurations.
configurations = [
    (field_type, degree)
    for field_type in field_types
    for degree in degrees
]

qoi_box = (
    (0.375, 0.625),
    (0.000, 0.250),
)

c0 = Expression(
    """
    x[0] >= x_left &&
    x[0] <= x_right &&
    x[1] >= y_bottom &&
    x[1] <= y_top
    ? c_value : 0.0
    """,
    degree=0,
    x_left=0.125,
    x_right=0.875,
    y_bottom=0.75,
    y_top=0.875,
    c_value=5.0,
)


# ==========================
# Random-field settings
# ==========================

stoch_dim = 15
corr_length = 0.1
mu = 1.0
sigma = 0.25

number_of_phases = 2
phase_values = [2.0, 3.0]

inclusion_values = (2.0, 3.0)
moving_inclusion_value = 3.0

base_seed = 12345


def make_field(field_type):
    if field_type == "LognormalField":
        return LognormalField(
            dim=stoch_dim,
            corrLength=corr_length,
            mu=mu,
            sigma=sigma,
        )

    if field_type == "LognormalInclusion":
        return LognormalInclusion(
            dim=stoch_dim,
            corrLength=corr_length,
            mu=mu,
            sigma=sigma,
            inclusionValues=inclusion_values,
        )

    if field_type == "LognormalMovingInclusion":
        return LognormalMovingInclusion(
            dim=stoch_dim,
            corrLength=corr_length,
            mu=mu,
            sigma=sigma,
            inclusionValue=moving_inclusion_value,
            maxShift=(0.40, 0.40),
        )

    if field_type == "MultiphaseField":
        return MultiphaseField(
            dim=stoch_dim,
            numberOfPhases=number_of_phases,
            phaseValues=phase_values,
            phaseBoundaries=[np.exp(mu)],
            corrLength=corr_length,
            mu=mu,
            sigma=sigma,
        )

    if field_type == "RandomShapeField":
        return RandomShapeField(
            dim=stoch_dim,
            inclusionValue=3.0,
            backgroundValue=2.0,
            center=(0.5, 0.5),
            baseRadius=0.25,
            deformationStrength=0.5,
        )

    raise ValueError(f"Unknown field type: {field_type}")


def sample_xi(sample_id):
    """Return a reproducible stochastic input for a sample ID."""
    rng = np.random.default_rng(base_seed + int(sample_id))
    return rng.uniform(-1.0, 1.0, stoch_dim)


def make_problem(field_type, degree, level):
    return Transport(
        permeabilityField=make_field(field_type),
        meshLevelTransport=level,
        degreeTransport=degree,
        elementTypeTransport="DG",
        meshLevelDarcy=level,
        degreeDarcy=degree + 1,
        elementTypeDarcy="RT",
        transportSolver="gmres",
        transportPrecon="ilu",
        darcySolver="mumps",
        darcyPrecon=None,
        cInflow=Constant(0.0),
        c0=c0,
        theta=theta,
        T=T,
        nSteps=4 * 2**level,
        adaptCfl=False,
        qoiBox=qoi_box,
        relTol=1e-10,
        absTol=1e-12,
        maxIter=3000,
        plotSolution=False,
        storeSolutionHistory=False,
        linearSolverVerbose=False,
        femTransportVerbose=False,
        femDarcyVerbose=False,
    )


def evaluate(problem, xi):
    start = time.perf_counter()
    qoi = np.asarray(problem.qoi(xi), dtype=float)
    runtime = time.perf_counter() - start

    info = problem.info["transport"]

    return {
        "qoi": qoi,
        "runtime": runtime,
        "mesh_level": info["mesh_level"],
        "dofs": info["dofs"],
        "h_min": info["h_min"],
        "h_max": info["h_max"],
        "dt": info["dt"],
        "n_steps": info["n_steps"],
        "cfl": info["cfl"],
        "linear_iterations": info["nIteration"],
        "linear_residual": info["linear_residual"],
        "mass_loss": info["mass_loss"],
        "relative_mass_loss": info["rel_mass_loss"],
    }


def make_result_row(
    field_type,
    degree,
    level,
    sample_id,
    xi,
    fine,
    coarse,
):
    if coarse is None: # There is no lower discretization below the first MLMC level
        coarse_qoi = np.zeros_like(fine["qoi"])
        delta_qoi = fine["qoi"]

        coarse_mesh_level = np.nan
        coarse_dofs = np.nan
        coarse_h_min = np.nan
        coarse_h_max = np.nan
        coarse_dt = np.nan
        coarse_n_steps = np.nan
        coarse_runtime = np.nan
        coarse_cfl = np.nan
        coarse_linear_iterations = np.nan
        coarse_linear_residual = np.nan
        coarse_mass_loss = np.nan
        coarse_relative_mass_loss = np.nan
        coarse_reused = False
    else:
        coarse_qoi = coarse["qoi"]
        delta_qoi = fine["qoi"] - coarse_qoi

        coarse_mesh_level = coarse["mesh_level"]
        coarse_dofs = coarse["dofs"]
        coarse_h_min = coarse["h_min"]
        coarse_h_max = coarse["h_max"]
        coarse_dt = coarse["dt"]
        coarse_n_steps = coarse["n_steps"]
        coarse_runtime = coarse["runtime"]
        coarse_cfl = coarse["cfl"]
        coarse_linear_iterations = coarse["linear_iterations"]
        coarse_linear_residual = coarse["linear_residual"]
        coarse_mass_loss = coarse["mass_loss"]
        coarse_relative_mass_loss = coarse["relative_mass_loss"]
        coarse_reused = True

    row = {
        "field_type": field_type,
        "degree_transport": degree,
        "degree_darcy": degree + 1,
        "mlmc_level": level,
        "sample_id": int(sample_id),
        "mpi_rank": rank,
        "target_samples_per_configuration": SAMPLES_PER_CONFIGURATION,
        "theta": theta,
        "T": T,
        "fine_mesh_level": fine["mesh_level"],
        "coarse_mesh_level": coarse_mesh_level,
        "fine_dofs": fine["dofs"],
        "coarse_dofs": coarse_dofs,
        "fine_h_min": fine["h_min"],
        "fine_h_max": fine["h_max"],
        "coarse_h_min": coarse_h_min,
        "coarse_h_max": coarse_h_max,
        "fine_dt": fine["dt"],
        "coarse_dt": coarse_dt,
        "fine_n_steps": fine["n_steps"],
        "coarse_n_steps": coarse_n_steps,
        "fine_runtime": fine["runtime"],
        "coarse_runtime": coarse_runtime,
        "coupled_runtime": fine["runtime"],
        "coarse_reused": coarse_reused,
        "has_coarse_level": coarse is not None,
        "fine_cfl": fine["cfl"],
        "coarse_cfl": coarse_cfl,
        "fine_linear_iterations": fine["linear_iterations"],
        "coarse_linear_iterations": coarse_linear_iterations,
        "fine_linear_residual": fine["linear_residual"],
        "coarse_linear_residual": coarse_linear_residual,
        "fine_mass_loss": fine["mass_loss"],
        "coarse_mass_loss": coarse_mass_loss,
        "fine_relative_mass_loss": fine["relative_mass_loss"],
        "coarse_relative_mass_loss": coarse_relative_mass_loss,
    }

    for j, value in enumerate(xi):
        row[f"xi_{j}"] = value

    for j, qoi_name in enumerate(qoi_names):
        row[f"fine_{qoi_name}"] = fine["qoi"][j]
        row[f"coarse_{qoi_name}"] = (
            np.nan if coarse is None else coarse_qoi[j]
        )
        row[f"delta_{qoi_name}"] = delta_qoi[j]

    return row


def progress_bar(completed, total, width=32):
    fraction = min(max(completed / total, 0.0), 1.0)
    filled = int(round(width * fraction)) # Convert the fraction into a number of filled characters
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ==========================
# Run all configurations
# ==========================

if rank == 0:
    rows = []

    # Stores fine results so level L can reuse level L-1 as its coarse result.
    fine_results = {}

    total_rows = (
        len(levels)
        * len(configurations)
        * SAMPLES_PER_CONFIGURATION
    )

    print(
        f"Starting {total_rows} samples on {size} MPI ranks.",
        flush=True,
    )
else:
    rows = None
    fine_results = None
    total_rows = None

comm.Barrier() # Wait until every MPI process has finished initialization
start_time = time.perf_counter()
completed_rows = 0

for level_index, level in enumerate(levels):
    for field_type, degree in configurations:
        fine_problem = make_problem(field_type, degree, level)

        for batch_start in range(0, SAMPLES_PER_CONFIGURATION, size):
            batch_sample_ids = list(
                range(
                    batch_start,
                    min(batch_start + size, SAMPLES_PER_CONFIGURATION),
                )
            )

            if rank == 0:
                if level_index == 0:
                    coarse_batch = [None] * len(batch_sample_ids)
                else:
                    coarse_level = levels[level_index - 1]
                    coarse_batch = [
                        fine_results[
                            (
                                field_type,
                                degree,
                                coarse_level,
                                sample_id,
                            )
                        ]
                        for sample_id in batch_sample_ids
                    ]
            else:
                coarse_batch = None

            coarse_batch = comm.bcast(coarse_batch, root=0)

            if rank < len(batch_sample_ids):
                sample_id = batch_sample_ids[rank]
                coarse = coarse_batch[rank]

                xi = sample_xi(sample_id)
                fine = evaluate(fine_problem, xi)

                local_result = {
                    "row": make_result_row(
                        field_type=field_type,
                        degree=degree,
                        level=level,
                        sample_id=sample_id,
                        xi=xi,
                        fine=fine,
                        coarse=coarse,
                    ),
                    "fine": fine,
                }
            else:
                local_result = None

            gathered_results = comm.gather(local_result, root=0)

            if rank == 0:
                for result in gathered_results:
                    if result is None:
                        continue

                    row = result["row"]
                    fine = result["fine"]

                    rows.append(row)
                    fine_results[
                        (
                            field_type,
                            degree,
                            level,
                            int(row["sample_id"]),
                        )
                    ] = fine

                    completed_rows += 1

                elapsed = time.perf_counter() - start_time
                seconds_per_row = elapsed / completed_rows
                remaining_rows = total_rows - completed_rows
                eta = seconds_per_row * remaining_rows

                bar = progress_bar(completed_rows, total_rows)

                print(
                    f"{bar} "
                    f"{completed_rows}/{total_rows} rows | "
                    f"level={level} | "
                    f"{field_type}/DG{degree} | "
                    f"ETA={format_seconds(eta)}",
                    flush=True,
                )

        del fine_problem

    if rank == 0:
        print(
            f"Completed level {level}: "
            f"{len(configurations) * SAMPLES_PER_CONFIGURATION} rows",
            flush=True,
        )


if rank == 0:
    key_columns = [
        "field_type",
        "degree_transport",
        "mlmc_level",
        "sample_id",
    ]

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values(key_columns).reset_index(drop=True)

    output_folder.mkdir(parents=True, exist_ok=True)
    dataframe.to_pickle(output_file)

    elapsed = time.perf_counter() - start_time

    print("=" * 80, flush=True)
    print(f"Saved {len(dataframe)} rows.", flush=True)
    print(f"Elapsed time: {format_seconds(elapsed)}", flush=True)
    print(f"Output file: {output_file}", flush=True)