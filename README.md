## `permeability`

The permeability/ folder provides stochastic permeability fields for the Darcy and transport problem.

All fields inherit from ’RandomField’, which defines the common interface for sampling and setting the stochastic vector as well as evaluation and interpolation.

| Class | Description |
|---|---|
| `LognormalField` | Smooth lognormal field obtained by exponentiating a truncated Fourier/KL-type expansion. |
| `LognormalInclusion` | Lognormal background with two fixed square inclusions of prescribed or field-dependent permeability. |
| `LognormalMovingInclusion` | Lognormal background with a square inclusion whose position depends on the first two stochastic parameters. |
| `MultiphaseField` | Discontinuous field obtained by thresholding a lognormal realization into two or more constant phases. |
| `RandomShapeField` | Piecewise-constant field with a randomly deformed inclusion described by angular Fourier modes. |


## `fem`

The `fem/` folder contains the finite element model problems used by the uncertainty-quantification framework. The implementations use FEniCS for finite element assembly and PETSc for solving the resulting linear systems.

| Class | Description |
|---|---|
| `BaseProblem` | Abstract interface defining the methods `build()`, `solve(xi)`, and `qoi(xi)` expected by the UQ framework. |
| `Darcy` | Solves the elliptic Darcy problem and computes the pressure and Darcy flux. |
| `Transport` | Solves the Darcy-driven linear transport equation using an upwind DG discretisation in space and a theta scheme in time. |

### Darcy problem

The `Darcy` class solves $-\nabla\cdot(\kappa \nabla p)=f,$ and $q=-\kappa \nabla p,$

on the unit square. Pressure values are prescribed at the top and bottom boundaries, while a no-flow condition is imposed on the lateral boundaries.

Two spatial discretisations are supported:

- continuous Lagrange (`CG`) elements for the pressure;
- mixed Raviart--Thomas (`RT`) elements for the flux, coupled with `DG` elements for the pressure.

Discontinuous permeability fields are represented in `DG0`, while smooth fields are represented using continuous Lagrange elements.

The Darcy quantity-of-interest vector contains:

1. total outflow through the bottom boundary;
2. mean flux magnitude;
3. mean pressure in the observation box;
4. pressure $L^2$-norm;
5. flux $L^2$-norm.

### Transport problem

The `Transport` class first solves the Darcy problem and then uses the resulting flux in $\partial_t \rho + \nabla\cdot(\rho q)=0.$

The concentration is discretised with upwind discontinuous Galerkin elements. Time integration is controlled by `theta`; in particular, `theta=1.0` corresponds to implicit Euler.

The class supports CFL estimation, optional time-step adaptation, storage of the solution history, and transport animations. Its quantity-of-interest vector contains the normalized first outflow moment, the time-averaged concentration in an observation box, and the time-integrated squared $L^2$-norm.


## `uq`

The `uq/` folder contains uncertainty-quantification methods for estimating statistics of quantities of interest. Both implementations expect a model problem that provides

```python
problem.qoi(xi)
```

where `xi` is a stochastic parameter vector and the return value is a NumPy array containing one or more quantities of interest.

| Class | Description |
|---|---|
| `MonteCarlo` | MPI-parallel Monte Carlo estimator using uniformly distributed samples in \([-1,1]^d\). |
| `SparseGridSC` | MPI-parallel stochastic collocation based on sparse grids provided by the Tasmanian library. |

### Monte Carlo

The `MonteCarlo` class distributes independent samples across the available MPI ranks and evaluates `problem.qoi(xi)` for each realization. Existing sample points can also be supplied, which is useful for reproducible or nested Monte Carlo experiments.

The method computes:

- the sample mean;
- the unbiased sample variance;
- the standard deviation;
- the variance of the Monte Carlo estimator;
- the estimated stochastic error.



### Sparse-grid stochastic collocation

The `SparseGridSC` class constructs sparse grids using Tasmanian and evaluates the model at the corresponding collocation points. Global and local-polynomial grids are supported. The default global grid uses Clenshaw-Curtis points.

If several nested levels are requested, the model is evaluated only on the finest grid. Statistics for the coarser levels are then recovered using level-dependent quadrature weights.

The method computes:

- the mean;
- the second moment;
- the variance;
- the standard deviation.

Both methods use `mpi4py` to distribute model evaluations across MPI ranks. They can therefore be applied directly to the Darcy and transport problems defined in the `fem/` folder.


## Basic usage

The same model problem can be evaluated deterministically or passed directly to the Monte Carlo and stochastic-collocation methods.

```python
from permeability.lognormalField import LognormalField
from fem.darcyProblem import Darcy
from uq.monteCarlo import MonteCarlo
from uq.sparseGridSC import SparseGridSC


# Permeability field
field = LognormalField(
    dim=15,
    corrLength=0.1,
    mu=1.0,
    sigma=0.25,
)

# Darcy problem
darcy = Darcy(
    permeabilityField=field,
    meshLevel=5,
    degree=1,
    elementType="RT",
)

qoi_names = [
    "outflow",
    "mean_flux",
    "mean_pressure_box",
    "pressure_L2",
    "flux_L2",
]


# Deterministic realization
xi = field.sampleXi(seed=1)

pressure, flux, info = darcy.solve(xi)
darcy_qoi = darcy.qoi(xi)


# Monte Carlo
mc = MonteCarlo(
    problem=darcy,
    qoiNames=qoi_names,
    stochDim=field.dim,
    nSamples=1_000,
)

mc_mean, mc_variance, mc_estimator_variance, mc_error = mc.solve()


# Sparse-grid stochastic collocation
sc = SparseGridSC(
    problem=darcy,
    qoiNames=qoi_names,
    stochDim=field.dim,
    level=[1, 2, 3],
    gridType="level",
    rule="clenshaw-curtis",
)

sc_statistics = sc.solve()
```

Both UQ methods use MPI and can be executed in parallel.
