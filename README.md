# master-thesis-nice-karlsruhe
ToDo

## permeability  
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

The `Darcy` class solves

\[
-\nabla\cdot(K\nabla p)=f,
\qquad
q=-K\nabla p,
\]

on the unit square. Pressure values are prescribed at the top and bottom boundaries, while a no-flow condition is imposed on the lateral boundaries.

Two spatial discretisations are supported:

- continuous Lagrange (`CG`) elements for the pressure;
- mixed Raviart--Thomas (`RT`) elements for the flux, coupled with `DG` elements for the pressure.

Discontinuous permeability fields are represented in `DG0`, while smooth fields are represented using continuous Lagrange elements.

The Darcy quantity-of-interest vector contains:

1. total outflow through the bottom boundary;
2. mean flux magnitude;
3. mean pressure in the observation box;
4. pressure \(L^2\)-norm;
5. flux \(L^2\)-norm.

### Transport problem

The `Transport` class first solves the Darcy problem and then uses the resulting flux in

\begin{equation*}
\partial_t \rho+\nabla\cdot(\rho q)=0.
\end{equation*}

The concentration is discretised with upwind discontinuous Galerkin elements. Time integration is controlled by `theta`; in particular, `theta=1.0` corresponds to implicit Euler.

The class supports CFL estimation, optional time-step adaptation, storage of the solution history, and transport animations. Its quantity-of-interest vector contains the normalized first outflow moment, the time-averaged concentration in an observation box, and the time-integrated squared \(L^2\)-norm.

### Basic usage

```python
field = LognormalField(
    dim=15,
    corrLength=0.1,
    mu=1.0,
    sigma=0.25,
)

xi = field.sampleXi(seed=1)

darcy = Darcy(
    permeabilityField=field,
    meshLevel=5,
    degree=1,
    elementType="RT",
)

pressure, flux, info = darcy.solve(xi)
darcy_qoi = darcy.qoi(xi)
```

