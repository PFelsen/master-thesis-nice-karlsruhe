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

