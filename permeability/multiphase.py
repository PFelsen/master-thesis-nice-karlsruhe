from permeability.lognormalField import LognormalField
import numpy as np


class MultiphaseField(LognormalField):
    """
    Multiphase field obtained by thresholding a lognormal Fourier field.

    Inherits from LognormalField, so the underlying field is evaluated by
    super().evaluate(x, y, xi).
    """

    def __init__(
        self,
        dim,
        numberOfPhases,
        phaseValues=None,
        phaseBoundaries=None,
        corrLength=0.1,
        mu=1.0,
        sigma=0.25,
        domain=((0.0, 1.0), (0.0, 1.0)),
        normalize=True
    ):
        if numberOfPhases < 2:
            raise ValueError("numberOfPhases must be at least 2.")

        super().__init__(
            dim=dim,
            corrLength=corrLength,
            mu=mu,
            sigma=sigma,
            domain=domain,
            normalize=normalize,
        )

        self.numberOfPhases = numberOfPhases

        self.xGrid = np.linspace(domain[0][0], domain[0][1], 200)
        self.yGrid = np.linspace(domain[1][0], domain[1][1], 200)

        self.isDiscontinuous = True

        if phaseValues is None:
            self.phaseValues = None
        else:
            self.phaseValues = np.asarray(phaseValues, dtype=float)
            if len(self.phaseValues) != numberOfPhases:
                raise ValueError("phaseValues must have length numberOfPhases.")

        if phaseBoundaries is None:
            self.phaseBoundaries = None
        else:
            self.phaseBoundaries = np.asarray(phaseBoundaries, dtype=float)
            if len(self.phaseBoundaries) != numberOfPhases - 1:
                raise ValueError("phaseBoundaries must have length numberOfPhases - 1.")

    def lognormalField(self, x, y, xi=None):
        return super().evaluate(x, y, xi=xi)

    def estimateBoundaries(self, xi=None):
        X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
        values = self.lognormalField(X, Y, xi=xi)

        qs = np.linspace(0.0, 1.0, self.numberOfPhases + 1)[1:-1]
        return np.quantile(values, qs)

    def evaluate(self, x, y, xi=None):
        values = self.lognormalField(x, y, xi=xi)

        if self.phaseBoundaries is None:
            boundaries = self.estimateBoundaries(xi=xi)
        else:
            boundaries = self.phaseBoundaries

        phaseIndex = np.digitize(values, boundaries)

        if self.phaseValues is not None:
            return self.phaseValues[phaseIndex]

        fieldMin = np.min(values)
        fieldMax = np.max(values)

        intervalLimits = np.concatenate(([fieldMin], boundaries, [fieldMax]))
        autoPhaseValues = intervalLimits[1:]

        return autoPhaseValues[phaseIndex]
    
    

