from permeability.abstractRandomField import RandomField
import numpy as np
from matplotlib.path import Path


class RandomShapeField(RandomField):

    def __init__(
        self,
        dim=15,
        backgroundValue=2.0,
        inclusionValue=4.0,
        center=(0.5, 0.5),
        baseRadius=0.25,
        deformationStrength=0.30,
    ):
        self.dim = dim
        self.backgroundValue = backgroundValue
        self.inclusionValue = inclusionValue
        self.center = np.asarray(center)

        self.baseRadius = baseRadius
        self.deformationStrength = deformationStrength

        self.shapeModes = self.buildShapeModes(dim)
        self.shapeWeights = self.buildShapeWeights()

        self.xi = None
        self.isDiscontinuous = True
        

    def buildShapeModes(self, dim):

        modes = []

        for j in range(dim):
            k = j // 2 + 1

            if j % 2 == 0:
                modes.append(("cos", k))
            else:
                modes.append(("sin", k))

        return modes

    def buildShapeWeights(self):

        weights = []

        for kind, k in self.shapeModes:
            weights.append(np.exp(-0.35 * (k - 1)))

        return np.asarray(weights)

    def radius(self, theta, xi=None):
        if xi is None:
            xi = self.xi

        if xi is None:
            raise ValueError("xi is not set. Call setXi(xi) or pass xi explicitly.")

        log_r = np.zeros_like(theta)

        for xi_j, w_j, (kind, k) in zip(
            xi,
            self.shapeWeights,
            self.shapeModes,
        ):

            if kind == "cos":
                basis = np.cos(k * theta)
            else:
                basis = np.sin(k * theta)

            log_r += (
                self.deformationStrength
                * w_j
                * xi_j
                * basis
            )

        return self.baseRadius * np.exp(log_r)

    def boundary(self, xi=None, nPoints=600):

        theta = np.linspace(
            0,
            2 * np.pi,
            nPoints,
            endpoint=False,
        )

        r = self.radius(theta, xi=xi)

        x = self.center[0] + r * np.cos(theta)
        y = self.center[1] + r * np.sin(theta)

        return np.column_stack((x, y))

    def inclusionMask(self, X, Y, xi=None):

        boundary = self.boundary(xi=xi)

        path = Path(boundary)

        pts = np.column_stack(
            (
                X.ravel(),
                Y.ravel(),
            )
        )

        mask = path.contains_points(pts)

        return mask.reshape(X.shape)

    def evaluate(self, X, Y, xi=None):
        if xi is not None:
            self.xi = np.asarray(xi, dtype=float)

        K = np.full_like(np.asarray(X), self.backgroundValue, dtype=float)

        mask = self.inclusionMask(X, Y, xi=xi)

        K = K.copy()
        K[mask] = self.inclusionValue

        return K
    
