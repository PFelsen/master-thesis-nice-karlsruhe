from permeability.lognormalField import LognormalField
import numpy as np

class LognormalMovingInclusion(LognormalField):
    """
    Lognormal background field with one moving square inclusion.

    Outside the square:
        K(x,y;xi) = K_lognormal(x,y;xi)

    Inside the square:
        K(x,y;xi) = inclusionValue

    The square moves according to xi:
        center_x(xi) = center0_x + maxShift_x * xi[0]
        center_y(xi) = center0_y + maxShift_y * xi[1]

    The inclusion value is fixed and prescribed.
    """

    def __init__(
        self,
        dim,
        corrLength=0.1,
        mu=1.0,
        sigma=0.25,
        domain=((0.0, 1.0), (0.0, 1.0)),
        normalize=True,
        inclusionValue=3.0,
        center0=(0.5, 0.5),
        squareSize=0.20,
        maxShift=(0.40, 0.40),
        keepInside=True,
    ):
        super().__init__(
            dim=dim,
            corrLength=corrLength,
            mu=mu,
            sigma=sigma,
            domain=domain,
            normalize=normalize,
        )

        if dim < 2:
            raise ValueError("Need dim >= 2 because xi[0], xi[1] move the square.")

        self.inclusionValue = float(inclusionValue)
        self.center0 = np.asarray(center0, dtype=float)
        self.squareSize = float(squareSize)
        self.maxShift = np.asarray(maxShift, dtype=float)
        self.keepInside = bool(keepInside)

        self.isDiscontinuous = True

    def moving_center(self, xi=None):
        if xi is None:
            xi = self.xi

        xi = np.asarray(xi, dtype=float)

        cx = self.center0[0] + self.maxShift[0] * xi[0]
        cy = self.center0[1] + self.maxShift[1] * xi[1]

        if self.keepInside:
            x0, x1 = self.domain[0]
            y0, y1 = self.domain[1]

            half = 0.5 * self.squareSize

            cx = np.clip(cx, x0 + half, x1 - half)
            cy = np.clip(cy, y0 + half, y1 - half)

        return cx, cy

    def square_bounds(self, xi=None):
        cx, cy = self.moving_center(xi=xi)

        half = 0.5 * self.squareSize

        xmin = cx - half
        xmax = cx + half
        ymin = cy - half
        ymax = cy + half

        return xmin, xmax, ymin, ymax

    def square_mask(self, x, y, xi=None):
        xArr = np.asarray(x)
        yArr = np.asarray(y)

        xmin, xmax, ymin, ymax = self.square_bounds(xi=xi)

        return (
            (xArr >= xmin)
            & (xArr <= xmax)
            & (yArr >= ymin)
            & (yArr <= ymax)
        )

    def evaluate(self, x, y, xi=None):
        values = super().evaluate(x, y, xi=xi)
        values = np.array(values, copy=True)

        mask = self.square_mask(x, y, xi=xi)

        if np.any(mask):
            values[mask] = self.inclusionValue

        return values