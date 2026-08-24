from permeability.lognormalField import LognormalField
import numpy as np

class LognormalInclusion(LognormalField):
    """
    Lognormal field with two fixed square inclusions:
        - top-left
        - bottom-right

    mode="predefined":
        Inside each square, set a predefined constant value.

    mode="max":
        Inside each square, set a constant value equal to the maximum
        of the background lognormal field inside that square.

    Outside the squares:
        K(x,y;xi) = K_lognormal(x,y;xi)

    The interfaces are fixed in both cases.
    """

    def __init__(
        self,
        dim,
        corrLength=0.1,
        mu=1.0,
        sigma=0.25,
        domain=((0.0, 1.0), (0.0, 1.0)),
        normalize=True,
        mode="predefined",
        inclusionValues=(2.0, 3.0),
        topLeftSquare=((0.10, 0.30), (0.70, 0.90)),
        bottomRightSquare=((0.70, 0.90), (0.10, 0.30)),
    ):
        super().__init__(
            dim=dim,
            corrLength=corrLength,
            mu=mu,
            sigma=sigma,
            domain=domain,
            normalize=normalize,
        )

        if mode not in ["predefined", "max"]:
            raise ValueError("mode must be either 'predefined' or 'max'.")

        self.mode = mode
        self.inclusionValues = np.asarray(inclusionValues, dtype=float)

        if len(self.inclusionValues) != 2:
            raise ValueError("inclusionValues must have length 2.")

        self.topLeftSquare = topLeftSquare
        self.bottomRightSquare = bottomRightSquare
        self.squares = [topLeftSquare, bottomRightSquare]

        self.isDiscontinuous = True


    def square_mask(self, x, y, square):
        (x0, x1), (y0, y1) = square

        return (
            (x >= x0)
            & (x <= x1)
            & (y >= y0)
            & (y <= y1)
        )

    def maxVal(self, square, xi=None):
        (x0, x1), (y0, y1) = square

        xs = np.linspace(x0, x1, 100)
        ys = np.linspace(y0, y1, 100)

        X, Y = np.meshgrid(xs, ys, indexing="ij")

        values = super().evaluate(X, Y, xi=xi)

        return float(np.max(values))

    def evaluate(self, x, y, xi=None):
        values = super().evaluate(x, y, xi=xi)
        values = np.array(values, copy=True)

        xArr = np.asarray(x)
        yArr = np.asarray(y)

        for square_id, square in enumerate(self.squares):
            mask = self.square_mask(xArr, yArr, square)

            if not np.any(mask):
                continue

            if self.mode == "predefined":
                inclusion_value = self.inclusionValues[square_id]

            elif self.mode == "max":
                inclusion_value = self.maxVal(square, xi=xi)

            values[mask] = inclusion_value

        return values