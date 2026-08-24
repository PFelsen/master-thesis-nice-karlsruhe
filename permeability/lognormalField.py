from permeability.abstractRandomField import RandomField
import numpy as np

class LognormalField(RandomField):
    """
    Lognormal random permeability field based on a truncated
    Karhunen–Loève-type expansion.

    The underlying Gaussian field is given by

        G(x, y; ξ) = μ + σ Σ_i c_i ξ_i φ_i(x, y),

    where φ_i are cosine basis functions and c_i are
    exponentially decaying coefficients. The permeability field is

        K(x, y; ξ) = exp(G(x, y; ξ)).
    """

    def __init__(self, dim, corrLength=0.1, mu=1.0, sigma=0.25, domain=((0.0, 1.0), (0.0, 1.0)), normalize=True):
        """
        Initialize a lognormal random field.

        Parameters
        dim : int
            Number of terms retained in the expansion.
        corrLength : float, default=0.2
            Correlation length controlling the decay of the expansion coefficients.
        mu : float, default=0.0
            Mean of the underlying Gaussian field.
        sigma : float, default=1.0
            Standard deviation of the underlying Gaussian field.
        domain : tuple, default=((0.0, 1.0), (0.0, 1.0))
            Spatial domain given as ((xmin, xmax), (ymin, ymax)).
        normalize : bool, default=True
            If True, normalize the expansion coefficients such that the variance of the Gaussian field is independent of dim.
        """
        self.dim = dim
        self.corrLength = corrLength
        self.mu = mu
        self.sigma = sigma
        self.domain = domain
        self.normalize = normalize

        self.xi = None
        self.isDiscontinuous = False


        self.modes = self.buildModes(dim)
        self.coeffs = self.buildCoeffs(self.modes)


    def buildModes(self, dim):
        """
        Construct the Fourier modes used in the expansion.

        Parameters
        dim : int
            Number of modes to generate.

        Returns
        list of tuple[int, int]
            List of Fourier mode indices (m, n), ordered by increasing spatial frequency.
        """
        max_freq = int(np.ceil(np.sqrt(dim))) + 3

        modes = [
            (m, n)
            for m in range(max_freq)
            for n in range(max_freq)
            if not (m == 0 and n == 0)
        ]

        modes.sort(key=lambda mn: mn[0]**2 + mn[1]**2)

        return modes[:dim]


    def buildCoeffs(self, modes):
        """
        Construct the expansion coefficients.

        Parameters
        modes : list of tuple[int, int]
            Fourier mode indices returned by buildModes.

        Returns
        numpy.ndarray
            One-dimensional array containing the coefficient associated with each mode.
        """
        coeffs = []

        for m, n in modes:
            l = m**2 + n**2
            c = np.exp(-0.5 * (np.pi * self.corrLength) ** 2 * l) # Exponential decay
            coeffs.append(c)

        coeffs = np.asarray(coeffs, dtype=float)

        # Normalize. Otherwise increasing dimension would increase variance.
        if self.normalize:
            # xi ~ U[-1, 1] has variance 1/3.
            variance = np.sum(coeffs**2) / 3.0
            if variance > 1e-14:
                coeffs = coeffs / np.sqrt(variance)

        return coeffs


    def gaussian_field(self, x, y, xi=None):
        """
        Evaluate the underlying Gaussian random field.

        Parameters
        x : array_like
            x-coordinates.
        y : array_like
            y-coordinates.
        xi : array_like, optional
            Random parameter vector. If None, the internally stored realization self.xi is used.

        Returns
        numpy.ndarray
            Values of the Gaussian random field evaluated at (x, y).
        """
        if xi is None:
            xi = self.xi

        if xi is None:
            raise ValueError("xi is not set. Call setXi(xi) or pass xi explicitly.")

        xi = np.asarray(xi, dtype=float)

        xArr = np.asarray(x)
        yArr = np.asarray(y)

        x0, x1 = self.domain[0]
        y0, y1 = self.domain[1]

        # Rescale corrdinates, e.g. normalize
        xHat = (xArr - x0) / (x1 - x0) 
        yHat = (yArr - y0) / (y1 - y0)

        G = np.zeros_like(xHat, dtype=float)

        for xi_i, c_i, (m, n) in zip(xi, self.coeffs, self.modes):
            phi = np.cos(np.pi * m * xHat) * np.cos(np.pi * n * yHat)
            G += c_i * xi_i * phi

        return self.mu + self.sigma * G


    def evaluate(self, x, y, xi=None):
        """
        Evaluate the lognormal random field.

        Parameters
        x : array_like
            x-coordinates.
        y : array_like
            y-coordinates.
        xi : array_like, optional
            Random parameter vector. If None, the internally stored realization self.xi is used.

        Returns
        numpy.ndarray
            Values of the lognormal random field evaluated at (x, y).
        """
        G = self.gaussian_field(x, y, xi=xi)
        return np.exp(G)


