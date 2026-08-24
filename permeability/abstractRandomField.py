import numpy as np
from fenics import FunctionSpace, Function
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod



class RandomField(ABC):
    """
    Abstract base class for stochastic random fields in a KL-expansion type style.
    """

    def sampleXi(self, seed=None, distribution ="uniform"):
        """
        Sample a random vector of length self.dim with given distribution. 

        Parameters: 
        seed : int, optional
            Seed for the random number generator. If None, a random seed is used.
        distribution : {"uniform", "normal"}, default="uniform"
            Distribution used to sample the vector.

        Returns:
        numpy.ndarray
            One-dimensional array of shape (self.dim, ) containing the sampled value.
        """
        rng = np.random.default_rng(seed)

        if distribution == "uniform":
            return rng.uniform(-1.0, 1.0, self.dim)
        
        if distribution == "normal":
            return rng.normal(0.0, 1.0, self.dim)
        
        raise ValueError("Distribution must be 'uniform' or 'normal'.")
    

    def setXi(self, xi):
        """
        Set the random parameter vector. 
        Used in UQ runs, where the field is generated for prescribed realisations of the random parameters.
        
        Parameters: 
        xi : array
            One dimensional array of length self.dim containing the random-parameter vector.
        """
        xi = np.asarray(xi, dtype=float)

        if xi.shape[0] != self.dim:
            raise ValueError(f"Expected xi of length {self.dim}, got {xi.shape[0]}.")

        self.xi = xi


    def adaptToMesh(self, mesh, elementType="CG", degree=1):
        """
        Interpolate the random field onto a FEniCS function space.

        Parameters
        mesh : dolfin.Mesh
            Mesh on which the field is to be represented.
        elementType : str, default="CG"
            Finite element family used to construct the function space
            (e.g. ``"CG"`` or ``"DG"``).
        degree : int, default=1
            Polynomial degree of the finite element space.

        Returns
        dolfin.Function
            FEniCS function representing the random field on the specified
            function space.
        """

        VK = FunctionSpace(mesh, elementType, degree)

        coords = VK.tabulate_dof_coordinates().reshape((-1, 2))

        KValues = self.evaluate(coords[:, 0], coords[:, 1])

        K = Function(VK)
        K.vector().set_local(KValues)
        K.vector().apply("insert")

        return K


    def plot(self, title=None , colorbarLabel="K(x,y)"):
        """
        Creates a generic plot for the random field. 

        Parameters: 
        title : string
            Title for the plot.
        colorbarLabel : string
            Label for the colorbar

        Returns:
        numpy.ndarray
            One-dimensional array of shape (self.dim, ) containing the sampled value.
        """
        x = np.linspace(self.domain[0][0], self.domain[0][1], 256)
        y = np.linspace(self.domain[1][0], self.domain[1][1], 256)

        X, Y = np.meshgrid(x, y, indexing="ij")

        field = self.evaluate(X,Y)

        extent = (x[0], x[-1], y[0], y[-1])

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(field.T, extent=extent, origin="lower", aspect="equal", cmap="viridis", interpolation="nearest")

        fig.colorbar(im, ax=ax, label=colorbarLabel)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_title(title)

        fig.tight_layout()

        return fig, ax
    

    @abstractmethod
    def evaluate(self, X, Y):
        """
        Evaluates the random field at given coordinates.

        Parameters:
        X : array
            Coordinates in X direction
        Y : array
            Coordinates in Y direction
        """
        pass


