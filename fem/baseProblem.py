from abc import ABC, abstractmethod


class BaseProblem(ABC):
    """
    Abstract interface expected by the UQ layer.

    SparseGridSC only needs:
        problem.qoi(xi) -> numpy array

    Concrete subclasses may also expose solve(), printInfo(), etc.
    """

    @abstractmethod
    def build(self):
        """
        Build all deterministic, xi-independent FEM data once.
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self, xi=None):
        """
        Solve the deterministic PDE for a given stochastic input xi.
        """
        raise NotImplementedError

    @abstractmethod
    def qoi(self, xi):
        """
        Return the quantity-of-interest vector for stochastic input xi.
        """
        raise NotImplementedError
