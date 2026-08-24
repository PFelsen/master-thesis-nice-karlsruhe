"""
Disclaimer:
The AI tool ChatGPT was used for realising the MPI parallel implementation part. 
See lines with # AI written at the end.

"""
import time
import numpy as np
import Tasmanian
from tqdm import tqdm
from mpi4py import MPI


class SparseGridSC:
    """
    MPI-parallel stochastic collocation.

    If level is an int:
        returns mean, secondMoment, variance, std

    If level is a list:
        evaluates only the finest level once and returns all statistics
        for all levels using nested sparse-grid weights.
    """

    def __init__(
        self,
        problem,
        qoiNames,
        stochDim,
        level,
        gridType="level",
        rule="clenshaw-curtis",
        localDegree=0,
        verbose=True,
    ):
        self.problem = problem
        self.qoiNames = qoiNames
        self.stochDim = stochDim
        self.level = level
        self.gridType = gridType
        self.rule = rule
        self.localDegree = localDegree
        self.verbose = verbose

        self.levels = (
            list(level)
            if isinstance(level, (list, tuple, np.ndarray))
            else [int(level)]
        )
        self.levels = sorted([int(l) for l in self.levels])
        self.multiLevel = len(self.levels) > 1
        self.finestLevel = max(self.levels)

        self.comm = MPI.COMM_WORLD # AI
        self.rank = self.comm.Get_rank() # AI
        self.size = self.comm.Get_size() # AI

        self.grid = None
        self.localPoints = None
        self.localWeightsByLevel = None

        self.mean = None
        self.secondMoment = None
        self.variance = None
        self.std = None

        self.info = {}


    def makeGrid(self, level):
        grid = Tasmanian.SparseGrid()

        if self.gridType == "wavelet":
            grid.makeWaveletGrid(
                self.stochDim,
                0,
                int(level),
                self.localDegree,
            )

        elif self.gridType == "local":
            grid.makeLocalPolynomialGrid(
                self.stochDim,
                0,
                int(level),
                self.localDegree,
                "localp",
            )

        else:
            grid.makeGlobalGrid(
                self.stochDim,
                0,
                int(level),
                self.gridType,
                self.rule,
            )

        return grid

    @staticmethod
    def pointKey(x, decimals=14):
        x = np.asarray(x, dtype=float).ravel()
        return tuple(np.round(x, decimals=decimals))

    def buildGrid(self):
        """
        Build finest grid only.

        For each requested level, create a weight vector over the finest
        point set. Points not present at a coarse level get weight zero.
        """

        if self.rank == 0: # Create grid on rank 0
            self.grid = self.makeGrid(self.finestLevel)

            finestPoints = self.grid.getPoints()
            nPoints = finestPoints.shape[0]

            pointToFineId = {self.pointKey(x): i for i, x in enumerate(finestPoints)}

            weightsByLevel = np.zeros((nPoints, len(self.levels)), dtype=float)

            for levelId, lev in enumerate(self.levels):
                gridLev = self.makeGrid(lev)

                pointsLev = gridLev.getPoints()
                weightsLev = gridLev.getQuadratureWeights()
                weightsLev = weightsLev / np.sum(weightsLev)

                for x, w in zip(pointsLev, weightsLev):
                    key = self.pointKey(x)

                    if key not in pointToFineId:
                        raise RuntimeError(
                            f"Level {lev} is not nested in finest level "
                            f"{self.finestLevel}. Rule/grid may not be nested."
                        )

                    fineId = pointToFineId[key]
                    weightsByLevel[fineId, levelId] = w

            remainder = nPoints % self.size

            if remainder != 0:
                nPad = self.size - remainder

                padPoints = np.zeros((nPad, self.stochDim), dtype=float)
                padWeights = np.zeros((nPad, len(self.levels)), dtype=float)

                finestPoints = np.vstack([finestPoints, padPoints])
                weightsByLevel = np.vstack([weightsByLevel, padWeights])
            else:
                nPad = 0

            self.info["levels"] = self.levels
            self.info["finest_level"] = self.finestLevel
            self.info["n_points_original"] = nPoints
            self.info["n_points_padded"] = finestPoints.shape[0]
            self.info["n_padding"] = nPad
            self.info["weight_sum_by_level"] = np.sum(weightsByLevel, axis=0)

            nTotal = finestPoints.shape[0]

        else:
            finestPoints = None
            weightsByLevel = None
            nTotal = None

        nTotal = self.comm.bcast(nTotal, root=0)
        nLocal = nTotal // self.size

        self.localPoints = np.empty((nLocal, self.stochDim), dtype=float)
        self.localWeightsByLevel = np.empty(
            (nLocal, len(self.levels)),
            dtype=float,
        )

        self.comm.Scatter(finestPoints, self.localPoints, root=0) # AI
        self.comm.Scatter(weightsByLevel, self.localWeightsByLevel, root=0) # AI


    def solve(self):
        """
        Evaluate QoIs once on finest grid.

        Mean and second moment are computed by matrix products.

        Variance is computed by the stable weighted formula Var[Q] = E[(Q - E[Q])²]

        """

        self.buildGrid()

        startTime = time.perf_counter()

        activeMask = np.any(self.localWeightsByLevel != 0.0, axis=1)
        activePoints = self.localPoints[activeMask]
        activeWeights = self.localWeightsByLevel[activeMask]

        localQoiValues = []
        localTotal = activePoints.shape[0]

        iterator = activePoints

        if self.verbose and self.rank == 0:
            iterator = tqdm(
                activePoints,
                total=activePoints.shape[0],
                desc="[SC] finest-grid samples",
                unit="sample",
                dynamic_ncols=True,
            )

        for xi in iterator:
            qoiValue = np.asarray(self.problem.qoi(xi), dtype=float)
            localQoiValues.append(qoiValue)


        localNQoi = 0

        if len(localQoiValues) > 0:
            localQoiValues = np.asarray(localQoiValues, dtype=float)
            localNQoi = localQoiValues.shape[1]

        nQoi = self.comm.allreduce(localNQoi, op=MPI.MAX) # AI
        nLevels = len(self.levels)

        if len(localQoiValues) == 0:
            localWeightedSum = np.zeros((nLevels, nQoi), dtype=float)
            localWeightedSecondMoment = np.zeros((nLevels, nQoi), dtype=float)
        else:
            localWeightedSum = activeWeights.T @ localQoiValues
            localWeightedSecondMoment = activeWeights.T @ (localQoiValues**2)

        globalWeightedSum = np.zeros((nLevels, nQoi), dtype=float)
        globalWeightedSecondMoment = np.zeros((nLevels, nQoi), dtype=float)

        self.comm.Allreduce(localWeightedSum, globalWeightedSum, op=MPI.SUM) # AI
        self.comm.Allreduce( # AI
            localWeightedSecondMoment, # AI
            globalWeightedSecondMoment, # AI
            op=MPI.SUM, # AI
        ) # AI

        self.mean = globalWeightedSum
        self.secondMoment = globalWeightedSecondMoment

        # Stable two-pass variance
        if len(localQoiValues) == 0:
            localWeightedVariance = np.zeros((nLevels, nQoi), dtype=float)
        else:
            localWeightedVariance = np.zeros((nLevels, nQoi), dtype=float)

            for levelId in range(nLevels):
                diff = localQoiValues - self.mean[levelId]
                localWeightedVariance[levelId] = (
                    activeWeights[:, levelId] @ (diff**2)
                )

        globalWeightedVariance = np.zeros((nLevels, nQoi), dtype=float)

        self.comm.Allreduce( # AI
            localWeightedVariance, # AI
            globalWeightedVariance, # AI
            op=MPI.SUM, # AI
        ) # AI

        self.variance = np.maximum(globalWeightedVariance, 0.0)
        self.std = np.sqrt(self.variance)

        runtime = time.perf_counter() - startTime


        self.info["runtime"] = runtime
        self.info["n_qoi"] = nQoi
        self.info["second_moment"] = self.secondMoment

        if self.rank == 0 and self.verbose:
            self.printInfo()

        if self.multiLevel:
            return {
                "levels": np.asarray(self.levels),
                "mean": self.mean,
                "secondMoment": self.secondMoment,
                "variance": self.variance,
                "std": self.std,
            }

        return self.mean[0], self.secondMoment[0], self.variance[0], self.std[0]


    def printInfo(self):
        print()
        print("=" * 90)
        print("MPI sparse grid created")
        print("=" * 90)
        print(f"{'MPI ranks':<35} {self.size:>20}")
        print(f"{'Stochastic dimension':<35} {self.stochDim:>20}")
        print(f"{'Levels':<35} {str(self.levels):>20}")
        print(f"{'Finest level':<35} {self.finestLevel:>20}")
        print(f"{'Grid type':<35} {self.gridType:>20}")
        print(f"{'Rule':<35} {self.rule:>20}")
        print(f"{'Original finest-grid points':<35} {self.info['n_points_original']:>20}")
        print(f"{'Padded points':<35} {self.info['n_points_padded']:>20}")
        print(f"{'Padding points':<35} {self.info['n_padding']:>20}")

        print()
        print("Weight sums by level:")
        for lev, weightSum in zip(self.levels, self.info["weight_sum_by_level"]):
            print(f"  level {lev:<5} {weightSum:>20.6e}")

        print()
        print("=" * 90)
        print("MPI stochastic collocation information")
        print("=" * 90)
        print(f"{'Runtime':<35} {self.info['runtime']:>20.3f} s")
        print(f"{'Number of QoIs':<35} {self.info['n_qoi']:>20}")

        qoiNames = self.qoiNames
        if qoiNames is None:
            qoiNames = [f"QoI {i}" for i in range(self.info["n_qoi"])]


        if isinstance(self.levels, int):
            levelId = 0
            lev = self.levels
        else:
            levelId = len(self.levels) - 1
            lev = self.levels[levelId]

        print()
        print("-" * 120)
        print(f"Results for finest level {lev}")
        print("-" * 120)

        print(
            f"{'QoI':<30}"
            f"{'Mean':>18}"
            f"{'Second Moment':>18}"
            f"{'Variance':>18}"
            f"{'Std':>18}"
        )
        print("-" * 120)

        for name, mean, secondMoment, variance, std in zip(
            qoiNames,
            self.mean[levelId],
            self.secondMoment[levelId],
            self.variance[levelId],
            self.std[levelId],
        ):
            print(
                f"{name:<30}"
                f"{mean:>18.8e}"
                f"{secondMoment:>18.8e}"
                f"{variance:>18.8e}"
                f"{std:>18.8e}"
            )

        print("=" * 120)

