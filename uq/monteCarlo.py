import time
import numpy as np
from mpi4py import MPI

"""
Disclaimer:
The AI tool ChatGPT was used for realising the MPI parallel implementation part. 
See lines with # AI written at the end.
"""


class MonteCarlo:
    """
    MPI-parallel Monte Carlo method.
    Assumes mapping qoi(xi) -> np.array([...])
    """

    def __init__(
        self,
        problem,
        qoiNames,
        stochDim,
        nSamples,
        points=None,
        mcVerbose=True
    ):
        self.problem = problem
        self.qoiNames = qoiNames
        self.stochDim = stochDim
        self.nSamples = nSamples
        self.mcVerbose = mcVerbose

        self.comm = MPI.COMM_WORLD # AI
        self.rank = self.comm.Get_rank() # AI
        self.size = self.comm.Get_size() # AI

        self.points = points
        self.weights = None
        self.npad = None

        self.localPoints = None
        self.localWeights = None

        self.mean = None
        self.secondMoment = None
        self.variance = None
        self.std = None

        self.estimatorVariance = None  # Var(mu_hat) = Var(Q) / N
        self.stochasticError = None    # sqrt(Var(Q) / N)

        self.info = {}



    def genSamples(self):
        """
        Generate or distribute Monte Carlo samples.
        If self.points is provided, use its first nSamples rows.
        This enables nested MC evaluations.
        """

        if self.points is not None:
            points = np.asarray(self.points[:self.nSamples], dtype=float)

            if points.shape != (self.nSamples, self.stochDim):
                raise ValueError(
                    f"Expected points with shape "
                    f"({self.nSamples}, {self.stochDim}), "
                    f"got {points.shape}."
                )

            local_indices = np.arange(self.rank, self.nSamples, self.size)
            self.localPoints = points[local_indices]

        else:
            base = self.nSamples // self.size
            remainder = self.nSamples % self.size
            nLocal = base + (1 if self.rank < remainder else 0)

            self.localPoints = np.random.uniform(
                low=-1.0,
                high=1.0,
                size=(nLocal, self.stochDim),
            )

        self.localWeights = np.ones(len(self.localPoints), dtype=float) / self.nSamples

        self.nPad = 0
        self.info["initSamples"] = self.nSamples
        self.info["totalSamples"] = self.nSamples
        self.info["paddedPoints"] = 0


    def solve(self):
        """
        Evaluate QoIs in parallel and compute mean, variance,
        standard deviation, and Monte Carlo error in one pass.
        """

        self.genSamples()

        startTime = time.perf_counter()

        localWeightedSum = None
        localWeightedSecondMoment = None
        nQoi = None

        # Used for ETA calculation on rank 0
        localTotal = np.count_nonzero(self.localWeights)
        localFinished = 0


        for xi, weight in zip(self.localPoints, self.localWeights):
            if weight == 0.0:
                continue  # Ignore padded samples

            qoiValue = np.asarray(self.problem.qoi(xi), dtype=float)

            localFinished += 1

            # ETA calculation was implemented using ChatGPT
            if self.mcVerbose and self.rank == 0:
                elapsed = time.perf_counter() - startTime
                avgTimePerSample = elapsed / localFinished
                remaining = localTotal - localFinished
                eta = avgTimePerSample * remaining
                progress = 100.0 * localFinished / localTotal

                print(
                    f"[MC rank 0] "
                    f"{progress:6.2f}% | "
                    f"{localFinished:5d}/{localTotal} local samples | "
                    f"elapsed={elapsed:8.1f}s | "
                    f"remaining={eta:8.1f}s",
                    flush=True,
                )

            if localWeightedSum is None:
                nQoi = len(qoiValue)
                localWeightedSum = np.zeros(nQoi, dtype=float)
                localWeightedSecondMoment = np.zeros(nQoi, dtype=float)

            localWeightedSum += weight * qoiValue
            localWeightedSecondMoment += weight * qoiValue**2

        localNQoi = 0 if nQoi is None else nQoi
        nQoi = self.comm.allreduce(localNQoi, op=MPI.MAX) # AI, nQoi becomes maximum over all samples

        if localWeightedSum is None:
            localWeightedSum = np.zeros(nQoi, dtype=float)
            localWeightedSecondMoment = np.zeros(nQoi, dtype=float)

        globalWeightedSum = np.zeros(nQoi, dtype=float)
        globalWeightedSecondMoment = np.zeros(nQoi, dtype=float)

        self.comm.Allreduce(localWeightedSum, globalWeightedSum, op=MPI.SUM) # AI, Sum contributions over all ranks
        self.comm.Allreduce(localWeightedSecondMoment, globalWeightedSecondMoment, op=MPI.SUM) # AI, Sum contributions over all ranks

        runtime = time.perf_counter() - startTime

        self.mean = globalWeightedSum
        self.secondMoment = globalWeightedSecondMoment

        biased_variance = np.maximum(self.secondMoment - self.mean**2, 0.0)

        if self.nSamples > 1:
            self.variance = self.nSamples / (self.nSamples - 1) * biased_variance
        else:
            self.variance = np.full_like(biased_variance, np.nan)

        self.std = np.sqrt(self.variance)

        # MC estimator uncertainty
        self.estimatorVariance = self.variance / self.nSamples
        self.stochasticError = np.sqrt(self.estimatorVariance)

        self.info["runtime"] = runtime
        self.info["n_qoi"] = nQoi
        self.info["second_moment"] = self.secondMoment
        self.info["estimator_variance"] = self.estimatorVariance
        self.info["stochastic_error"] = self.stochasticError

        if self.rank == 0 and self.mcVerbose:
            self.printInfo()

        return (
            self.mean,
            self.variance,
            self.estimatorVariance,
            self.stochasticError,
        )



    def printInfo(self):
        """
        Print Monte Carlo summary and QoI statistics.
        """

        print()
        print("=" * 100)
        print("MPI Monte Carlo information")
        print("=" * 100)

        print(f"{'MPI ranks':<40} {self.size:>20}")
        print(f"{'Stochastic dimension':<40} {self.stochDim:>20}")
        print(f"{'Number of samples':<40} {self.nSamples:>20}")

        if "n_points_padded" in self.info:
            print(f"{'Padded samples':<40} {self.info['n_points_padded']:>20}")

        if "n_padding" in self.info:
            print(f"{'Padding samples':<40} {self.info['n_padding']:>20}")


        print(f"{'Runtime':<40} {self.info['runtime']:>20.3f} s")
        print(f"{'Number of QoIs':<40} {self.info['n_qoi']:>20}")

        print()

        print("=" * 100)
        print("Monte Carlo statistics")
        print("=" * 100)


        qoiNames = getattr(self, "qoiNames", None)

        if qoiNames is None:
            qoiNames = [f"QoI {i}" for i in range(self.info["n_qoi"])]

        print()
        print("-" * 150)

        print(
            f"{'QoI':<30}"
            f"{'Mean':>18}"
            f"{'Variance':>18}"
            f"{'Std':>18}"
            f"{'Estimator Var':>18}"
            f"{'MC Error':>18}"
        )

        print("-" * 150)

        for (name, mean, variance, std, estimatorVariance, stochasticError) in zip(qoiNames, self.mean, self.variance, self.std, self.estimatorVariance, self.stochasticError):

            print(
                f"{name:<30}"
                f"{mean:>18.8e}"
                f"{variance:>18.8e}"
                f"{std:>18.8e}"
                f"{estimatorVariance:>18.8e}"
                f"{stochasticError:>18.8e}"
            )

        print("=" * 150)