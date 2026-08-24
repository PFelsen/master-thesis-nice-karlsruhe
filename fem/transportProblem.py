import time
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fenics import *
from fenics import set_log_level, LogLevel

from petsc4py import PETSc

from fem.baseProblem import BaseProblem
from fem.darcyProblem import Darcy

set_log_level(LogLevel.ERROR)
logging.getLogger("FFC").setLevel(logging.ERROR)
logging.getLogger("UFL").setLevel(logging.ERROR)
logging.getLogger("dijitso").setLevel(logging.ERROR)


class Transport(BaseProblem):
    """
    Darcy-driven DG transport problem.

    Darcy problem:
        q is computed by the Darcy problem on the Darcy mesh.

    Transport problem:
        dc/dt + div(q c) = 0

    QoI map:
        xi -> [
            mean_c_box_time_average,
            l2,
            l2_squared,
        ]
    """

    def __init__(
        self,
        # Permeability
        permeabilityField,

        # Transport FEM settings
        meshLevelTransport=2,
        degreeTransport=1,
        elementTypeTransport="DG",
        transportSolver="gmres",
        transportPrecon="ilu",
        cInflow=Constant(0.0),
        c0=None,
        linearSolverVerbose=False,
        femTransportVerbose=False,

        # Darcy settings
        meshLevelDarcy=None, # Usually set to meshLevelTransport
        degreeDarcy= None, # If None, use degreeTransport + 1 for RT
        elementTypeDarcy="RT",
        darcySolver="mumps",
        darcyPrecon=None,
        pTop=1.0,
        pBottom=0.0,
        femDarcyVerbose=False,
        computeMore=True,

        # Solver tolerances
        relTol=1e-8,
        absTol=1e-12,
        maxIter=10000,

        # Time settings
        theta=1.0,
        T=0.5,
        nSteps=100,
        adaptCfl=True,
        maxCfl=0.25,
        maxSteps=2500,

        # QoI
        qoiBox=((0.35, 0.65), (0.0, 0.2)),

        # Plotting
        plotSolution=False,
        storeSolutionHistory = False, # Used for convergence studies without plotting

        # Only for convergence study/Debugging. Overwrites darcy solving with anayltic solution -y. 
        analyticFluxConst = False

    ):
        self.permeabilityField = permeabilityField

        # Transport settings
        self.meshLevelTransport = meshLevelTransport 
        self.degreeTransport = degreeTransport
        self.elementTypeTransport = elementTypeTransport
        self.transportSolver = transportSolver
        self.transportPrecon = transportPrecon
        self.cInflow = cInflow
        self.c0 = c0
        self.linearSolverVerbose = linearSolverVerbose
        self.femTransportVerbose = femTransportVerbose

        # Darcy settings
        self.meshLevelDarcy = (meshLevelTransport if meshLevelDarcy is None else meshLevelDarcy)
        self.degreeDarcy = (degreeTransport + 1 if degreeDarcy is None and elementTypeDarcy == "RT" else degreeTransport if degreeDarcy is None else degreeDarcy)   
        self.elementTypeDarcy = elementTypeDarcy
        self.darcySolver = darcySolver
        self.darcyPrecon = darcyPrecon
        self.pTop = pTop
        self.pBottom = pBottom
        self.femDarcyVerbose = femDarcyVerbose
        self.computeMore = computeMore


        # Solver tolerances
        self.relTol = relTol
        self.absTol = absTol
        self.maxIter = maxIter

        # Time settings
        self.theta = theta
        self.T = T
        self.nSteps = nSteps
        self.dt = T / nSteps
        self.adaptCfl = adaptCfl
        self.maxCfl = maxCfl
        self.maxSteps = maxSteps

        # QoI / plotting
        self.qoiBox = qoiBox
        self.plotSolution = plotSolution
        self.storeSolutionHistory = storeSolutionHistory

        # Only for convergence study. Overwrites darcy solving with anayltic solution -y. Debugging!
        self.analyticFluxConst = analyticFluxConst

        # Mesh, spaces, measures
        self.mesh = None
        self.n = None
        self.dsBottom = None
        self.Vc = None
        self.VqTransport = None

        # Darcy object and fields
        self.darcyProblem = None
        self.p = None
        self.q = None

        # Transport unknowns and cached forms
        self.cTrial = None
        self.wTest = None
        self.cOld = None
        self.cNew = None
        self.dtConstant = None
        self.thetaConstant = None
        self.transportAForm = None
        self.transportLForm = None

        # QoI helpers
        self.chi = None
        self.boxArea = None

        # Results / history
        self.cFinal = None
        self.solutionHistory = []
        self.timeHistory = []
        self.outflowHistory = []
        self.cflNumber = None
        self.info = {}

        self.build()


    def build(self):
        """
        Build the model at the beginning.
        """
        self.createMesh()
        self.createBoundaryMeasures()
        self.createFunctionSpaces()
        self.createQoiObjects()
        self.createDarcyProblem()
        self.createTransportForms()


    def createMesh(self):
        """
        Create mesh on every rank.
        """
        if self.mesh is not None:
            return

        nCells = 2 ** self.meshLevelTransport
        self.mesh = UnitSquareMesh(MPI.comm_self, nCells, nCells)
        self.n = FacetNormal(self.mesh)


    def createBoundaryMeasures(self):
        """
        Used for QOI calculations
        """
        bottomMarker = MeshFunction(
            "size_t",
            self.mesh,
            self.mesh.topology().dim() - 1,
        )
        bottomMarker.set_all(0)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0.0)

        BottomBoundary().mark(bottomMarker, 1)

        self.dsBottom = Measure("ds", domain=self.mesh, subdomain_data=bottomMarker)


    def createFunctionSpaces(self):
        """
        Create function spaces to solve both pronlems.
        """
        self.Vc = FunctionSpace(self.mesh, self.elementTypeTransport, self.degreeTransport)

    
        if self.elementTypeDarcy == "RT":
            #self.VqTransport = VectorFunctionSpace(self.mesh, "DG", self.degreeDarcy)
            self.VqTransport = FunctionSpace(self.mesh, "RT", self.degreeDarcy)
        else:
            self.VqTransport = VectorFunctionSpace(self.mesh, "CG", self.degreeDarcy)

        self.q = Function(self.VqTransport)

        self.cTrial = TrialFunction(self.Vc)
        self.wTest = TestFunction(self.Vc)
        self.cOld = Function(self.Vc)
        self.cNew = Function(self.Vc)


    def createQoiObjects(self):
        """
        Create the indicator used for the box-averaged QoI.
        """
        (x0, x1), (y0, y1) = self.qoiBox

        self.chi = Expression(
            """
            x[0] >= x0 &&
            x[0] <= x1 &&
            x[1] >= y0 &&
            x[1] <= y1
            ? 1.0 : 0.0
            """,
            degree=0,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        )

        # Exact physical area of the rectangular QoI region.
        self.boxArea = float(
            (x1 - x0) * (y1 - y0)
        )

        if self.boxArea <= 0.0:
            raise ValueError(
                f"Invalid QoI box {self.qoiBox}: "
                f"area={self.boxArea}"
            )


    def createDarcyProblem(self):
        """
        Assemble the underlying Darcy problem.
        """
        self.darcyProblem = Darcy(
            permeabilityField=self.permeabilityField,
            meshLevel=self.meshLevelDarcy,
            degree=self.degreeDarcy,
            elementType=self.elementTypeDarcy,
            darcySolver=self.darcySolver,
            darcyPrecon=self.darcyPrecon,
            pTop=self.pTop,
            pBottom=self.pBottom,
            plotSolution=False,
            femDarcyVerbose=self.femDarcyVerbose,
            linearSolverVerbose=self.linearSolverVerbose,
            estimateFemError=False,
            computeMore=self.computeMore,
        )


    def createTransportForms(self):
        """
        Create a,L for transport depending on theta.
        """
        self.dtConstant = Constant(self.dt)
        self.thetaConstant = Constant(self.theta)

        c = self.cTrial
        w = self.wTest
        cOld = self.cOld
        q = self.q
        n = self.n

        # Interior upwind flux
        qn = dot(avg(q), n("+"))
        qnPos = 0.5 * (qn + abs(qn))
        qnNeg = 0.5 * (qn - abs(qn))

        # Boundary upwind flux
        qnb = dot(q, n)
        qnbPos = 0.5 * (qnb + abs(qnb))
        qnbNeg = 0.5 * (qnb - abs(qnb))

        # dS = interior faces
        # ds = boundary faces
        operatorNew = (-c * dot(q, grad(w)) * dx + (qnPos * c("+") + qnNeg * c("-")) * jump(w) * dS + qnbPos * c * w * ds)

        operatorOld = (-cOld * dot(q, grad(w)) * dx + (qnPos * cOld("+") + qnNeg * cOld("-")) * jump(w) * dS + qnbPos * cOld * w * ds)

        self.transportAForm = (c * w * dx + self.dtConstant * self.thetaConstant * operatorNew)

        self.transportLForm = (cOld * w * dx - self.dtConstant * (1.0 - self.thetaConstant) * operatorOld - self.dtConstant * qnbNeg * self.cInflow * w * ds )


    def solveDarcy(self, xi=None):
        """
        Solve Darcy for one xi.
        """

        if self.analyticFluxConst:
            """
            Debugging/Convergence study. Return analytical solution if permeabilty=1.
            """
            pAnalytic = Expression("x[1]", degree=self.degreeDarcy)
            qAnalytic = Expression(("0.0", "-1.0"), degree=self.degreeDarcy)

            self.p = pAnalytic

            qTransport = interpolate(qAnalytic, self.VqTransport)
            self.q.assign(qTransport)

            self.info["darcy"] = {}
            return self.p, self.q, {}
        

        p, qDarcy, info = self.darcyProblem.solve(xi)
        self.p = p

        qTransport = interpolate(qDarcy, self.VqTransport)
        self.q.assign(qTransport)

        self.info["darcy"] = info


    def computeCfl(self):
        """
        Computes cfl number and overwrites dt and n if adaptCfl= True
        """
        darcyInfo = self.info.get("darcy", {})

        if "flux_max" in darcyInfo:
            vmax = darcyInfo["flux_max"]
        else:
            qMag = project(
                sqrt(dot(self.q, self.q)),
                FunctionSpace(self.mesh, "DG", 0),
            )
            vmax = qMag.vector().get_local().max()

        h = self.mesh.hmin()
        cflNumber = self.dt * vmax / h

        if self.adaptCfl and cflNumber > self.maxCfl:
            oldNSteps = self.nSteps

            dtRequired = self.maxCfl * h / max(vmax, 1e-14)
            requiredNSteps = int(np.ceil(self.T / dtRequired))

            self.nSteps = min(requiredNSteps, self.maxSteps)
            self.dt = self.T / self.nSteps
            self.dtConstant.assign(self.dt)

            cflNumber = self.dt * vmax / h

            if self.femTransportVerbose:
                print(
                    f"Warning: CFL too large. "
                    f"Changed nSteps from {oldNSteps} to {self.nSteps}; "
                    f"CFL = {cflNumber:.3e}."
                )
        else:
            self.dtConstant.assign(self.dt)

        self.cflNumber = cflNumber
        return cflNumber
    

    def createSolver(self):
        """
        Create a transport Krylov solver.

        """
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setType(self.transportSolver)

        pc = ksp.getPC()
        if self.transportPrecon is None or self.transportPrecon == "none":
            pc.setType("none")
        else:
            pc.setType(self.transportPrecon)
            
        solver = PETScKrylovSolver(ksp)


        solver.parameters["relative_tolerance"] = self.relTol
        solver.parameters["absolute_tolerance"] = self.absTol
        solver.parameters["maximum_iterations"] = self.maxIter
        solver.parameters["monitor_convergence"] = self.linearSolverVerbose
        solver.parameters["error_on_nonconvergence"] = True

        return solver


    def reset(self):
        """
        Reset for UQ pipeline to overwrite history.
        """
        if self.c0 is None:
            self.cOld.interpolate(Constant(0.0))
        else:
            self.cOld.interpolate(self.c0)

        self.cNew.vector().zero()
        self.cNew.vector().apply("insert")

        self.solutionHistory = []
        self.timeHistory = []
        self.outflowHistory = []

        if self.plotSolution or self.storeSolutionHistory:
            cSave = Function(self.Vc)
            cSave.assign(self.cOld)
            self.solutionHistory.append(cSave)
            self.timeHistory.append(0.0)


    def solveTransport(self):
        """
        Solve the transport problem.
        """
        self.reset() # Initial reset. Used in repeated UQ runs

        initMass = assemble(self.cOld * dx(domain=self.mesh))

        c0Values = self.cOld.vector().get_local()
        cMinGlobal = float(np.min(c0Values))
        cMaxGlobal = float(np.max(c0Values))

        startTime = time.perf_counter()

        # Assemble A
        A = assemble(self.transportAForm)
        solver = self.createSolver()
        solver.set_operator(A)

        outflowTotal = 0.0
        outflowFirstMoment = 0.0
        timeIntegratedMeanCBox = 0.0
        normL2Squared = 0.0
        maxNumIter = 0
        b = None
        meanCBoxStep = 0.0

        qnBoundary = dot(self.q, self.n)

        for step in range(1, self.nSteps + 1):
            current_time = step * self.dt
            # Update time in cInflow expression if time dependent
            if hasattr(self.cInflow, "t"):
                self.cInflow.t = current_time
            
            b = assemble(self.transportLForm)
            numIter = solver.solve(self.cNew.vector(), b)
            maxNumIter = max(maxNumIter, numIter)

            cValues = self.cNew.vector().get_local()
            cMinGlobal = min(cMinGlobal, float(np.min(cValues)))
            cMaxGlobal = max(cMaxGlobal, float(np.max(cValues)))   

            # Theta-consistent state for time-integrated diagnostics
            cTheta = (self.thetaConstant * self.cNew + (1.0 - self.thetaConstant) * self.cOld) # Essentially apply theta scheme here as well to match integration
 

            stepL2Squared = assemble(cTheta * cTheta * dx(domain=self.mesh))
            normL2Squared += self.dt * stepL2Squared

            outflowRate = assemble(conditional(gt(qnBoundary, 0.0), qnBoundary * cTheta, 0.0) * self.dsBottom(1))
            self.outflowHistory.append(outflowRate)

            massBoxStep = assemble(cTheta * self.chi * dx(domain=self.mesh))
            meanCBoxStep = massBoxStep / max(self.boxArea, 1e-14)
            timeIntegratedMeanCBox += self.dt * meanCBoxStep
            outflowTotal += self.dt * outflowRate

            # First temporal moment of the outflow.
            # Correct for theta=1.0
            outflowFirstMoment += (
                self.dt * current_time * outflowRate
            )

            if self.plotSolution or self.storeSolutionHistory:
                cSave = Function(self.Vc)
                cSave.assign(self.cNew)
                self.solutionHistory.append(cSave)
                self.timeHistory.append(step * self.dt)

            self.cOld.assign(self.cNew)

        runtime = time.perf_counter() - startTime


        L2FinalSquared = assemble(self.cNew * self.cNew * dx(domain=self.mesh))
        L2Final = np.sqrt(L2FinalSquared)

        self.cFinal = Function(self.Vc)
        self.cFinal.assign(self.cNew)

        massFinal = assemble(self.cNew * dx(domain=self.mesh))
        massLoss = initMass - massFinal - outflowTotal
        relMassLoss = abs(massLoss) / max(abs(initMass), 1e-14)

        if b is not None:
            linearResidualVec = b.copy()
            A.mult(self.cNew.vector(), linearResidualVec)
            linearResidualVec.axpy(-1.0, b)
            linearResidual = linearResidualVec.norm("l2")
        else:
            linearResidual = 0.0


        meanCBoxTimeAverage = timeIntegratedMeanCBox / self.T
        normL2 = np.sqrt(max(normL2Squared, 0.0))

        normalizedFirstExitMoment = (
            outflowFirstMoment
            / max(abs(initMass), 1e-14)
        )


        self.info["transport"] = {
            "runtime": runtime,
            "mesh_level": self.meshLevelTransport,
            "polynomial_degree": self.degreeTransport,
            "element_type": self.elementTypeTransport,
            "dofs": self.Vc.dim(),
            "h_min": self.mesh.hmin(),
            "h_max": self.mesh.hmax(),
            "T": self.T,
            "dt": self.dt,
            "n_steps": self.nSteps,
            "theta": self.theta,
            "linear_solver": self.transportSolver,
            "preconditioner": self.transportPrecon,
            "nIteration": maxNumIter,
            "linear_residual": linearResidual,
            "cfl": self.cflNumber,
            "mass_initial": initMass,
            "mass_final": massFinal,
            "c_min_global": cMinGlobal,
            "c_max_global": cMaxGlobal,
            "outflow_total": outflowTotal,
            "outflow_first_moment": outflowFirstMoment,
            "outflow_first_moment_normalized": normalizedFirstExitMoment,
            "mass_loss": massLoss,
            "rel_mass_loss": relMassLoss,
            "mean_c_box_time_average": meanCBoxTimeAverage,
            "L2norm": normL2,
            "L2normSquared": normL2Squared,
            "L2normFinal": L2Final
        }


    def solve(self, xi=None):
        """
        Solving pipeline
        """
        self.solveDarcy(xi)
        self.computeCfl()
        self.solveTransport()

        if self.femTransportVerbose:
            self.printInfo()

        if self.plotSolution:
            self.plotFlux()
            self.animateTransport()

        return self.cFinal, self.info


    def qoi(self, xi):
        """
        QoI calculcations.
        """
        self.solve(xi)
        info = self.info["transport"]


        return np.array(
            [
                info["outflow_first_moment_normalized"],
                info["mean_c_box_time_average"],
                info["L2normSquared"]
            ],
            dtype=float
        )
    
    def printInfo(self):
        nEqual = 66
        info = self.info["transport"]

        print()
        print("=" * nEqual)
        print("Transport problem finished")
        print("=" * nEqual)

        def row(label, value):
            print(f"{label:<45} {value:>20}")

        row("Runtime", f"{info['runtime']:.3f} s")
        row("Mesh level", info["mesh_level"])
        row("Polynomial degree", info["polynomial_degree"])
        row("Element type", info["element_type"])
        row("DoFs", info["dofs"])
        row("h_min", f"{info['h_min']:.3e}")
        row("h_max", f"{info['h_max']:.3e}")

        print()
        row("Final time T", f"{info['T']:.3e}")
        row("Time step dt", f"{info['dt']:.3e}")
        row("Number of timesteps", info["n_steps"])
        row("Theta", f"{info['theta']:.2f}")
        row("CFL", f"{info['cfl']:.3e}")

        print()
        row("Linear solver", info["linear_solver"])
        row("Preconditioner", info["preconditioner"])
        row("Max nIter", info["nIteration"])
        row("Linear residual ‖Acₕ − b‖₂", f"{info['linear_residual']:.3e}")

        print()
        row("Initial mass", f"{info['mass_initial']:.6e}")
        row("Final mass", f"{info['mass_final']:.6e}")
        row("Bottom outflow", f"{info['outflow_total']:.6e}")
        row("Mass loss", f"{info['mass_loss']:.3e}")
        row("Relative mass loss", f"{info['rel_mass_loss']:.3e}")
        row("min c", f"{info['c_min_global']:.6e}")
        row("max c", f"{info['c_max_global']:.6e}")

        print()
        row("Mean c box time avg", f"{info['mean_c_box_time_average']:.6e}")
        row("||c|| L2(Dx[0,T])", f"{info['L2norm']:.6e}")
        row("||c||² L2(Dx[0,T])", f"{info['L2normSquared']:.6e}")
        print("=" * nEqual)


    def plotFlux(self):
        plt.figure(figsize=(6, 5))
        plot(self.q)
        plt.title("Darcy flux field q")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()

 
    def animateTransport(self):

        fig, ax = plt.subplots(figsize=(6, 6))

        # --- animation settings ---
        targetSeconds = 5.0
        fps = 24
        intervalMs = int(1000 / fps)

        nSavedFrames = len(self.solutionHistory)
        targetFrames = max(1, int(targetSeconds * fps))

        plotEvery = max(
            1,
            int(np.ceil(nSavedFrames / targetFrames))
        )

        frameIds = list(range(0, nSavedFrames, plotEvery))

        # make sure final frame is included
        if frameIds[-1] != nSavedFrames - 1:
            frameIds.append(nSavedFrames - 1)

        # -Global color
        cMin = self.info["transport"]["c_min_global"]
        cMax = self.info["transport"]["c_max_global"]

        # --- initial frame ---
        firstId = frameIds[0]

        tri = plot(
            self.solutionHistory[firstId],
            cmap="viridis",
            vmin=cMin,
            vmax=cMax,
        )

        # --- fixed colorbar ---
        cbar = fig.colorbar(
            tri,
            ax=ax,
            label="Concentration",
        )

        cbar.ax.set_title(
            f"min={cMin:.2f}\nmax={cMax:.2f}",
            fontsize=9,
            pad=10,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def animate(i):

            frameId = frameIds[i]

            cVals = self.solutionHistory[frameId].vector().get_local()
            tri.set_array(cVals)

            ax.set_title(
                f"DG transport, "
                f"t={self.timeHistory[frameId]:.3f}"
            )

            return (tri,)

        transportAnimation = animation.FuncAnimation(
            fig,
            animate,
            frames=len(frameIds),
            interval=intervalMs,
            blit=False,
        )

        plt.show()