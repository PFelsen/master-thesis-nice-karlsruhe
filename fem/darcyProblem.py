"""
Darcy flow problem:

Find pressure p such that

- div(K grad(p)) = f      in D = [0,1]²

with boundary conditions

p = p_top     on Gamma_top
p = p_bottom  on Gamma_bottom

The Darcy flux is then computed as in the case of continous galerkin

q = -K grad(p)

Here:
   K(x)  = permeability field
   p(x)  = pressure
   q(x)  = Darcy velocity / flux

Quantities of interest:
outflow = Total outflow through bottom boundary
mean_flux = Average flux inside the whole domain.
mean_pressure_box = Average pressure inside a box
pressureL² = L² norm of pressure
fluxL² 0 L² norm of flux
"""


import time
import numpy as np#
import matplotlib.pyplot as plt

from fenics import *
from fenics import set_log_level, LogLevel
from petsc4py import PETSc

from fem.baseProblem import BaseProblem

set_log_level(LogLevel.ERROR)

import logging
# Avoid consol output
logging.getLogger("FFC").setLevel(logging.ERROR) 
logging.getLogger("UFL").setLevel(logging.ERROR)
logging.getLogger("dijitso").setLevel(logging.ERROR)


class Darcy(BaseProblem):
    def __init__(
        self,
        permeabilityField,
        mesh=None,
        meshLevel=4,
        degree=1,
        elementType="RT",
        darcySolver="mumps",
        darcyPrecon=None,
        relTol = 1e-10,
        absTol = 1e-13,
        maxIter = 20000,
        pTop=1.0,
        pBottom=0.0,
        f=None,
        qoiBox=((0.35, 0.65), (0.0, 0.2)),
        plotSolution=False,
        femDarcyVerbose=False,
        linearSolverVerbose=False,
        estimateFemError=False,
        computeMore=False,
    ):
        self.permeabilityField = permeabilityField

        self.mesh = mesh
        self.meshLevel = meshLevel
        self.degree = degree
        self.elementType = elementType
        self.darcySolver = darcySolver
        self.darcyPrecon = darcyPrecon
        self.relTol = relTol
        self.absTol = absTol
        self.maxIter = maxIter

        self.pTop = pTop
        self.pBottom = pBottom
        self.f = Constant(0.0) if f is None else f # Function f has to be a FEniCS object
        self.qoiBox = qoiBox

        self.plotSolution = plotSolution
        self.femDarcyVerbose = femDarcyVerbose
        self.linearSolverVerbose = linearSolverVerbose
        self.estimateFemError = estimateFemError
        self.computeMore = computeMore

        self.Vp = None # Function space for pressure V_p = {v in C^0(D) : v|_K in P_k}
        self.Vq = None # Vector valued function space for flux q, V_q = V_p x V_p
        self.W = None
        self.K = None
        self.p = None
        self.q = None
        self.info = {}

        # Cached FEM / QoI objects.
        self.n = None # Normal vectors
        self.dsBottom = None # Integral over bottom boundary
        self.domainArea = None # Caluclated area domain
        self.chi = None # Indicator function over QoI domain
        self.boxArea = None # Caluclated area of QoI box

        # CG cached objects.
        self.pTrial = None # Symbolic unknown function p_h in Vp.
        self.vTest = None # Symbolic test function v_h in Vp.
        self.cgBcs = None
        self.cgL = None

        # RT cached objects.
        self.qTrial = None
        self.pTrialRT = None
        self.vTestRT = None
        self.rTestRT = None
        self.rtBcSides = None
        self.dsMarked = None
        self.rtL = None

        self.build()


    def build(self):
        """
        Builds the model problem.
        """
        self.createMesh()
        self.createBoundaryMeasures()
        self.createFunctionSpaces()
        self.createQoiObjects()

        if self.elementType == "CG":
            self.buildCGStaticObjects()
        elif self.elementType == "RT":
            self.buildRTStaticObjects()
        else:
            raise NotImplementedError("Only CG and RT are supported.")
    

    def createSolver(self):
        """
        Creates localized version of solver. 
        Avoids MPI conflics with internal parallelization in FEniCS.
        """

        ksp = PETSc.KSP().create(PETSc.COMM_SELF) # Creates solver only on current rank
        ksp.setType(self.darcySolver)

        pc = ksp.getPC()

        if self.darcyPrecon is None or self.darcyPrecon == "none":
            pc.setType("none")
        else:
            pc.setType(self.darcyPrecon)

        solver = PETScKrylovSolver(ksp)

        solver.parameters["relative_tolerance"] = self.relTol
        solver.parameters["absolute_tolerance"] = self.absTol
        solver.parameters["maximum_iterations"] = self.maxIter
        solver.parameters["monitor_convergence"] = self.linearSolverVerbose
        solver.parameters["error_on_nonconvergence"] = True

        return solver


    def createMesh(self):
        """
        Creates FiniteElement Mesh using UnitSquareMesh
        """
        if self.mesh is not None: # Avoids double creating
            return

        n = 2 ** self.meshLevel
        self.mesh = UnitSquareMesh(MPI.comm_self, n, n) # MPI: One mesh per rank, not shared.


    def createBoundaryMeasures(self):
        """
        Helper function to get boundary integration measure restricted to the bottom boundary
        """
        # Create marker function that labels each edge
        boundaryMarker = MeshFunction( 
            "size_t",
            self.mesh,
            self.mesh.topology().dim() - 1,
        )
        boundaryMarker.set_all(0) # Default label 0

        class BottomBoundary(SubDomain):
            """
            Extract facets on bottom boundary.
            """
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0.0)

        BottomBoundary().mark(boundaryMarker, 1) # Set bottom marker to 1

        # Create actual measure integrating only over bottom boundary.
        self.dsBottom = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=boundaryMarker,
        )


    def createFunctionSpaces(self):
        """
        Creates function spaces from FEniCS associated with the choosen parameters.
        """
        if self.elementType == "CG":
            self.Vp = FunctionSpace(self.mesh, "CG", self.degree) # Function space for pressure V_p = {v in C^0(D) : v|_K in P_k}
            self.Vq = VectorFunctionSpace(self.mesh, "CG", self.degree) # Vector valued function space for flux q, V_q = V_p x V_p
            self.pTrial = TrialFunction(self.Vp) # Symbolic unknown function p_h in Vp.
            self.vTest = TestFunction(self.Vp) # Symbolic test function v_h in Vp.

        elif self.elementType == "RT":
            self.Vq = FunctionSpace(self.mesh, "RT", self.degree) # RT Space for flux
            self.Vp = FunctionSpace(self.mesh, "DG", self.degree - 1) # DG space for pressure

            rtElement = FiniteElement("RT", self.mesh.ufl_cell(), self.degree)
            dgElement = FiniteElement("DG", self.mesh.ufl_cell(), self.degree - 1)
            mixedElement = MixedElement([rtElement, dgElement])
            self.W = FunctionSpace(self.mesh, mixedElement)


    def createQoiObjects(self):
        """
        Create objects associated with QoI calculations.
        """
        self.n = FacetNormal(self.mesh)

        self.domainArea = assemble(
            Constant(1.0) * dx(domain=self.mesh)
        )

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

        self.boxArea = float(
            (x1 - x0) * (y1 - y0)
        )

        if self.boxArea <= 0.0:
            raise ValueError(
                f"Invalid QoI box {self.qoiBox}: "
                f"area={self.boxArea}"
            )


    def buildCGStaticObjects(self):
        """
        Create objects associated with method=continous galerkin.
        """
        def topBoundary(x, on_boundary):
            return on_boundary and near(x[1], 1.0)

        def bottomBoundary(x, on_boundary):
            return on_boundary and near(x[1], 0.0)

        bcTop = DirichletBC(self.Vp, Constant(self.pTop), topBoundary)
        bcBottom = DirichletBC(self.Vp, Constant(self.pBottom), bottomBoundary)
        # For continous galerkin -> homogeneous Neumann BCs on the left/right are the natural boundary condition of the weak form.

        self.cgBcs = [bcTop, bcBottom] # continousGalerkinBoundaryCondiitions
        self.cgL = self.f * self.vTest * dx # contionousGalerkinL


    def buildRTStaticObjects(self):
        """
        Create objects associated with method=RT.
        """
        self.qTrial, self.pTrialRT = TrialFunctions(self.W)
        self.vTestRT, self.rTestRT = TestFunctions(self.W)

        # Need to specify Neumann conditions on the left and right
        def leftRightBoundary(x, on_boundary):
            return on_boundary and (near(x[0], 0.0) or near(x[0], 1.0))

        # rt Boundarie Conditions on the left/right sides
        self.rtBcSides = DirichletBC(
            self.W.sub(0),
            Constant((0.0, 0.0)),
            leftRightBoundary,
        )

        boundaryMarkers = MeshFunction(
            "size_t", # nonnegative integer type
            self.mesh,
            self.mesh.topology().dim() - 1,
        )
        boundaryMarkers.set_all(0)

        # Specify top/bottom boundary
        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 1.0)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0.0)

        # Mark them
        TopBoundary().mark(boundaryMarkers, 1)
        BottomBoundary().mark(boundaryMarkers, 2)

        self.dsMarked = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=boundaryMarkers,
        ) # self.dsMarked(1) top, self.dsMarked(2) bottom

        # L form for RT elements
        self.rtL = ( self.f * self.rTestRT * dx(domain=self.mesh)
            -Constant(self.pTop) * dot(self.vTestRT, self.n) * self.dsMarked(1)
            -Constant(self.pBottom) * dot(self.vTestRT, self.n) * self.dsMarked(2)
        )


    def updatePermeability(self, xi=None):
        """
        Allows to update xi for UQ runs.
        """
        if xi is not None:
            self.permeabilityField.setXi(xi)

        if not hasattr(self.permeabilityField, "adaptToMesh"):
            raise TypeError("permeabilityField must provide adaptToMesh(mesh).")

        if self.permeabilityField.isDiscontinuous: # Uses marker of multiphase fields
            self.K = self.permeabilityField.adaptToMesh(
                self.mesh,
                elementType="DG", # DG0 = constant over cells, good for nonsmooth indicator fields
                degree=0,
            )
        else:
            self.K = self.permeabilityField.adaptToMesh(
                self.mesh,
                elementType="CG",
                degree=self.degree,
            )


    def solve(self, xi=None):
        """
        Solve pipeline. Differentiate between CG and RT.
        """
        self.updatePermeability(xi)

        if self.elementType == "CG":
            result = self.solveCG()
        elif self.elementType == "RT":
            result = self.solveRT()
        else:
            raise NotImplementedError("Only CG and RT are supported.")

        if self.estimateFemError:
            pressureError, fluxError = self.estimateFemErrorByMeshRefinement()

            self.info["fem_pressure_error"] = pressureError
            self.info["fem_flux_error"] = fluxError
            
        if self.femDarcyVerbose:
            self.printInfo()

        if self.plotSolution:
            self.plotSolutionFields()

        return result


    def solveCG(self):
        """
        Solving in case of continous Galerkin.
        """
        pSol = Function(self.Vp)

        a = self.K * dot(grad(self.pTrial), grad(self.vTest)) * dx
        A = assemble(a) # Assemble matrix A
        b = assemble(self.cgL) # Assemble b from continoutsGalerkin L

        for bc in self.cgBcs: # Apply BC
            bc.apply(A, b)

        startTime = time.perf_counter()
        ksp_reason = None
        ksp_residual_norm = None

        if self.darcySolver == "mumps":
            # CreateSolver can't handle mumps
            solve(A, pSol.vector(), b, "mumps")
            numIter = None # Since it is a direct solver
        else:
            solver = self.createSolver() 
            numIter = solver.solve(A, pSol.vector(), b)
            ksp = solver.ksp()
            ksp_reason = ksp.getConvergedReason()
            ksp_residual_norm = ksp.getResidualNorm()


        runtime = time.perf_counter() - startTime

        self.p = pSol
        self.q = project(-self.K * grad(self.p), self.Vq) # Get flux q

        self.fillInfo(runtime=runtime, iterations=numIter)
        if self.darcySolver != "mumps":
            self.info["ksp_reason"] = ksp_reason
            self.info["ksp_residual_norm"] = ksp_residual_norm
        return self.p, self.q, self.info


    def solveRT(self):
        """
        Solving in case of RT.
        """
        sol = Function(self.W)

        a = (
            (1.0 / self.K) * dot(self.qTrial, self.vTestRT) * dx
            - self.pTrialRT * div(self.vTestRT) * dx
            + div(self.qTrial) * self.rTestRT * dx
        )

        A, b = assemble_system(a, self.rtL, [self.rtBcSides])

        startTime = time.perf_counter()

        ksp_reason = None
        ksp_residual_norm = None

        if self.darcySolver == "mumps":
            # CreateSolver can't handle mumps
            solve(A, sol.vector(), b, "mumps")
            numIter = None # Since it is a direct solver
        else:
            solver = self.createSolver() # Call to avoid MPI conflict
            numIter = solver.solve(A, sol.vector(), b)
            ksp = solver.ksp()
            ksp_reason = ksp.getConvergedReason()
            ksp_residual_norm = ksp.getResidualNorm()

        runtime = time.perf_counter() - startTime

        qSol, pSol = sol.split(deepcopy=True) # sol contains both solutions together
        self.q = qSol
        self.p = pSol

        self.fillInfo(runtime=runtime, iterations=numIter)

        if self.darcySolver != "mumps":
            self.info["ksp_reason"] = ksp_reason
            self.info["ksp_residual_norm"] = ksp_residual_norm

        return self.p, self.q, self.info


    def qoi(self, xi):
        """
        Quantity of Interest function used by UQ framework. 
        Call solve pipeline and return Quantities of Interest.
        """
        self.solve(xi)

        return np.array([ self.info["outflow"], self.info["mean_flux"], self.info["mean_pressure_box"], self.info["pressure_L2"], self.info["flux_L2"]], dtype=float)


    def fillInfo(self, runtime, iterations):
        """
        Fill info dictionary with information when called.
        """

        outflow = assemble(dot(self.q, self.n) * self.dsBottom(1)) # Integrate to get outflow
        meanFlux = assemble(sqrt(dot(self.q, self.q)) * dx) / self.domainArea
        meanPressureBox = assemble(self.p * self.chi * dx) / self.boxArea

        pressureL2 = norm(self.p, "L2")
        fluxL2 = norm(self.q, "L2")

        self.info = {
            "runtime": runtime,
            "mesh_level": self.meshLevel,
            "polynomial_degree": self.degree,
            "element_type": self.elementType,
            "linear_solver": self.darcySolver,
            "preconditioner": self.darcyPrecon,
            "iterations": iterations,
            "h_max": self.mesh.hmax(),
            "outflow": outflow,
            "mean_flux": meanFlux,
            "mean_pressure_box": meanPressureBox,
            "pressure_L2": pressureL2,
            "flux_L2": fluxL2
        }

        if self.elementType == "CG":
            self.info["dofs"] = self.Vp.dim()
        else:
            self.info["dofs_flux"] = self.Vq.dim()
            self.info["dofs_pressure"] = self.Vp.dim()
            self.info["dofs"] = self.W.dim()

        if self.computeMore:
            self.additionalQuantities()


    def additionalQuantities(self):
        """
        In case of single run, calculcate additional quantities. Avoid if using UQ for computational reasons.
        """

        KVals = self.K.vector().get_local()
        pVals = self.p.vector().get_local()

        if self.elementType == "CG":
            residualField = project(-div(self.K * grad(self.p)) - self.f, self.Vp)
            pdeResidual = norm(residualField, "L2")
            divQ = project(div(self.q) - self.f, self.Vp)
            massError = norm(divQ, "L2")
            qMag = project(sqrt(dot(self.q, self.q)), self.Vp)
        else:
            divQ = project(div(self.q) - self.f, self.Vp)
            massError = norm(divQ, "L2")
            pdeResidual = massError
            qMag = project(sqrt(dot(self.q, self.q)), FunctionSpace(self.mesh, "DG", 0))

        qVals = qMag.vector().get_local()

        self.info.update(
            {
                "pde_residual": pdeResidual,
                "mass_error": massError,
                "K_min": KVals.min(),
                "K_max": KVals.max(),
                "K_mean": KVals.mean(),
                "pressure_min": pVals.min(),
                "pressure_max": pVals.max(),
                "flux_max": qVals.max(),
            }
        )

    def estimateFemErrorByMeshRefinement(self):

        refinedDarcy = Darcy(
            permeabilityField=self.permeabilityField,
            meshLevel=self.meshLevel + 1,
            degree=self.degree,
            elementType=self.elementType,
            darcySolver=self.darcySolver,
            darcyPrecon=self.darcyPrecon,
            relTol=self.relTol,
            absTol=self.absTol,
            maxIter=self.maxIter,
            pTop=self.pTop,
            pBottom=self.pBottom,
            f = self.f,
            qoiBox=self.qoiBox,
            plotSolution=False,
            femDarcyVerbose=False,
            linearSolverVerbose=False,
            estimateFemError=False,
            computeMore=False,
        )

        pFine, qFine, _ = refinedDarcy.solve()

        # Compare pressure on the fine mesh
        pCoarseOnFine = interpolate(self.p, refinedDarcy.Vp)

        pressureError = errornorm(
            pFine,
            pCoarseOnFine,
            norm_type="L2",
            degree_rise=2,
        )

        # Compare flux on the fine mesh
        if self.elementType == "CG":
            qCoarseOnFine = project(self.q, refinedDarcy.Vq)

            fluxError = errornorm(
                qFine,
                qCoarseOnFine,
                norm_type="L2",
                degree_rise=2,
            )

        else:  # RT
            qCoarseOnFine = interpolate(self.q, refinedDarcy.Vq)
            #qCoarseOnFine = project(self.q, refinedDarcy.Vq)

            fluxError = errornorm(
                qFine,
                qCoarseOnFine,
                norm_type="L2",
                degree_rise=2,
            )

        return pressureError, fluxError


    def plotSolutionFields(self):
        """
        Plot solution fields. Called only if plotSolution=True.
        """
        self.plotPermeability()
        self.plotPressure()
        self.plotFlux()


    def plotPermeability(self):
        """
        Plot adapted permeability field K when available.
        """
        if self.K is None:
            return

        plt.figure(figsize=(6, 5))
        kPlot = plot(self.K)
        plt.colorbar(kPlot)
        plt.title("Darcy permeability K")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()


    def plotPressure(self):
        """
        Plot Darcy pressure field.
        """
        if self.p is None:
            return

        plt.figure(figsize=(6, 5))
        pPlot = plot(self.p)
        plt.colorbar(pPlot)
        plt.title("Darcy pressure p")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()


    def plotFlux(self):
        """
        Plot Darcy flux field.
        """
        if self.q is None:
            return

        plt.figure(figsize=(6, 5))
        plot(self.q)
        plt.title("Darcy flux q")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()


    def printInfo(self):
        """
        Print function for information.
        """
        nEqual = 66
        info = self.info

        print("=" * nEqual)
        print("Darcy problem finished")
        print("=" * nEqual)

        def row(label, value):
            print(f"{label:<45} {value:>20}")

        row("Runtime", f"{info['runtime']:.3f} s")
        row("Mesh level", info["mesh_level"])
        row("Polynomial degree", info["polynomial_degree"])
        row("Element type", info["element_type"])
        row("DoFs", info["dofs"])
        row("h_max", f"{info['h_max']:.3e}")

        print()
        row("Linear solver", info["linear_solver"])
        row("Preconditioner", info["preconditioner"])
        row("Iterations", "direct solver" if info["iterations"] is None else info["iterations"])

        if "pde_residual" in info:
            print()
            row("PDE residual", f"{info['pde_residual']:.3e}")
            row("Mass error", f"{info['mass_error']:.3e}")
            row("K min", f"{info['K_min']:.3e}")
            row("K max", f"{info['K_max']:.3e}")
            row("K mean", f"{info['K_mean']:.3e}")
            row("Pressure min", f"{info['pressure_min']:.3e}")
            row("Pressure max", f"{info['pressure_max']:.3e}")
            row("Flux max", f"{info['flux_max']:.3e}")

        if info.get("fem_pressure_error", None) is not None:
            print()
            row("FEM pressure error", f"{info['fem_pressure_error']:.3e}")
            if info.get("fem_flux_error", None) is not None:
                row("FEM flux error", f"{info['fem_flux_error']:.3e}")

        print()
        row("Outflow", f"{info['outflow']:.8e}")
        row("Mean q", f"{info['mean_flux']:.8e}")
        row("Mean p in box", f"{info['mean_pressure_box']:.8e}")
        row("Pressure L²", f"{info['pressure_L2']:.8e}")
        row("Flux L²", f"{info['flux_L2']:.8e}")

        print("=" * nEqual)

