"""2D D2Q9 Lattice Boltzmann solvers for far-field cylinder flow.

Classical Shan-He thermal LBM is the verification baseline.
Quantum Yang-Hung UUB-BGK is a separate formulation.
"""

__version__ = "0.1.0"

from lbm.classical import SimulationConfig, SimulationResult, run_simulation

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "__version__",
    "run_simulation",
]
