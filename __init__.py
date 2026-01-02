"""
Racing Line Optimization using Particle Swarm Optimization (PSO)

A modular Python package for optimizing motorsport racing lines using
Particle Swarm Optimization with physics-based lap time simulation.

Main modules:
- geometry: Track representation, splines, and curvature computation
- lap_simulation: Lap time computation with vehicle dynamics
- pso_optimizer: Particle Swarm Optimization implementation

Example usage:
    >>> from geometry import create_simple_oval_track
    >>> from lap_simulation import LapSimulator, evaluate_racing_line
    >>> from pso_optimizer import PSOOptimizer
    >>>
    >>> track = create_simple_oval_track()
    >>> simulator = LapSimulator(lateral_g=2.0)
    >>> 
    >>> def objective(alpha):
    >>>     return evaluate_racing_line(track, alpha, simulator)
    >>>
    >>> optimizer = PSOOptimizer(objective, dimension=track.n_points)
    >>> best_alpha, best_time, info = optimizer.optimize()
"""

__version__ = "1.0.0"
__author__ = "Racing Line Optimization Team"

# Import main classes for convenient access
from geometry import Track, RacingLineSpline, create_simple_oval_track
from lap_simulation import LapSimulator, evaluate_racing_line
from pso_optimizer import PSOOptimizer, Particle

__all__ = [
    'Track',
    'RacingLineSpline',
    'create_simple_oval_track',
    'LapSimulator',
    'evaluate_racing_line',
    'PSOOptimizer',
    'Particle',
]
