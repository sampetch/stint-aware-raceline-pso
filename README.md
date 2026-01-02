# Racing Line Optimization using Particle Swarm Optimization (PSO)

A Python project that optimizes motorsport racing lines using Particle Swarm Optimization. The system models a racing track as a centerline with width, defines racing lines using alpha values between inner and outer borders, converts paths to splines for smooth interpolation, computes curvature and lap times using lateral grip limits, and minimizes lap time through PSO optimization.

## Features

- **Modular Architecture**: Clean separation of concerns into geometry, lap simulation, and optimization modules
- **Numerically Stable**: Robust handling of edge cases and numerical precision
- **Physical Simulation**: Point-mass vehicle model with lateral grip limits and acceleration constraints
- **PSO Optimization**: Efficient particle swarm optimization with multiple restart capability
- **Visualization**: Built-in plotting for tracks, racing lines, speed profiles, and convergence

## Project Structure

```
├── geometry.py          # Core geometry module (tracks, splines, curvature)
├── lap_simulation.py    # Lap time simulation with physics constraints
├── pso_optimizer.py     # Particle Swarm Optimization implementation
├── example.py           # Example script demonstrating usage
└── requirements.txt     # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sampetch/stint-aware-raceline-pso.git
cd stint-aware-raceline-pso
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the example script to see the optimization in action:

```bash
python example.py
```

This will:
1. Create a simple oval track
2. Optimize the racing line using PSO
3. Generate visualization plots showing baseline vs. optimized racing lines
4. Display lap time improvements

## Usage

### Basic Usage

```python
from geometry import create_simple_oval_track
from lap_simulation import LapSimulator
from pso_optimizer import PSOOptimizer
from lap_simulation import evaluate_racing_line

# Create a track
track = create_simple_oval_track(
    length=500.0,      # Length of straights (m)
    width_track=100.0, # Width of oval (m)
    track_width=12.0,  # Racing surface width (m)
    n_points=80        # Discretization points
)

# Create lap simulator
simulator = LapSimulator(lateral_g=2.0)  # 2g lateral acceleration limit

# Define objective function
def objective(alpha):
    return evaluate_racing_line(track, alpha, simulator)

# Create and run PSO optimizer
optimizer = PSOOptimizer(
    objective_function=objective,
    dimension=track.n_points,
    n_particles=30,
    bounds=(0.0, 1.0)
)

# Optimize
best_alpha, best_time, info = optimizer.optimize(max_iterations=100)

print(f"Optimized lap time: {best_time:.2f}s")
```

### Custom Track

```python
import numpy as np
from geometry import Track

# Define your own centerline
centerline = np.array([
    [0, 0],
    [100, 0],
    [150, 50],
    [100, 100],
    [0, 100],
    [-50, 50]
])

# Create track
track = Track(centerline, track_width=10.0, closed=True)

# Use with optimization...
```

## Modules

### geometry.py

Core geometric operations for track and racing line representation:

- **Track**: Represents a racing track with centerline and width
  - `get_racing_line(alpha)`: Convert alpha values to racing line coordinates
  - Alpha values: 0 = inner border, 0.5 = centerline, 1 = outer border

- **RacingLineSpline**: Smooth spline representation of racing lines
  - `compute_curvature(s)`: Calculate curvature at given positions
  - `evaluate(s)`: Evaluate spline at parameter values

- **create_simple_oval_track()**: Helper to create test tracks

### lap_simulation.py

Physics-based lap time simulation:

- **LapSimulator**: Computes lap times from racing lines
  - `compute_speed_from_curvature(curvature)`: Speed limits from lateral grip
  - `compute_lap_time(spline)`: Total lap time with acceleration constraints
  - Point-mass model with configurable lateral acceleration limits

- **evaluate_racing_line()**: Objective function for PSO

### pso_optimizer.py

Particle Swarm Optimization implementation:

- **Particle**: Individual particle in the swarm
- **PSOOptimizer**: Main PSO algorithm
  - `optimize(max_iterations)`: Run optimization
  - `optimize_with_restarts(n_restarts)`: Multiple restarts to avoid local minima
  - Configurable inertia weight and learning factors

## Algorithm Overview

### 1. Track Representation
- Track defined by centerline (sequence of x,y points) and width
- Alpha values (0-1) define racing line position between inner/outer borders
- Normal vectors computed to offset from centerline

### 2. Spline Interpolation
- Racing line points converted to smooth spline
- Cubic spline interpolation with arc-length parameterization
- Curvature computed from spline derivatives: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)

### 3. Lap Time Simulation
- Maximum speed from curvature: v = sqrt(a_lat / |κ|)
- Forward/backward passes for acceleration/braking constraints
- Integration along spline to compute total lap time

### 4. PSO Optimization
- Swarm of particles explores alpha value space
- Each particle tracks personal best and contributes to global best
- Velocity update: v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
- Position update with boundary constraints
- Converges to minimum lap time

## Configuration

### Simulator Parameters

```python
simulator = LapSimulator(
    lateral_g=2.0,      # Lateral acceleration limit (g's)
    min_speed=10.0      # Minimum speed constraint (m/s)
)
```

### PSO Parameters

```python
optimizer = PSOOptimizer(
    objective_function=objective,
    dimension=n_points,
    n_particles=30,     # Swarm size
    bounds=(0.0, 1.0),  # Alpha value bounds
    w=0.7,              # Inertia weight
    c1=1.5,             # Cognitive learning factor
    c2=1.5              # Social learning factor
)
```

## Output

The example script generates three visualization files:

1. **baseline_racing_line.png**: Shows centerline (baseline) racing line
2. **optimized_racing_line.png**: Shows PSO-optimized racing line with speed profile
3. **convergence.png**: PSO convergence plot showing lap time improvement

## Performance Considerations

- **Control Points**: Use fewer control points (e.g., 40) for faster optimization, then interpolate to full resolution
- **Particles**: More particles (30-50) explore solution space better but increase computation
- **Iterations**: Typically converges within 50-100 iterations
- **Restarts**: Use `optimize_with_restarts()` for better global optima

## Numerical Stability

The code includes several stability features:

- Minimum thresholds to avoid division by zero
- Bounded alpha values (0-1) with clipping
- Velocity limits in PSO to prevent explosion
- Graceful handling of zero curvature (straight sections)
- Double precision floating point throughout

## Future Enhancements

Potential extensions for this project:

- Tire degradation models (stint-aware optimization)
- Multi-objective optimization (lap time vs. tire wear)
- Dynamic track conditions (wet/dry lines)
- More sophisticated vehicle models
- Real track data import
- 3D elevation support

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. IEEE International Conference on Neural Networks.
- Braghin, F., et al. (2008). Race driver model. Computers & Structures.
- Optimization techniques in motorsport applications
