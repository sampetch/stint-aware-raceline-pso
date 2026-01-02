# Racing Line Optimization - Usage Guide

This guide provides detailed instructions on how to use the racing line optimization package.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding the Modules](#understanding-the-modules)
3. [Creating Tracks](#creating-tracks)
4. [Running Optimization](#running-optimization)
5. [Customizing Parameters](#customizing-parameters)
6. [Visualization](#visualization)
7. [Advanced Usage](#advanced-usage)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sampetch/stint-aware-raceline-pso.git
cd stint-aware-raceline-pso

# Install dependencies
pip install -r requirements.txt

# Run basic example
python example.py

# Run complex track example
python example_complex.py
```

### Minimal Example

```python
from geometry import create_simple_oval_track
from lap_simulation import LapSimulator, evaluate_racing_line
from pso_optimizer import PSOOptimizer
import numpy as np

# Create track
track = create_simple_oval_track(n_points=50)

# Create simulator
simulator = LapSimulator(lateral_g=2.0)

# Define objective function
def objective(alpha):
    return evaluate_racing_line(track, alpha, simulator)

# Optimize
optimizer = PSOOptimizer(objective, dimension=50)
best_alpha, best_time, info = optimizer.optimize(max_iterations=50)

print(f"Optimized lap time: {best_time:.2f}s")
```

## Understanding the Modules

### geometry.py - Track Geometry

The geometry module handles all spatial aspects of the track and racing line.

**Track Class**
- Represents a racing track with centerline and width
- Centerline: Sequence of (x, y) coordinates
- Width: Total racing surface width

**Alpha Values**
- Define racing line position between track borders
- 0.0 = inner border (tight line)
- 0.5 = centerline
- 1.0 = outer border (wide line)

**RacingLineSpline**
- Smooth spline interpolation of racing line
- Computes curvature from derivatives
- Arc-length parameterization

### lap_simulation.py - Physics Simulation

Computes lap times based on vehicle physics.

**LapSimulator Parameters:**
- `lateral_g`: Lateral acceleration limit in g's (typical: 1.5-3.0)
- `min_speed`: Minimum speed constraint in m/s

**Physics Model:**
- Point-mass vehicle model
- Speed from curvature: v = sqrt(a_lat / κ)
- Acceleration/braking constraints
- Forward and backward passes for realistic speed profile

### pso_optimizer.py - Optimization

Particle Swarm Optimization implementation.

**Key Parameters:**
- `n_particles`: Swarm size (typical: 20-50)
- `w`: Inertia weight (typical: 0.5-0.9)
- `c1`: Cognitive learning factor (typical: 1.5-2.0)
- `c2`: Social learning factor (typical: 1.5-2.0)

## Creating Tracks

### Using Pre-built Tracks

```python
from geometry import create_simple_oval_track

# Simple oval
track = create_simple_oval_track(
    length=500.0,      # Straight length in meters
    width_track=100.0, # Oval width in meters
    track_width=12.0,  # Racing surface width
    n_points=80        # Number of discretization points
)
```

### Custom Track from Centerline

```python
import numpy as np
from geometry import Track

# Define centerline points
centerline = np.array([
    [0, 0],
    [100, 0],
    [150, 50],
    [150, 150],
    [100, 200],
    [0, 200],
    [-50, 150],
    [-50, 50],
])

# Create track
track = Track(
    centerline=centerline,
    track_width=10.0,  # Width in meters
    closed=True        # Closed loop
)
```

### Loading Track from File

```python
import numpy as np
from geometry import Track

# Load from CSV (x, y columns)
centerline = np.loadtxt('track_centerline.csv', delimiter=',')

track = Track(centerline, track_width=12.0, closed=True)
```

## Running Optimization

### Basic Optimization

```python
from pso_optimizer import PSOOptimizer

# Define objective function
def objective(alpha):
    return evaluate_racing_line(track, alpha, simulator)

# Create optimizer
optimizer = PSOOptimizer(
    objective_function=objective,
    dimension=track.n_points,
    n_particles=30,
    bounds=(0.0, 1.0)
)

# Run optimization
best_alpha, best_time, info = optimizer.optimize(
    max_iterations=100,
    verbose=True
)
```

### Optimization with Multiple Restarts

```python
# Use multiple restarts to avoid local minima
best_alpha, best_time, info = optimizer.optimize_with_restarts(
    n_restarts=3,
    max_iterations_per_restart=50,
    verbose=True
)
```

### Using Fewer Control Points

For faster optimization, use fewer control points and interpolate:

```python
n_control = 40  # Fewer control points

def objective(alpha_control):
    # Interpolate to full resolution
    alpha_full = np.interp(
        np.linspace(0, 1, track.n_points),
        np.linspace(0, 1, len(alpha_control)),
        alpha_control
    )
    return evaluate_racing_line(track, alpha_full, simulator)

optimizer = PSOOptimizer(objective, dimension=n_control)
alpha_control, best_time, info = optimizer.optimize()

# Get full resolution result
alpha_full = np.interp(
    np.linspace(0, 1, track.n_points),
    np.linspace(0, 1, n_control),
    alpha_control
)
```

## Customizing Parameters

### Vehicle Parameters

```python
# High-performance race car
simulator = LapSimulator(
    lateral_g=3.0,      # High downforce
    min_speed=30.0      # Higher minimum speed
)

# Lower-performance car
simulator = LapSimulator(
    lateral_g=1.5,      # Less grip
    min_speed=15.0      # Lower minimum speed
)
```

### PSO Parameters

```python
# Aggressive exploration
optimizer = PSOOptimizer(
    objective_function=objective,
    dimension=n_points,
    n_particles=50,     # More particles
    w=0.9,              # High inertia
    c1=2.0,             # High cognitive
    c2=2.0              # High social
)

# Conservative convergence
optimizer = PSOOptimizer(
    objective_function=objective,
    dimension=n_points,
    n_particles=20,     # Fewer particles
    w=0.5,              # Low inertia
    c1=1.0,             # Low cognitive
    c2=1.0              # Low social
)
```

## Visualization

### Basic Visualization

```python
import matplotlib.pyplot as plt
from geometry import RacingLineSpline

# Get racing line
racing_line = track.get_racing_line(alpha)

# Plot track boundaries
normals = track._compute_normals()
inner = track.centerline - normals * (track.track_width / 2)
outer = track.centerline + normals * (track.track_width / 2)

plt.figure(figsize=(12, 8))
plt.plot(inner[:, 0], inner[:, 1], 'k-', label='Inner')
plt.plot(outer[:, 0], outer[:, 1], 'k-', label='Outer')
plt.plot(racing_line[:, 0], racing_line[:, 1], 'r-', linewidth=2, label='Racing Line')
plt.legend()
plt.axis('equal')
plt.show()
```

### Speed Profile Visualization

```python
# Compute lap time with info
spline = RacingLineSpline(racing_line, closed=track.closed)
lap_time, info = simulator.compute_lap_time(spline)

# Plot speed profile
plt.figure(figsize=(12, 6))
plt.plot(info['distances'], info['speeds'] * 3.6)
plt.xlabel('Distance (m)')
plt.ylabel('Speed (km/h)')
plt.title(f'Speed Profile (Lap Time: {lap_time:.2f}s)')
plt.grid(True)
plt.show()
```

### Convergence Plot

```python
# After optimization
plt.figure(figsize=(10, 6))
plt.plot(info['best_fitness_history'])
plt.xlabel('Iteration')
plt.ylabel('Best Lap Time (s)')
plt.title('PSO Convergence')
plt.grid(True)
plt.show()
```

## Advanced Usage

### Custom Objective Function

Add penalties for specific behaviors:

```python
def custom_objective(alpha):
    # Base lap time
    lap_time = evaluate_racing_line(track, alpha, simulator)
    
    # Penalty for aggressive line changes
    smoothness_penalty = 10.0 * np.sum(np.diff(alpha)**2)
    
    # Penalty for using outer edge
    outer_penalty = 5.0 * np.sum((alpha - 0.5)**2)
    
    return lap_time + smoothness_penalty + outer_penalty
```

### Sector Time Optimization

```python
# Define sector boundaries
sector_bounds = np.array([0, 300, 600, 900, 1200])

# Compute sector times
sector_times, info = simulator.compute_sector_times(
    spline, sector_bounds
)

print("Sector times:", sector_times)
```

### Parallel Evaluation

For large optimizations, parallelize fitness evaluation:

```python
from multiprocessing import Pool

def evaluate_batch(alpha_batch):
    return [evaluate_racing_line(track, alpha, simulator) 
            for alpha in alpha_batch]

# Use in custom PSO implementation
```

### Saving and Loading Results

```python
import pickle

# Save results
results = {
    'track': track,
    'alpha': best_alpha,
    'lap_time': best_time,
    'info': info
}

with open('optimization_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Load results
with open('optimization_results.pkl', 'rb') as f:
    results = pickle.load(f)
```

## Tips and Best Practices

1. **Start Simple**: Begin with fewer control points for faster iteration
2. **Multiple Restarts**: Use restarts to avoid local minima
3. **Parameter Tuning**: Adjust PSO parameters based on convergence behavior
4. **Validate Results**: Always visualize the racing line to ensure physical plausibility
5. **Numerical Stability**: The code handles edge cases, but extreme parameters may cause issues
6. **Performance**: Reduce n_particles or max_iterations for faster optimization during development

## Troubleshooting

**Problem**: Optimization doesn't improve lap time
- Solution: Try more particles, longer iterations, or multiple restarts

**Problem**: Racing line looks unnatural
- Solution: Add smoothness penalty to objective function

**Problem**: Speed profile shows discontinuities
- Solution: Increase spline sampling points (n_samples parameter)

**Problem**: Optimization is too slow
- Solution: Use fewer control points, fewer particles, or shorter tracks

## References

- PSO Algorithm: Kennedy & Eberhart (1995)
- Racing Line Theory: Optimal control theory
- Vehicle Dynamics: Point-mass model with lateral acceleration limits
