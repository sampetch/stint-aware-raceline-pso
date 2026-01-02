# Technical Design Document: Racing Line Optimization using PSO

## Overview

This document describes the technical design and implementation details of the racing line optimization system.

## Architecture

### Module Structure

```
raceline-pso/
├── geometry.py           # Core geometry and spline operations
├── lap_simulation.py     # Physics-based lap time computation
├── pso_optimizer.py      # Particle Swarm Optimization
├── example.py            # Basic demonstration
├── example_complex.py    # Advanced demonstration
├── test_modules.py       # Unit and integration tests
├── __init__.py           # Package initialization
├── setup.py              # Package installation
└── requirements.txt      # Dependencies
```

### Design Principles

1. **Modularity**: Clear separation of concerns between geometry, physics, and optimization
2. **Readability**: Well-documented code with type hints and docstrings
3. **Numerical Stability**: Robust handling of edge cases and numerical precision
4. **Extensibility**: Easy to add new track types, vehicle models, or optimization algorithms

## Module Details

### 1. geometry.py - Core Geometry Module

#### Track Class

**Purpose**: Represent a racing track with centerline and width.

**Key Methods**:
- `__init__(centerline, track_width, closed)`: Initialize track
- `get_racing_line(alpha)`: Convert alpha values to racing line coordinates
- `_compute_normals()`: Compute normal vectors at centerline points

**Implementation Details**:
- Centerline stored as Nx2 numpy array (float64)
- Normal vectors computed using centered differences
- Closed tracks handle wraparound correctly
- Alpha values linearly interpolate between inner (0) and outer (1) borders

**Numerical Considerations**:
- Minimum tangent length threshold (1e-10) prevents division by zero
- Normal vectors normalized for stability
- Smooth tangent computation using averaging

#### RacingLineSpline Class

**Purpose**: Smooth spline representation for interpolation and curvature computation.

**Key Methods**:
- `__init__(points, closed, smoothing)`: Create spline from points
- `evaluate(s)`: Evaluate spline at parameter values
- `compute_curvature(s)`: Compute curvature at parameter values
- `get_total_length()`: Get total arc length

**Implementation Details**:
- Uses scipy.interpolate.UnivariateSpline
- Arc-length parameterization for uniform sampling
- Cubic splines (k=3) for smooth curvature
- Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)

**Numerical Considerations**:
- Denominator clamped to minimum 1e-10
- Double precision throughout
- Smoothing parameter for noisy data (default 0 for interpolation)

### 2. lap_simulation.py - Lap Time Simulation

#### LapSimulator Class

**Purpose**: Compute lap times from racing lines using vehicle dynamics.

**Physics Model**:
- Point-mass vehicle model
- Lateral acceleration limit: a_lat = lateral_g × 9.81 m/s²
- Speed from curvature: v = sqrt(a_lat / |κ|)
- Longitudinal acceleration limits (10 m/s² accel, 15 m/s² braking)

**Key Methods**:
- `compute_speed_from_curvature(curvature)`: Maximum speed from curvature
- `compute_lap_time(spline, n_samples)`: Total lap time with detailed info
- `_apply_acceleration_limits(speeds, distances)`: Refine speed profile

**Implementation Details**:

1. **Initial Speed Profile**:
   - Compute from curvature limits
   - Straight sections: 100 m/s default (power/drag limited)
   - Zero curvature threshold: 1e-6

2. **Forward Pass (Acceleration)**:
   - For each segment: v_max = sqrt(v_prev² + 2a×ds)
   - Take minimum of curvature-limited and acceleration-limited

3. **Backward Pass (Braking)**:
   - Reverse iteration: v_max = sqrt(v_next² + 2a×ds)
   - Ensures vehicle can brake to corner entry speed

4. **Time Integration**:
   - Average speed per segment
   - dt = ds / v_avg
   - Total time = sum(dt)

**Numerical Considerations**:
- Minimum speed constraint prevents division by zero
- Segment distance validation (skip if ds ≤ 0)
- Safe handling of zero curvature
- Exception handling returns large penalty (1e6)

### 3. pso_optimizer.py - Particle Swarm Optimization

#### Particle Class

**Purpose**: Represent individual solution in swarm.

**Attributes**:
- `position`: Current position (alpha values)
- `velocity`: Current velocity
- `best_position`: Personal best position
- `best_fitness`: Personal best fitness

**Update Equations**:

Velocity update:
```
v_new = w×v + c1×r1×(p_best - x) + c2×r2×(g_best - x)
```

Position update:
```
x_new = x + v_new
```

Where:
- w: inertia weight
- c1, c2: learning factors
- r1, r2: random values [0,1]
- p_best: personal best
- g_best: global best

#### PSOOptimizer Class

**Purpose**: Implement PSO algorithm for optimization.

**Algorithm Flow**:
1. Initialize swarm randomly (or from initial guess)
2. Evaluate fitness for all particles
3. Update personal and global bests
4. Update velocities using PSO equation
5. Update positions with boundary constraints
6. Repeat until convergence or max iterations

**Key Methods**:
- `initialize_swarm(initial_guess)`: Initialize particle positions
- `evaluate_fitness()`: Evaluate and update bests
- `update_swarm()`: Update velocities and positions
- `optimize(max_iterations, tolerance)`: Main optimization loop
- `optimize_with_restarts(n_restarts)`: Multiple restart strategy

**Implementation Details**:
- Velocity clamping: ±20% of search space
- Boundary handling: clip to [0, 1]
- Convergence check: improvement < tolerance
- History tracking for analysis

**Numerical Considerations**:
- Velocity limits prevent explosion
- Random initialization within bounds
- Safe fitness evaluation with try-catch

### 4. Integration and Workflow

#### Typical Optimization Workflow

```
1. Create Track
   └─> Define centerline and width
   
2. Create Simulator
   └─> Set vehicle parameters (lateral_g)
   
3. Define Objective Function
   └─> Maps alpha → lap_time
   └─> Uses track.get_racing_line()
   └─> Creates spline
   └─> Computes lap time
   
4. Create PSO Optimizer
   └─> Set dimension (number of alpha values)
   └─> Set swarm parameters
   
5. Run Optimization
   └─> Iteratively improve racing line
   └─> Track convergence
   
6. Analyze Results
   └─> Visualize racing line
   └─> Plot speed profile
   └─> Compare to baseline
```

## Performance Characteristics

### Computational Complexity

- **Track Creation**: O(n) where n = number of points
- **Racing Line Generation**: O(n) for normal computation
- **Spline Creation**: O(n log n) for scipy spline fitting
- **Curvature Computation**: O(m) where m = sample points
- **Lap Time Computation**: O(m) for integration
- **PSO Iteration**: O(p × f) where p = particles, f = fitness cost
- **Total Optimization**: O(i × p × (n + m)) where i = iterations

### Typical Performance

On a modern CPU:
- Simple track (50 points, 30 particles, 50 iterations): ~0.5-1s
- Complex track (100 points, 40 particles, 100 iterations): ~2-5s
- Large optimization (200 points, 50 particles, 200 iterations): ~20-60s

### Optimization Tips

1. **Reduce Control Points**: Use 40-50 control points, interpolate to full resolution
2. **Parallel Evaluation**: Fitness evaluations are independent (not implemented)
3. **Early Stopping**: Use convergence tolerance to stop early
4. **Smart Initialization**: Start from good guess (e.g., centerline)

## Testing Strategy

### Unit Tests

- **Geometry Module**:
  - Track creation
  - Alpha to racing line conversion
  - Spline interpolation
  - Curvature computation

- **Lap Simulation**:
  - Speed from curvature
  - Lap time computation
  - Acceleration limits

- **PSO Optimizer**:
  - Particle creation
  - Velocity/position updates
  - Simple optimization test

### Integration Tests

- Full optimization workflow
- Result validation (time improvement)
- Visualization generation

### Validation Methods

1. **Physical Plausibility**: Racing lines should be smooth and reasonable
2. **Performance**: Optimization should improve or match baseline
3. **Convergence**: PSO should converge within iterations
4. **Numerical Stability**: No NaN, Inf, or crashes

## Extensibility

### Adding New Features

#### Custom Vehicle Model

```python
class AdvancedSimulator(LapSimulator):
    def compute_speed_from_curvature(self, curvature):
        # Add tire model, aerodynamics, etc.
        pass
```

#### Different Optimization Algorithm

```python
class GeneticAlgorithm:
    def optimize(self, objective_function):
        # Implement GA
        pass
```

#### 3D Tracks with Elevation

```python
class Track3D(Track):
    def __init__(self, centerline, elevation, track_width):
        # Add elevation dimension
        pass
```

## Known Limitations

1. **Point-Mass Model**: Simplified vehicle dynamics (no weight transfer, tire model)
2. **2D Only**: No elevation changes or banking
3. **Single Lap**: No tire degradation or fuel consumption
4. **Fixed Parameters**: Lateral g is constant (no aerodynamic/speed dependence)
5. **No Obstacles**: Assumes clear track
6. **Simplified Constraints**: Only alpha bounds, no collision detection

## Future Enhancements

1. **Tire Degradation**: Add stint-aware optimization
2. **Multi-Lap Strategy**: Optimize fuel/tire strategy
3. **Real Track Import**: Load from GPS/telemetry data
4. **Advanced Vehicle Model**: More realistic physics
5. **Multi-Objective**: Balance lap time vs. tire wear
6. **Parallel Processing**: Speed up fitness evaluation
7. **Real-Time Visualization**: Animated optimization progress
8. **Track Database**: Library of real-world circuits

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization
- Brayshaw, D., & Harrison, M. (2005). Use of numerical optimization to determine racing line
- Lot, R., & Dal Bianco, N. (2016). Lap time optimization of a racing go-kart
- Scipy documentation: https://docs.scipy.org/doc/scipy/reference/interpolate.html
