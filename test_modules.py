"""
Simple test script to validate the racing line optimization modules.
"""

import numpy as np
import sys

def test_geometry():
    """Test geometry module."""
    print("Testing geometry module...")
    from geometry import Track, RacingLineSpline, create_simple_oval_track
    
    # Test track creation
    track = create_simple_oval_track(length=500, width_track=100, track_width=12, n_points=50)
    assert track.n_points == 50, "Track should have 50 points"
    assert track.track_width == 12, "Track width should be 12"
    print("  ✓ Track creation successful")
    
    # Test racing line from alpha values
    alpha = np.full(50, 0.5)  # Centerline
    racing_line = track.get_racing_line(alpha)
    assert racing_line.shape == (50, 2), "Racing line should be 50x2"
    print("  ✓ Racing line generation successful")
    
    # Test spline creation
    spline = RacingLineSpline(racing_line, closed=True)
    curvature = spline.compute_curvature()
    assert len(curvature) == 50, "Curvature array should have 50 points"
    print("  ✓ Spline and curvature computation successful")
    
    print("Geometry module: PASSED ✓\n")
    return True

def test_lap_simulation():
    """Test lap simulation module."""
    print("Testing lap simulation module...")
    from geometry import create_simple_oval_track, RacingLineSpline
    from lap_simulation import LapSimulator, evaluate_racing_line
    
    # Create test track
    track = create_simple_oval_track(length=500, width_track=100, track_width=12, n_points=50)
    
    # Create simulator
    simulator = LapSimulator(lateral_g=2.0)
    print("  ✓ Simulator creation successful")
    
    # Test speed from curvature
    curvature = np.array([0.01, 0.02, 0.0, 0.005])
    speeds = simulator.compute_speed_from_curvature(curvature)
    assert len(speeds) == 4, "Speed array should have 4 elements"
    assert np.all(speeds > 0), "All speeds should be positive"
    print("  ✓ Speed computation successful")
    
    # Test lap time computation
    alpha = np.full(50, 0.5)
    racing_line = track.get_racing_line(alpha)
    spline = RacingLineSpline(racing_line, closed=True)
    lap_time, info = simulator.compute_lap_time(spline)
    assert lap_time > 0, "Lap time should be positive"
    assert 'speeds' in info, "Info should contain speeds"
    print(f"  ✓ Lap time computation successful (time: {lap_time:.2f}s)")
    
    # Test evaluate_racing_line
    lap_time2 = evaluate_racing_line(track, alpha, simulator)
    assert lap_time2 > 0, "Lap time should be positive"
    print(f"  ✓ evaluate_racing_line successful (time: {lap_time2:.2f}s)")
    
    print("Lap simulation module: PASSED ✓\n")
    return True

def test_pso_optimizer():
    """Test PSO optimizer module."""
    print("Testing PSO optimizer module...")
    from pso_optimizer import Particle, PSOOptimizer
    
    # Test particle creation
    particle = Particle(dimension=10, bounds=(0.0, 1.0))
    assert particle.dimension == 10, "Particle dimension should be 10"
    assert len(particle.position) == 10, "Position should have 10 elements"
    assert np.all(particle.position >= 0) and np.all(particle.position <= 1), "Position should be in bounds"
    print("  ✓ Particle creation successful")
    
    # Test simple optimization (minimize sum of squares)
    def simple_objective(x):
        return np.sum((x - 0.5)**2)
    
    optimizer = PSOOptimizer(
        objective_function=simple_objective,
        dimension=5,
        n_particles=10,
        bounds=(0.0, 1.0)
    )
    print("  ✓ Optimizer creation successful")
    
    # Run optimization
    best_position, best_fitness, info = optimizer.optimize(max_iterations=20, verbose=False)
    assert len(best_position) == 5, "Best position should have 5 elements"
    assert best_fitness >= 0, "Best fitness should be non-negative"
    assert best_fitness < 0.1, "Should converge to near 0 (all values near 0.5)"
    print(f"  ✓ Optimization successful (fitness: {best_fitness:.6f})")
    
    print("PSO optimizer module: PASSED ✓\n")
    return True

def test_integration():
    """Test full integration."""
    print("Testing full integration...")
    from geometry import create_simple_oval_track
    from lap_simulation import LapSimulator, evaluate_racing_line
    from pso_optimizer import PSOOptimizer
    
    # Create track
    track = create_simple_oval_track(length=300, width_track=80, track_width=10, n_points=30)
    print("  ✓ Track created")
    
    # Create simulator
    simulator = LapSimulator(lateral_g=2.0)
    
    # Baseline (centerline)
    baseline_alpha = np.full(30, 0.5)
    baseline_time = evaluate_racing_line(track, baseline_alpha, simulator)
    print(f"  ✓ Baseline lap time: {baseline_time:.2f}s")
    
    # Define objective
    def objective(alpha):
        return evaluate_racing_line(track, alpha, simulator)
    
    # Optimize
    optimizer = PSOOptimizer(
        objective_function=objective,
        dimension=30,
        n_particles=10,
        bounds=(0.0, 1.0)
    )
    
    optimizer.initialize_swarm(baseline_alpha)
    best_alpha, best_time, info = optimizer.optimize(max_iterations=10, verbose=False)
    
    print(f"  ✓ Optimized lap time: {best_time:.2f}s")
    
    # Check improvement
    improvement = baseline_time - best_time
    print(f"  ✓ Improvement: {improvement:.2f}s ({improvement/baseline_time*100:.1f}%)")
    
    # Verify improvement (should be better or equal)
    assert best_time <= baseline_time + 0.1, "Optimized time should be better or equal to baseline"
    
    print("Full integration: PASSED ✓\n")
    return True

def main():
    """Run all tests."""
    print("="*70)
    print("RACING LINE OPTIMIZATION - MODULE VALIDATION")
    print("="*70)
    print()
    
    tests = [
        ("Geometry Module", test_geometry),
        ("Lap Simulation Module", test_lap_simulation),
        ("PSO Optimizer Module", test_pso_optimizer),
        ("Full Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"{test_name}: FAILED ✗")
            print(f"Error: {e}\n")
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "PASSED ✓" if result else "FAILED ✗"
        print(f"{test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
