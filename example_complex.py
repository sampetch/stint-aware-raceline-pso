"""
Advanced example with a more complex track showing better optimization results.

This demonstrates the racing line optimization on a track with varying corner radii,
which typically shows more significant improvements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from geometry import Track, RacingLineSpline
from lap_simulation import LapSimulator, evaluate_racing_line
from pso_optimizer import PSOOptimizer


def create_complex_track(n_points: int = 100) -> Track:
    """
    Create a more complex track with varying corner radii.
    
    This track has:
    - Fast sweeping corners
    - Tight hairpins
    - Straights of different lengths
    
    Args:
        n_points: Number of points to discretize the track
        
    Returns:
        Track object
    """
    # Create a track with multiple distinct sections
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    for i, t in enumerate(theta):
        # Complex shape with varying radius
        # Base radius varies with position
        base_r = 200 + 100 * np.sin(3 * t)
        
        # Add some variation for different corner types
        if t < np.pi / 2:
            # Fast sweeping corner
            r = base_r * 1.2
        elif t < np.pi:
            # Tight hairpin
            r = base_r * 0.6
        elif t < 3 * np.pi / 2:
            # Medium corner
            r = base_r
        else:
            # Another fast corner
            r = base_r * 1.1
        
        x[i] = r * np.cos(t)
        y[i] = r * np.sin(t)
    
    centerline = np.column_stack([x, y])
    
    return Track(centerline, track_width=15.0, closed=True)


def optimize_and_visualize_complex_track():
    """
    Run optimization on complex track and visualize results.
    """
    print("="*70)
    print("Complex Track Racing Line Optimization")
    print("="*70)
    print()
    
    # Create complex track
    print("Creating complex track...")
    track = create_complex_track(n_points=100)
    print(f"Track created: {track.n_points} points, {track.track_width}m width")
    
    # Calculate approximate track length
    track_length = np.sum(np.linalg.norm(np.diff(track.centerline, axis=0), axis=1))
    print(f"Track length: ~{track_length:.1f}m")
    print()
    
    # Create simulator with typical race car parameters
    simulator = LapSimulator(lateral_g=2.5, min_speed=20.0)
    
    # Evaluate baseline (centerline)
    print("Evaluating baseline (centerline)...")
    baseline_alpha = np.full(track.n_points, 0.5)
    baseline_time = evaluate_racing_line(track, baseline_alpha, simulator)
    print(f"Baseline lap time: {baseline_time:.2f}s")
    print()
    
    # Run optimization with more iterations
    print("Running optimization (this may take a minute)...")
    
    def objective(alpha):
        # Interpolate control points to full resolution
        alpha_full = np.interp(
            np.linspace(0, 1, track.n_points),
            np.linspace(0, 1, len(alpha)),
            alpha
        )
        return evaluate_racing_line(track, alpha_full, simulator)
    
    # Use fewer control points for faster optimization
    n_control_points = 50
    
    optimizer = PSOOptimizer(
        objective_function=objective,
        dimension=n_control_points,
        n_particles=40,
        bounds=(0.0, 1.0),
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    
    # Initialize with slight variations around centerline
    initial_guess = np.random.uniform(0.4, 0.6, n_control_points)
    optimizer.initialize_swarm(initial_guess)
    
    # Run optimization
    best_alpha_control, best_time, info = optimizer.optimize(
        max_iterations=100,
        tolerance=1e-6,
        verbose=True
    )
    
    # Interpolate to full resolution
    best_alpha = np.interp(
        np.linspace(0, 1, track.n_points),
        np.linspace(0, 1, n_control_points),
        best_alpha_control
    )
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline lap time:  {baseline_time:.2f}s")
    print(f"Optimized lap time: {best_time:.2f}s")
    improvement = baseline_time - best_time
    print(f"Improvement:        {improvement:.2f}s ({improvement/baseline_time*100:.1f}%)")
    print("="*70)
    print()
    
    # Visualize
    print("Generating visualizations...")
    visualize_comparison(track, baseline_alpha, best_alpha, baseline_time, best_time)
    print("Saved: complex_track_comparison.png")
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(info['best_fitness_history'], linewidth=2, color='#2E86AB')
    ax.axhline(y=baseline_time, color='red', linestyle='--', 
               label=f'Baseline: {baseline_time:.2f}s', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Lap Time (s)', fontsize=12)
    ax.set_title('PSO Convergence on Complex Track', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('complex_track_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: complex_track_convergence.png")
    
    print()
    print("Done! Check the generated PNG files.")


def visualize_comparison(track: Track, baseline_alpha: np.ndarray, 
                        optimized_alpha: np.ndarray,
                        baseline_time: float, optimized_time: float):
    """
    Create a comprehensive comparison visualization.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Track visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Bottom row: Data plots
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    simulator = LapSimulator(lateral_g=2.5, min_speed=20.0)
    
    # Plot baseline track
    plot_racing_line(ax1, track, baseline_alpha, simulator, 
                     f"Baseline (Centerline)\nLap Time: {baseline_time:.2f}s")
    
    # Plot optimized track
    plot_racing_line(ax2, track, optimized_alpha, simulator,
                     f"Optimized (PSO)\nLap Time: {optimized_time:.2f}s")
    
    # Plot overlay
    plot_overlay(ax3, track, baseline_alpha, optimized_alpha)
    
    # Plot alpha comparison
    plot_alpha_comparison(ax4, baseline_alpha, optimized_alpha)
    
    # Plot speed comparison
    plot_speed_comparison(ax5, track, baseline_alpha, optimized_alpha, simulator)
    
    # Plot curvature comparison
    plot_curvature_comparison(ax6, track, baseline_alpha, optimized_alpha)
    
    plt.savefig('complex_track_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_racing_line(ax, track, alpha, simulator, title):
    """Plot a single racing line on track."""
    centerline = track.centerline
    racing_line = track.get_racing_line(alpha)
    
    # Track boundaries
    normals = track._compute_normals()
    inner = centerline - normals * (track.track_width / 2)
    outer = centerline + normals * (track.track_width / 2)
    
    ax.plot(inner[:, 0], inner[:, 1], 'k-', linewidth=1.5)
    ax.plot(outer[:, 0], outer[:, 1], 'k-', linewidth=1.5)
    ax.fill(np.vstack([inner, outer[::-1]])[:, 0],
            np.vstack([inner, outer[::-1]])[:, 1],
            color='gray', alpha=0.2)
    
    # Racing line with speed coloring
    spline = RacingLineSpline(racing_line, closed=track.closed)
    s = np.linspace(0, spline.get_total_length(), 200)
    points = spline.evaluate(s)
    _, info = simulator.compute_lap_time(spline)
    
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    speeds = info['speeds'][:-1] * 3.6  # Convert to km/h
    
    lc = LineCollection(segments, array=speeds, cmap='RdYlGn',
                       linewidth=3, norm=plt.Normalize(vmin=0, vmax=300))
    ax.add_collection(lc)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)


def plot_overlay(ax, track, baseline_alpha, optimized_alpha):
    """Plot both racing lines overlaid."""
    centerline = track.centerline
    baseline_line = track.get_racing_line(baseline_alpha)
    optimized_line = track.get_racing_line(optimized_alpha)
    
    # Track boundaries
    normals = track._compute_normals()
    inner = centerline - normals * (track.track_width / 2)
    outer = centerline + normals * (track.track_width / 2)
    
    ax.plot(inner[:, 0], inner[:, 1], 'k-', linewidth=1.5)
    ax.plot(outer[:, 0], outer[:, 1], 'k-', linewidth=1.5)
    ax.fill(np.vstack([inner, outer[::-1]])[:, 0],
            np.vstack([inner, outer[::-1]])[:, 1],
            color='gray', alpha=0.1)
    
    ax.plot(baseline_line[:, 0], baseline_line[:, 1], 'b--',
            linewidth=2, label='Baseline', alpha=0.7)
    ax.plot(optimized_line[:, 0], optimized_line[:, 1], 'r-',
            linewidth=2, label='Optimized', alpha=0.9)
    
    ax.set_title('Overlay Comparison', fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)


def plot_alpha_comparison(ax, baseline_alpha, optimized_alpha):
    """Plot alpha values comparison."""
    idx = np.arange(len(baseline_alpha))
    ax.plot(idx, baseline_alpha, 'b--', linewidth=2, label='Baseline', alpha=0.7)
    ax.plot(idx, optimized_alpha, 'r-', linewidth=2, label='Optimized', alpha=0.9)
    ax.fill_between(idx, baseline_alpha, optimized_alpha, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Alpha (0=inner, 1=outer)')
    ax.set_title('Racing Line Position', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)


def plot_speed_comparison(ax, track, baseline_alpha, optimized_alpha, simulator):
    """Plot speed profiles."""
    baseline_line = track.get_racing_line(baseline_alpha)
    optimized_line = track.get_racing_line(optimized_alpha)
    
    baseline_spline = RacingLineSpline(baseline_line, closed=track.closed)
    optimized_spline = RacingLineSpline(optimized_line, closed=track.closed)
    
    _, baseline_info = simulator.compute_lap_time(baseline_spline)
    _, optimized_info = simulator.compute_lap_time(optimized_spline)
    
    s_baseline = baseline_info['distances']
    s_optimized = optimized_info['distances']
    
    ax.plot(s_baseline, baseline_info['speeds'] * 3.6, 'b--',
            linewidth=2, label='Baseline', alpha=0.7)
    ax.plot(s_optimized, optimized_info['speeds'] * 3.6, 'r-',
            linewidth=2, label='Optimized', alpha=0.9)
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed Profile', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_curvature_comparison(ax, track, baseline_alpha, optimized_alpha):
    """Plot curvature comparison."""
    baseline_line = track.get_racing_line(baseline_alpha)
    optimized_line = track.get_racing_line(optimized_alpha)
    
    baseline_spline = RacingLineSpline(baseline_line, closed=track.closed)
    optimized_spline = RacingLineSpline(optimized_line, closed=track.closed)
    
    s_baseline = np.linspace(0, baseline_spline.get_total_length(), 200)
    s_optimized = np.linspace(0, optimized_spline.get_total_length(), 200)
    
    curv_baseline = np.abs(baseline_spline.compute_curvature(s_baseline))
    curv_optimized = np.abs(optimized_spline.compute_curvature(s_optimized))
    
    ax.plot(s_baseline, curv_baseline * 1000, 'b--',
            linewidth=2, label='Baseline', alpha=0.7)
    ax.plot(s_optimized, curv_optimized * 1000, 'r-',
            linewidth=2, label='Optimized', alpha=0.9)
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Curvature (×10⁻³ m⁻¹)')
    ax.set_title('Curvature Profile', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    optimize_and_visualize_complex_track()
