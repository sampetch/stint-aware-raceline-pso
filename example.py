"""
Example script demonstrating racing line optimization using PSO.

This script creates a simple oval track, optimizes the racing line,
and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from typing import Optional

from geometry import Track, RacingLineSpline, create_simple_oval_track
from lap_simulation import LapSimulator, evaluate_racing_line
from pso_optimizer import PSOOptimizer


def visualize_track_and_racing_line(track: Track, alpha: np.ndarray,
                                   title: str = "Racing Line Optimization"):
    """
    Visualize the track with its racing line.
    
    Args:
        track: Track object
        alpha: Array of alpha values defining the racing line
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Track and racing line
    centerline = track.centerline
    racing_line = track.get_racing_line(alpha)
    
    # Compute track boundaries
    normals = track._compute_normals()
    inner_boundary = centerline - normals * (track.track_width / 2)
    outer_boundary = centerline + normals * (track.track_width / 2)
    
    # Plot track boundaries
    ax1.plot(inner_boundary[:, 0], inner_boundary[:, 1], 'k-', linewidth=2, label='Track boundaries')
    ax1.plot(outer_boundary[:, 0], outer_boundary[:, 1], 'k-', linewidth=2)
    
    # Fill track area
    if track.closed:
        track_polygon = np.vstack([inner_boundary, outer_boundary[::-1]])
        ax1.fill(track_polygon[:, 0], track_polygon[:, 1], color='gray', alpha=0.2)
    
    # Plot centerline
    ax1.plot(centerline[:, 0], centerline[:, 1], 'b--', linewidth=1, alpha=0.5, label='Centerline')
    
    # Plot racing line with color gradient based on curvature
    spline = RacingLineSpline(racing_line, closed=track.closed)
    s_samples = np.linspace(0, spline.get_total_length(), 200)
    points = spline.evaluate(s_samples)
    curvature = spline.compute_curvature(s_samples)
    
    # Create colored line segments
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    curvature_colors = np.abs(curvature[:-1])
    
    lc = LineCollection(segments, array=curvature_colors, cmap='coolwarm', 
                       linewidth=3, label='Racing line')
    ax1.add_collection(lc)
    
    cbar = plt.colorbar(lc, ax=ax1)
    cbar.set_label('Curvature (1/m)', rotation=270, labelpad=20)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(title)
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Alpha values and speed profile
    ax2_twin = ax2.twinx()
    
    # Plot alpha values
    point_indices = np.arange(len(alpha))
    ax2.plot(point_indices, alpha, 'b-', linewidth=2, label='Alpha values')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Centerline (α=0.5)')
    ax2.fill_between(point_indices, 0, 1, color='gray', alpha=0.1)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Alpha (0=inner, 1=outer)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Plot speed profile
    simulator = LapSimulator()
    lap_time, info = simulator.compute_lap_time(spline)
    
    # Map speeds back to point indices (approximately)
    speed_indices = np.linspace(0, len(alpha)-1, len(info['speeds']))
    ax2_twin.plot(speed_indices, info['speeds'] * 3.6, 'r-', linewidth=2, label='Speed (km/h)')
    ax2_twin.set_ylabel('Speed (km/h)', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title(f'Racing Line Parameters\nLap Time: {lap_time:.2f}s')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig


def optimize_racing_line(track: Track, n_control_points: Optional[int] = None,
                         n_particles: int = 30, max_iterations: int = 100,
                         verbose: bool = True):
    """
    Optimize racing line for a given track.
    
    Args:
        track: Track object
        n_control_points: Number of control points (alpha values). 
                         If None, uses track's n_points
        n_particles: Number of PSO particles
        max_iterations: Maximum PSO iterations
        verbose: Whether to print progress
        
    Returns:
        Tuple of (optimized_alpha, lap_time, info)
    """
    if n_control_points is None:
        n_control_points = track.n_points
    
    # Create lap simulator
    simulator = LapSimulator(lateral_g=2.0)
    
    # Define objective function
    def objective_function(alpha):
        # If using fewer control points, interpolate to full resolution
        if len(alpha) < track.n_points:
            alpha_full = np.interp(
                np.linspace(0, 1, track.n_points),
                np.linspace(0, 1, len(alpha)),
                alpha
            )
        else:
            alpha_full = alpha
        
        return evaluate_racing_line(track, alpha_full, simulator)
    
    # Create PSO optimizer
    optimizer = PSOOptimizer(
        objective_function=objective_function,
        dimension=n_control_points,
        n_particles=n_particles,
        bounds=(0.0, 1.0),
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    
    # Initial guess: centerline (alpha = 0.5)
    initial_guess = np.full(n_control_points, 0.5)
    optimizer.initialize_swarm(initial_guess)
    
    # Run optimization
    if verbose:
        print("Starting racing line optimization...")
        print(f"Track points: {track.n_points}")
        print(f"Control points: {n_control_points}")
        print(f"PSO particles: {n_particles}")
        print(f"Max iterations: {max_iterations}")
        print()
    
    best_alpha, best_fitness, info = optimizer.optimize(
        max_iterations=max_iterations,
        tolerance=1e-6,
        verbose=verbose
    )
    
    # Interpolate to full resolution if needed
    if len(best_alpha) < track.n_points:
        best_alpha_full = np.interp(
            np.linspace(0, 1, track.n_points),
            np.linspace(0, 1, len(best_alpha)),
            best_alpha
        )
    else:
        best_alpha_full = best_alpha
    
    return best_alpha_full, best_fitness, info


def main():
    """
    Main example: Optimize racing line for an oval track.
    """
    print("="*70)
    print("Racing Line Optimization using Particle Swarm Optimization (PSO)")
    print("="*70)
    print()
    
    # Create an oval track
    print("Creating oval track...")
    track = create_simple_oval_track(
        length=500.0,      # 500m straights
        width_track=100.0, # 100m wide (50m radius turns)
        track_width=12.0,  # 12m racing surface width
        n_points=80        # 80 points for discretization
    )
    print(f"Track created: {track.n_points} points, {track.track_width}m width")
    print(f"Track length: ~{np.sum(np.linalg.norm(np.diff(track.centerline, axis=0), axis=1)):.1f}m")
    print()
    
    # Evaluate baseline (centerline)
    print("Evaluating baseline (centerline)...")
    simulator = LapSimulator(lateral_g=2.0)
    baseline_alpha = np.full(track.n_points, 0.5)
    baseline_time = evaluate_racing_line(track, baseline_alpha, simulator)
    print(f"Baseline lap time: {baseline_time:.2f}s")
    print()
    
    # Optimize racing line
    optimized_alpha, optimized_time, info = optimize_racing_line(
        track=track,
        n_control_points=40,  # Use fewer control points for faster optimization
        n_particles=30,
        max_iterations=50,
        verbose=True
    )
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline lap time:  {baseline_time:.2f}s")
    print(f"Optimized lap time: {optimized_time:.2f}s")
    print(f"Improvement:        {baseline_time - optimized_time:.2f}s ({(baseline_time - optimized_time)/baseline_time*100:.1f}%)")
    print("="*70)
    print()
    
    # Visualize results
    print("Generating visualization...")
    
    # Plot baseline
    fig1 = visualize_track_and_racing_line(
        track, baseline_alpha,
        title="Baseline Racing Line (Centerline)"
    )
    plt.savefig('baseline_racing_line.png', dpi=150, bbox_inches='tight')
    print("Saved: baseline_racing_line.png")
    
    # Plot optimized
    fig2 = visualize_track_and_racing_line(
        track, optimized_alpha,
        title="Optimized Racing Line (PSO)"
    )
    plt.savefig('optimized_racing_line.png', dpi=150, bbox_inches='tight')
    print("Saved: optimized_racing_line.png")
    
    # Plot convergence
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(info['best_fitness_history'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Lap Time (s)')
    ax.set_title('PSO Convergence')
    ax.grid(True, alpha=0.3)
    plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: convergence.png")
    
    print()
    print("Optimization complete! Check the generated PNG files for visualizations.")
    
    # Show plots (comment out if running headless)
    # plt.show()


if __name__ == "__main__":
    main()
