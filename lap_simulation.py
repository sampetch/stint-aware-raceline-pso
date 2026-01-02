"""
Lap simulation module for computing lap times from racing lines.

This module calculates vehicle speed based on curvature and lateral grip limits,
then computes the total lap time.
"""

import numpy as np
from typing import Optional, Tuple
from geometry import RacingLineSpline


class LapSimulator:
    """
    Simulates lap time for a given racing line.
    
    Uses a point-mass model with lateral grip constraints to compute
    maximum speed through each section of the track.
    """
    
    def __init__(self, lateral_g: float = 2.0, min_speed: float = 10.0):
        """
        Initialize the lap simulator.
        
        Args:
            lateral_g: Maximum lateral acceleration in g's (typically 1.5-3.0 for race cars)
            min_speed: Minimum speed constraint in m/s
        """
        self.lateral_g = lateral_g
        self.min_speed = min_speed
        self.g = 9.81  # Gravitational acceleration in m/s²
        self.max_lateral_accel = self.lateral_g * self.g
    
    def compute_speed_from_curvature(self, curvature: np.ndarray) -> np.ndarray:
        """
        Compute maximum speed based on curvature and lateral grip.
        
        For a given curvature κ and lateral acceleration limit a_lat:
        v_max = sqrt(a_lat / |κ|)
        
        Args:
            curvature: Array of curvature values (1/radius)
            
        Returns:
            Array of maximum speeds in m/s
        """
        # Handle zero curvature (straight sections)
        abs_curvature = np.abs(curvature)
        
        # Compute speed from curvature
        # v² = a_lat / κ  =>  v = sqrt(a_lat / κ)
        # For zero curvature, speed is limited by longitudinal acceleration
        speed = np.zeros_like(curvature)
        
        # For curved sections
        mask = abs_curvature > 1e-6
        speed[mask] = np.sqrt(self.max_lateral_accel / abs_curvature[mask])
        
        # For straight sections, use a high default speed
        # (in practice, this would be limited by power/drag)
        speed[~mask] = 100.0  # m/s (~360 km/h, typical max for race cars)
        
        # Apply minimum speed constraint
        speed = np.maximum(speed, self.min_speed)
        
        return speed
    
    def compute_lap_time(self, spline: RacingLineSpline, n_samples: int = 500) -> Tuple[float, dict]:
        """
        Compute lap time for a racing line spline.
        
        Args:
            spline: RacingLineSpline object representing the racing line
            n_samples: Number of points to sample along the spline for integration
            
        Returns:
            Tuple of (lap_time, info_dict) where info_dict contains:
                - speeds: Array of speeds at sample points
                - curvatures: Array of curvatures at sample points
                - distances: Array of distance intervals
                - sample_points: Parameter values where samples were taken
        """
        # Sample points along the spline
        s_max = spline.get_total_length()
        s_samples = np.linspace(0, s_max, n_samples)
        
        # Compute curvature at sample points
        curvatures = spline.compute_curvature(s_samples)
        
        # Compute speeds from curvature
        speeds = self.compute_speed_from_curvature(curvatures)
        
        # Refine speeds with forward/backward passes for acceleration limits
        speeds = self._apply_acceleration_limits(speeds, s_samples)
        
        # Compute time for each segment
        # ds = distance, v = speed, dt = ds / v
        ds = np.diff(s_samples)
        avg_speeds = (speeds[:-1] + speeds[1:]) / 2  # Average speed in each segment
        avg_speeds = np.maximum(avg_speeds, self.min_speed)  # Ensure no division by zero
        
        dt = ds / avg_speeds
        
        # Total lap time
        lap_time = np.sum(dt)
        
        # Return info for analysis
        info = {
            'speeds': speeds,
            'curvatures': curvatures,
            'distances': s_samples,
            'sample_points': s_samples,
            'segment_times': dt
        }
        
        return lap_time, info
    
    def _apply_acceleration_limits(self, speeds: np.ndarray, distances: np.ndarray,
                                   max_accel: float = 10.0, max_decel: float = 15.0) -> np.ndarray:
        """
        Apply longitudinal acceleration limits to speed profile.
        
        This ensures the vehicle can actually achieve the computed speeds
        given braking and acceleration constraints.
        
        Args:
            speeds: Initial speed array (from curvature limits)
            distances: Distance parameter values
            max_accel: Maximum longitudinal acceleration in m/s²
            max_decel: Maximum longitudinal deceleration (braking) in m/s²
            
        Returns:
            Refined speed array
        """
        n = len(speeds)
        refined_speeds = speeds.copy()
        
        # Forward pass: limit acceleration
        for i in range(1, n):
            ds = distances[i] - distances[i-1]
            if ds <= 0:
                continue
                
            # Maximum speed achievable from previous speed with acceleration limit
            # v² = v₀² + 2as  =>  v = sqrt(v₀² + 2as)
            v_prev = refined_speeds[i-1]
            v_max_accel = np.sqrt(v_prev**2 + 2 * max_accel * ds)
            
            # Take minimum of curvature-limited and acceleration-limited speed
            refined_speeds[i] = min(refined_speeds[i], v_max_accel)
        
        # Backward pass: limit deceleration
        for i in range(n-2, -1, -1):
            ds = distances[i+1] - distances[i]
            if ds <= 0:
                continue
                
            # Maximum speed from which we can brake to next speed
            # v₀² = v² - 2as  =>  v₀ = sqrt(v² + 2as)
            v_next = refined_speeds[i+1]
            v_max_decel = np.sqrt(v_next**2 + 2 * max_decel * ds)
            
            # Take minimum
            refined_speeds[i] = min(refined_speeds[i], v_max_decel)
        
        return refined_speeds
    
    def compute_sector_times(self, spline: RacingLineSpline, 
                            sector_boundaries: np.ndarray,
                            n_samples: int = 500) -> Tuple[np.ndarray, dict]:
        """
        Compute sector times for a racing line.
        
        Args:
            spline: RacingLineSpline object
            sector_boundaries: Array of parameter values defining sector boundaries
            n_samples: Number of points to sample for integration
            
        Returns:
            Tuple of (sector_times, info_dict)
        """
        lap_time, info = self.compute_lap_time(spline, n_samples)
        
        # Map sample points to sectors
        sample_points = info['sample_points']
        segment_times = info['segment_times']
        
        # Initialize sector times
        n_sectors = len(sector_boundaries) - 1
        sector_times = np.zeros(n_sectors)
        
        # Accumulate times for each sector
        for i in range(len(segment_times)):
            s_start = sample_points[i]
            s_end = sample_points[i + 1]
            
            # Find which sector this segment belongs to
            for j in range(n_sectors):
                if sector_boundaries[j] <= s_start < sector_boundaries[j + 1]:
                    sector_times[j] += segment_times[i]
                    break
        
        return sector_times, info


def evaluate_racing_line(track, alpha: np.ndarray, simulator: LapSimulator) -> float:
    """
    Evaluate lap time for a racing line defined by alpha values.
    
    This is the objective function for the PSO optimizer.
    
    Args:
        track: Track object
        alpha: Array of alpha values defining the racing line
        simulator: LapSimulator object
        
    Returns:
        Lap time in seconds
    """
    try:
        # Convert alpha to racing line
        racing_line = track.get_racing_line(alpha)
        
        # Create spline
        spline = RacingLineSpline(racing_line, closed=track.closed)
        
        # Compute lap time
        lap_time, _ = simulator.compute_lap_time(spline)
        
        return lap_time
        
    except Exception as e:
        # Return a large penalty for invalid racing lines
        return 1e6


def validate_racing_line(alpha: np.ndarray, smoothness_penalty: float = 0.0) -> float:
    """
    Compute a penalty for racing line quality.
    
    Args:
        alpha: Array of alpha values
        smoothness_penalty: Weight for smoothness penalty
        
    Returns:
        Penalty value (0 for valid, smooth lines)
    """
    penalty = 0.0
    
    # Check bounds
    if np.any(alpha < 0) or np.any(alpha > 1):
        penalty += 1e6
    
    # Smoothness penalty (penalize large changes in alpha)
    if smoothness_penalty > 0:
        differences = np.diff(alpha)
        smoothness = np.sum(differences**2)
        penalty += smoothness_penalty * smoothness
    
    return penalty
