"""
Core geometry module for racing line optimization.

This module handles track representation, racing line generation from alpha values,
spline interpolation, and curvature computation.
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional


class Track:
    """
    Represents a racing track with centerline and width.
    
    The track is defined by a centerline (sequence of x, y points) and a width
    that defines the distance from centerline to the inner and outer borders.
    """
    
    def __init__(self, centerline: np.ndarray, track_width: float, closed: bool = True):
        """
        Initialize a track.
        
        Args:
            centerline: Nx2 array of (x, y) coordinates defining the track centerline
            track_width: Total width of the track (distance from inner to outer border)
            closed: Whether the track forms a closed loop
        """
        if centerline.shape[1] != 2:
            raise ValueError("Centerline must be an Nx2 array of (x, y) coordinates")
        
        self.centerline = np.array(centerline, dtype=np.float64)
        self.track_width = float(track_width)
        self.closed = closed
        self.n_points = len(centerline)
        
    def get_racing_line(self, alpha: np.ndarray) -> np.ndarray:
        """
        Convert alpha values to racing line coordinates.
        
        Alpha values define the position of the racing line between the inner
        and outer track borders:
        - alpha = 0: inner border
        - alpha = 0.5: centerline
        - alpha = 1: outer border
        
        Args:
            alpha: Array of alpha values (one per centerline point), range [0, 1]
            
        Returns:
            Nx2 array of (x, y) coordinates for the racing line
        """
        if len(alpha) != self.n_points:
            raise ValueError(f"Alpha array length ({len(alpha)}) must match "
                           f"centerline points ({self.n_points})")
        
        if np.any(alpha < 0) or np.any(alpha > 1):
            raise ValueError("Alpha values must be in range [0, 1]")
        
        # Compute normal vectors at each centerline point
        normals = self._compute_normals()
        
        # Convert alpha to displacement from centerline
        # alpha = 0 -> -track_width/2 (inner), alpha = 1 -> +track_width/2 (outer)
        displacement = (alpha - 0.5) * self.track_width
        
        # Compute racing line by offsetting centerline along normals
        racing_line = self.centerline + normals * displacement[:, np.newaxis]
        
        return racing_line
    
    def _compute_normals(self) -> np.ndarray:
        """
        Compute normal vectors at each centerline point.
        
        Returns:
            Nx2 array of unit normal vectors
        """
        # Compute tangent vectors
        if self.closed:
            # For closed tracks, use centered differences with wraparound
            tangents = np.zeros_like(self.centerline)
            tangents[:-1] = self.centerline[1:] - self.centerline[:-1]
            tangents[-1] = self.centerline[0] - self.centerline[-1]
            
            # Smooth tangents using centered differences
            tangents_smooth = np.zeros_like(tangents)
            tangents_smooth[0] = (tangents[0] + tangents[-1]) / 2
            tangents_smooth[1:] = (tangents[1:] + tangents[:-1]) / 2
            tangents = tangents_smooth
        else:
            # For open tracks, use forward/backward differences at ends
            tangents = np.zeros_like(self.centerline)
            tangents[1:-1] = self.centerline[2:] - self.centerline[:-2]
            tangents[0] = self.centerline[1] - self.centerline[0]
            tangents[-1] = self.centerline[-1] - self.centerline[-2]
        
        # Normalize tangents
        tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_lengths = np.maximum(tangent_lengths, 1e-10)  # Avoid division by zero
        tangents = tangents / tangent_lengths
        
        # Compute normals (perpendicular to tangents, pointing right)
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]
        
        return normals


class RacingLineSpline:
    """
    Spline representation of a racing line for smooth interpolation and curvature computation.
    """
    
    def __init__(self, points: np.ndarray, closed: bool = True, smoothing: float = 0):
        """
        Initialize a spline from racing line points.
        
        Args:
            points: Nx2 array of (x, y) coordinates
            closed: Whether the line forms a closed loop
            smoothing: Smoothing factor for spline (0 = interpolating spline)
        """
        self.points = np.array(points, dtype=np.float64)
        self.closed = closed
        self.n_points = len(points)
        
        # Create parameter array (cumulative distance along the line)
        self.s = self._compute_arc_length_parameter()
        
        # Create splines for x and y coordinates
        if closed:
            # For closed curves, add periodic boundary conditions
            self.spline_x = interpolate.UnivariateSpline(
                self.s, self.points[:, 0], s=smoothing, k=3, ext='const'
            )
            self.spline_y = interpolate.UnivariateSpline(
                self.s, self.points[:, 1], s=smoothing, k=3, ext='const'
            )
        else:
            self.spline_x = interpolate.UnivariateSpline(
                self.s, self.points[:, 0], s=smoothing, k=3
            )
            self.spline_y = interpolate.UnivariateSpline(
                self.s, self.points[:, 1], s=smoothing, k=3
            )
    
    def _compute_arc_length_parameter(self) -> np.ndarray:
        """
        Compute cumulative arc length parameter for the points.
        
        Returns:
            Array of arc length values starting from 0
        """
        if self.closed:
            # Include distance from last point to first for closed curves
            diffs = np.diff(self.points, axis=0, prepend=self.points[-1:])
        else:
            diffs = np.diff(self.points, axis=0, prepend=self.points[0:1])
        
        distances = np.linalg.norm(diffs, axis=1)
        distances[0] = 0  # First point is at s=0
        
        s = np.cumsum(distances)
        return s
    
    def evaluate(self, s: np.ndarray) -> np.ndarray:
        """
        Evaluate spline at given parameter values.
        
        Args:
            s: Parameter values at which to evaluate
            
        Returns:
            Nx2 array of (x, y) coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        return np.column_stack([x, y])
    
    def compute_curvature(self, s: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute curvature at given parameter values.
        
        Curvature κ is computed using the formula:
        κ = (x' * y'' - y' * x'') / (x'² + y'²)^(3/2)
        
        Args:
            s: Parameter values at which to compute curvature.
               If None, uses the original points' parameters.
               
        Returns:
            Array of curvature values (1/radius)
        """
        if s is None:
            s = self.s
        
        # Compute first derivatives
        dx_ds = self.spline_x.derivative(n=1)(s)
        dy_ds = self.spline_y.derivative(n=1)(s)
        
        # Compute second derivatives
        d2x_ds2 = self.spline_x.derivative(n=2)(s)
        d2y_ds2 = self.spline_y.derivative(n=2)(s)
        
        # Compute curvature with numerical stability
        numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
        denominator = (dx_ds**2 + dy_ds**2)**1.5
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)
        
        curvature = numerator / denominator
        
        return curvature
    
    def get_total_length(self) -> float:
        """
        Get the total arc length of the spline.
        
        Returns:
            Total length
        """
        return self.s[-1]


def create_simple_oval_track(length: float = 1000.0, width_track: float = 200.0,
                            track_width: float = 12.0, n_points: int = 100) -> Track:
    """
    Create a simple oval track for testing and demonstration.
    
    Args:
        length: Approximate length of the straight sections
        width_track: Width of the oval (diameter of the curved sections)
        track_width: Width of the racing surface
        n_points: Number of points to discretize the centerline
        
    Returns:
        Track object
    """
    # Create oval shape: two straights connected by semicircles
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # Simple oval: rectangle with semicircular ends
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    radius = width_track / 2
    
    for i, t in enumerate(theta):
        if t < np.pi / 2:
            # Right semicircle
            angle = t * 2
            x[i] = length / 2 + radius * np.cos(angle)
            y[i] = radius * np.sin(angle)
        elif t < np.pi:
            # Top straight
            progress = (t - np.pi / 2) / (np.pi / 2)
            x[i] = length / 2 - progress * length
            y[i] = radius
        elif t < 3 * np.pi / 2:
            # Left semicircle
            angle = np.pi + (t - np.pi) * 2
            x[i] = -length / 2 + radius * np.cos(angle)
            y[i] = radius * np.sin(angle)
        else:
            # Bottom straight
            progress = (t - 3 * np.pi / 2) / (np.pi / 2)
            x[i] = -length / 2 + progress * length
            y[i] = -radius
    
    centerline = np.column_stack([x, y])
    
    return Track(centerline, track_width, closed=True)
