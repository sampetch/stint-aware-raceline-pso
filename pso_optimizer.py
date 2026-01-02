"""
Particle Swarm Optimization (PSO) module for racing line optimization.

This module implements PSO to find the optimal alpha values that minimize lap time.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
import time


class Particle:
    """
    Represents a single particle in the PSO swarm.
    
    Each particle represents a potential racing line (alpha values).
    """
    
    def __init__(self, dimension: int, bounds: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize a particle with random position and velocity.
        
        Args:
            dimension: Number of dimensions (number of alpha values)
            bounds: Tuple of (min, max) values for position
        """
        self.dimension = dimension
        self.bounds = bounds
        
        # Initialize position randomly within bounds
        self.position = np.random.uniform(bounds[0], bounds[1], dimension)
        
        # Initialize velocity randomly (small)
        velocity_range = (bounds[1] - bounds[0]) * 0.1
        self.velocity = np.random.uniform(-velocity_range, velocity_range, dimension)
        
        # Best position found by this particle
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        
        # Current fitness
        self.fitness = float('inf')
    
    def update_velocity(self, global_best_position: np.ndarray,
                       w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        Update particle velocity using PSO equation.
        
        v_new = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        
        Args:
            global_best_position: Best position found by any particle
            w: Inertia weight
            c1: Cognitive (personal) learning factor
            c2: Social (global) learning factor
        """
        r1 = np.random.random(self.dimension)
        r2 = np.random.random(self.dimension)
        
        # PSO velocity update equation
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
        # Limit velocity to prevent explosion
        max_velocity = (self.bounds[1] - self.bounds[0]) * 0.2
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self):
        """
        Update particle position based on velocity.
        
        Applies boundary constraints.
        """
        self.position = self.position + self.velocity
        
        # Apply bounds
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
    
    def update_best(self):
        """
        Update personal best if current fitness is better.
        """
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()


class PSOOptimizer:
    """
    Particle Swarm Optimization for racing line optimization.
    """
    
    def __init__(self, objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 n_particles: int = 30,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Initialize PSO optimizer.
        
        Args:
            objective_function: Function to minimize, takes alpha array and returns fitness
            dimension: Number of dimensions (number of alpha values)
            n_particles: Number of particles in swarm
            bounds: Tuple of (min, max) values for variables
            w: Inertia weight
            c1: Cognitive learning factor
            c2: Social learning factor
        """
        self.objective_function = objective_function
        self.dimension = dimension
        self.n_particles = n_particles
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialize swarm
        self.particles = [Particle(dimension, bounds) for _ in range(n_particles)]
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # History for analysis
        self.fitness_history = []
        self.best_fitness_history = []
    
    def initialize_swarm(self, initial_guess: Optional[np.ndarray] = None):
        """
        Initialize swarm with an optional initial guess.
        
        Args:
            initial_guess: Optional initial position for one particle
        """
        if initial_guess is not None:
            # Set first particle to initial guess
            self.particles[0].position = np.clip(initial_guess, 
                                                self.bounds[0], 
                                                self.bounds[1])
    
    def evaluate_fitness(self):
        """
        Evaluate fitness for all particles.
        """
        for particle in self.particles:
            particle.fitness = self.objective_function(particle.position)
            
            # Update particle's personal best
            particle.update_best()
            
            # Update global best
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
    
    def update_swarm(self):
        """
        Update velocities and positions for all particles.
        """
        for particle in self.particles:
            particle.update_velocity(self.global_best_position, 
                                    self.w, self.c1, self.c2)
            particle.update_position()
    
    def optimize(self, max_iterations: int = 100,
                tolerance: float = 1e-6,
                verbose: bool = True) -> Tuple[np.ndarray, float, Dict]:
        """
        Run PSO optimization.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (stop if improvement < tolerance)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_position, best_fitness, info_dict)
        """
        start_time = time.time()
        
        # Initial evaluation
        self.evaluate_fitness()
        self.fitness_history.append([p.fitness for p in self.particles])
        self.best_fitness_history.append(self.global_best_fitness)
        
        if verbose:
            print(f"Iteration 0: Best fitness = {self.global_best_fitness:.4f}")
        
        # Main optimization loop
        converged = False
        for iteration in range(1, max_iterations + 1):
            # Store previous best for convergence check
            prev_best = self.global_best_fitness
            
            # Update swarm
            self.update_swarm()
            
            # Evaluate fitness
            self.evaluate_fitness()
            
            # Record history
            self.fitness_history.append([p.fitness for p in self.particles])
            self.best_fitness_history.append(self.global_best_fitness)
            
            # Check convergence
            improvement = prev_best - self.global_best_fitness
            if improvement < tolerance and iteration > 10:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Print progress
            if verbose and (iteration % 10 == 0 or iteration == max_iterations):
                elapsed = time.time() - start_time
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.4f} "
                      f"(improvement: {improvement:.6f}, elapsed: {elapsed:.2f}s)")
        
        total_time = time.time() - start_time
        
        # Prepare info dictionary
        info = {
            'iterations': iteration,
            'converged': converged,
            'total_time': total_time,
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'final_swarm_positions': [p.position.copy() for p in self.particles],
            'final_swarm_fitness': [p.fitness for p in self.particles]
        }
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final best fitness: {self.global_best_fitness:.4f}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Converged: {converged}")
        
        return self.global_best_position, self.global_best_fitness, info
    
    def optimize_with_restarts(self, n_restarts: int = 3,
                               max_iterations_per_restart: int = 50,
                               verbose: bool = True) -> Tuple[np.ndarray, float, Dict]:
        """
        Run PSO with multiple restarts to avoid local minima.
        
        Args:
            n_restarts: Number of restarts
            max_iterations_per_restart: Maximum iterations per restart
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_position, best_fitness, info_dict)
        """
        overall_best_position = None
        overall_best_fitness = float('inf')
        all_results = []
        
        for restart in range(n_restarts):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Restart {restart + 1}/{n_restarts}")
                print(f"{'='*60}")
            
            # Reinitialize swarm (except for first restart)
            if restart > 0:
                self.particles = [Particle(self.dimension, self.bounds) 
                                for _ in range(self.n_particles)]
                self.global_best_position = None
                self.global_best_fitness = float('inf')
            
            # Run optimization
            position, fitness, info = self.optimize(
                max_iterations=max_iterations_per_restart,
                verbose=verbose
            )
            
            # Store results
            all_results.append({
                'position': position,
                'fitness': fitness,
                'info': info
            })
            
            # Update overall best
            if fitness < overall_best_fitness:
                overall_best_fitness = fitness
                overall_best_position = position.copy()
        
        # Prepare combined info
        combined_info = {
            'n_restarts': n_restarts,
            'all_results': all_results,
            'best_restart': np.argmin([r['fitness'] for r in all_results])
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"All restarts complete!")
            print(f"Overall best fitness: {overall_best_fitness:.4f}")
            print(f"Best restart: {combined_info['best_restart'] + 1}")
            print(f"{'='*60}")
        
        return overall_best_position, overall_best_fitness, combined_info
