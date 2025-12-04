"""
Langevin Dynamics Simulation of Magnetically Controlled Active Turbulence
Based on Beppu & Timonen (2024) Communications Physics

Author: Michael A. Incorvaia
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow
import matplotlib.cm as cm
from scipy.spatial import cKDTree




class ActiveTurbulenceSimulation:
    """
    Simulates magnetically controlled active particles using Langevin dynamics.
    
    Position dynamics: dr/dt = v0 * p_hat + interactions + noise
    Orientation dynamics: dθ/dt = -κ*sin(θ - θ_B) + alignment + rotational_noise
    """
    
    def __init__(self, N=2500, L=200.0, dt=0.01, seed=42):
        """
        Initialize simulation parameters.
        
        Parameters:
        -----------
        N : int
            Number of particles
        L : float
            Box size (μm)
        dt : float
            Time step (seconds)
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # System parameters
        self.N = N
        self.L = L
        self.dt = dt
        
        # Physical parameters (based on B. subtilis)
        self.v0 = 15.0              # Self-propulsion speed (μm/s)
        self.D_r = 0.5              # Rotational diffusion (rad²/s)
        self.D_t = 0.1              # Translational diffusion (μm²/s)
        
        # Interaction parameters
        self.R_align = 10.0         # Alignment interaction range (μm)
        self.omega_align = 0.5      # Alignment strength (1/s)
        self.R_rep = 3.0            # Repulsion range (μm, ~ body length)
        self.F_rep = 50.0           # Repulsion strength (μm/s²)
        
        # Magnetic field parameters
        self.B = 0.0                # Magnetic field strength (mT)
        self.theta_B = 0.0          # Magnetic field direction (radians)
        self.kappa = 0.0            # Magnetic alignment strength (1/s)
        
        # Initialize positions and orientations
        self.positions = np.random.uniform(0, L, (N, 2))
        self.theta = np.random.uniform(-np.pi, np.pi, N)
        
        # For tracking
        self.time = 0.0
        self.trajectory_history = []
        self.order_parameter_history = []
        
    def set_magnetic_field(self, B, theta_B=0.0):
        """
        Set magnetic field strength and direction.
        
        Parameters:
        -----------
        B : float
            Magnetic field strength (mT)
        theta_B : float
            Field direction in radians (0 = x-axis)
        """
        self.B = B
        self.theta_B = theta_B
        # Magnetic alignment strength scales as B² (from paper)
        # kappa ~ 2*Gamma*epsilon_1*B^2
        # Calibrated so kappa ~ 5-10 rad/s at B=30mT
        self.kappa = 0.01 * B**2
        
    def calculate_nematic_order(self):
        """
        Calculate nematic order parameter: S = <cos(2θ)>
        S = 0: isotropic (turbulent)
        S = 1: perfectly aligned
        """
        return np.mean(np.cos(2 * self.theta))
    
    def calculate_polar_order(self):
        """
        Calculate polar order parameter for comparison.
        """
        vx = np.cos(self.theta)
        vy = np.sin(self.theta)
        return np.sqrt(np.mean(vx)**2 + np.mean(vy)**2)
    
    def get_periodic_distance(self, r1, r2):
        """
        Calculate distance with periodic boundary conditions.
        """
        dr = r1 - r2
        dr = dr - self.L * np.round(dr / self.L)
        return dr
    
    def calculate_interactions(self):
        """
        Calculate alignment and repulsion forces using neighbor search.
        Returns alignment torques and repulsion forces.
        """
        # Build KD-tree for efficient neighbor search
        # Need to account for periodic boundaries
        tree = cKDTree(self.positions, boxsize=self.L)
        
        alignment_torque = np.zeros(self.N)
        repulsion_force = np.zeros((self.N, 2))
        
        # Query neighbors within alignment range
        neighbors = tree.query_ball_point(self.positions, self.R_align)
        
        for i in range(self.N):
            # Alignment interaction (nematic, so use sin difference)
            for j in neighbors[i]:
                if i != j:
                    alignment_torque[i] += np.sin(self.theta[j] - self.theta[i])
            
            # Normalize by number of neighbors
            if len(neighbors[i]) > 1:
                alignment_torque[i] /= (len(neighbors[i]) - 1)
        
        # Repulsion (short-range, softer than alignment)
        rep_neighbors = tree.query_ball_point(self.positions, self.R_rep)
        
        for i in range(self.N):
            for j in rep_neighbors[i]:
                if i != j:
                    dr = self.get_periodic_distance(self.positions[i], 
                                                   self.positions[j])
                    dist = np.linalg.norm(dr)
                    if dist > 0 and dist < self.R_rep:
                        # Soft-core repulsion: F ~ (1 - r/R_rep)
                        force_mag = self.F_rep * (1 - dist / self.R_rep)
                        repulsion_force[i] += force_mag * dr / dist
        
        return alignment_torque, repulsion_force
    
    def step(self):
        """
        Perform one time step using Euler-Maruyama integration.
        """
        # Calculate interactions
        alignment_torque, repulsion_force = self.calculate_interactions()
        
        # Orientation dynamics
        # dθ/dt = -κ*sin(θ - θ_B) + ω*alignment + sqrt(2*D_r)*ξ
        magnetic_torque = -self.kappa * np.sin(self.theta - self.theta_B)
        
        noise_rot = np.sqrt(2 * self.D_r / self.dt) * np.random.randn(self.N)
        
        d_theta = (magnetic_torque + 
                   self.omega_align * alignment_torque + 
                   noise_rot) * self.dt
        
        self.theta += d_theta
        # Keep angles in [-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # Position dynamics
        # dr/dt = v0*p_hat + μ*F_rep + sqrt(2*D_t)*η
        p_hat = np.column_stack([np.cos(self.theta), np.sin(self.theta)])
        
        noise_trans = np.sqrt(2 * self.D_t / self.dt) * np.random.randn(self.N, 2)
        
        d_pos = (self.v0 * p_hat + 
                 repulsion_force + 
                 noise_trans) * self.dt
        
        self.positions += d_pos
        
        # Apply periodic boundary conditions
        self.positions = self.positions % self.L
        
        # Update time
        self.time += self.dt
    
    def run(self, T, save_interval=1.0):
        """
        Run simulation for time T.
        
        Parameters:
        -----------
        T : float
            Total simulation time (seconds)
        save_interval : float
            Time interval for saving snapshots (seconds)
        """
        n_steps = int(T / self.dt)
        save_every = int(save_interval / self.dt)
        
        print(f"Running simulation for {T}s ({n_steps} steps)...")
        print(f"Magnetic field: B = {self.B} mT")
        
        for step in range(n_steps):
            self.step()
            
            # Save trajectory snapshots
            if step % save_every == 0:
                self.trajectory_history.append({
                    'time': self.time,
                    'positions': self.positions.copy(),
                    'theta': self.theta.copy()
                })
                
                # Calculate and save order parameter
                S = self.calculate_nematic_order()
                self.order_parameter_history.append(S)
                
                if step % (save_every * 10) == 0:
                    print(f"  t = {self.time:.1f}s, S = {S:.3f}")
        
        print("Simulation complete!")
    
    def plot_snapshot(self, ax=None, show_velocities=True, 
                     arrow_scale=2.0, title=None):
        """
        Plot current state of the system.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot particles colored by orientation
        colors = cm.hsv((self.theta + np.pi) / (2 * np.pi))
        ax.scatter(self.positions[:, 0], self.positions[:, 1], 
                  c=colors, s=30, alpha=0.8)
        
        # Plot velocity vectors
        if show_velocities:
            u = arrow_scale * np.cos(self.theta)
            v = arrow_scale * np.sin(self.theta)
            ax.quiver(self.positions[:, 0], self.positions[:, 1],
                     u, v, color=colors, alpha=0.6, width=0.003)
        
        # Show magnetic field direction if present
        if self.B > 0:
            ax.arrow(self.L * 0.1, self.L * 0.9, 
                    15 * np.cos(self.theta_B), 15 * np.sin(self.theta_B),
                    head_width=3, head_length=4, fc='red', ec='red', 
                    linewidth=2, label=f'B = {self.B} mT')
            ax.legend(loc='upper right')
        
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_xlabel('x (μm)', fontsize=12)
        ax.set_ylabel('y (μm)', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            S = self.calculate_nematic_order()
            ax.set_title(f'Active Turbulence (t = {self.time:.1f}s, S = {S:.3f})', 
                        fontsize=14)
        
        return ax
    
    def plot_order_parameter(self, ax=None):
        """
        Plot time evolution of nematic order parameter.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        times = [snap['time'] for snap in self.trajectory_history]
        ax.plot(times, self.order_parameter_history, 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Nematic Order Parameter S', fontsize=12)
        ax.set_title(f'Order Parameter Evolution (B = {self.B} mT)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    # def save_vmd_trajectory(self, filename='trajectory.xyz'):
    #     """
    #     Save trajectory in XYZ format for VMD visualization.
    #     """
    #     with open(filename, 'w') as f:
    #         for snap in self.trajectory_history:
    #             N = len(snap['positions'])
    #             f.write(f"{N}\n")
    #             f.write(f"Time = {snap['time']:.3f} s\n")
                
    #             for i in range(N):
    #                 x, y = snap['positions'][i]
    #                 theta = snap['theta'][i]
    #                 # Use theta as a property (can be colored in VMD)
    #                 f.write(f"C {x:.3f} {y:.3f} 0.000 {theta:.3f}\n")
        
    #     print(f"Saved trajectory to {filename}")


def run_baseline_turbulence():
    """
    Example 1: Generate baseline turbulent state (B = 0).
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Baseline Turbulence (No Magnetic Field)")
    print("="*60)
    
    sim = ActiveTurbulenceSimulation(N=2500, L=200.0, dt=0.01)
    sim.set_magnetic_field(B=0.0)
    sim.run(T=20.0, save_interval=0.5)
    
    # Plot final state
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sim.plot_snapshot(ax=ax1, title='Baseline Turbulence (B = 0 mT)')
    sim.plot_order_parameter(ax=ax2)
    plt.tight_layout()
    plt.savefig('turbulence_baseline.png', dpi=150)
    print("Saved: turbulence_baseline.png")
    
    return sim


def run_magnetic_suppression():
    """
    Example 2: Strong magnetic field suppresses turbulence.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Magnetic Suppression (B = 25 mT)")
    print("="*60)
    
    sim = ActiveTurbulenceSimulation(N=2500, L=200.0, dt=0.01)
    sim.set_magnetic_field(B=25.0, theta_B=0.0)
    sim.run(T=20.0, save_interval=0.5)
    
    # Plot final state
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sim.plot_snapshot(ax=ax1, title='Aligned State (B = 25 mT)')
    sim.plot_order_parameter(ax=ax2)
    plt.tight_layout()
    plt.savefig('turbulence_suppressed.png', dpi=150)
    print("Saved: turbulence_suppressed.png")
    
    return sim


def run_field_sweep():
    """
    Example 3: Sweep magnetic field strength to create phase diagram.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Magnetic Field Sweep")
    print("="*60)
    
    B_values = [0, 5, 10, 15, 20, 25, 30]
    final_S_values = []
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, B in enumerate(B_values):
        print(f"\nRunning B = {B} mT...")
        sim = ActiveTurbulenceSimulation(N=2500, L=200.0, dt=0.01, seed=42+idx)
        sim.set_magnetic_field(B=B)
        sim.run(T=15.0, save_interval=1.0)
        
        # Plot snapshot
        sim.plot_snapshot(ax=axes[idx], show_velocities=True,
                         title=f'B = {B} mT')
        
        # Record final order parameter
        final_S = sim.calculate_nematic_order()
        final_S_values.append(final_S)
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('field_sweep_snapshots.png', dpi=150)
    print("\nSaved: field_sweep_snapshots.png")
    
    # Plot phase diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(B_values, final_S_values, 'bo-', linewidth=2, markersize=10)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(1, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Magnetic Field Strength B (mT)', fontsize=14)
    ax.set_ylabel('Nematic Order Parameter S', fontsize=14)
    ax.set_title('Phase Diagram: Turbulence → Nematic Transition', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('phase_diagram.png', dpi=150)
    print("Saved: phase_diagram.png")
    
    return B_values, final_S_values


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MAGNETICALLY CONTROLLED ACTIVE TURBULENCE SIMULATION")
    print("="*60)
    
    # Run examples
    sim1 = run_baseline_turbulence()
    sim2 = run_magnetic_suppression()
    B_vals, S_vals = run_field_sweep()
    
    # # Save VMD trajectory for one case
    # print("\nSaving VMD trajectory...")
    # sim2.save_vmd_trajectory('aligned_state.xyz')
    
    print("\n" + "="*60)
    print("ALL SIMULATIONS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - turbulence_baseline.png")
    print("  - turbulence_suppressed.png")
    print("  - field_sweep_snapshots.png")
    print("  - phase_diagram.png")
    print("  - aligned_state.xyz (for VMD)")
    
    plt.show()