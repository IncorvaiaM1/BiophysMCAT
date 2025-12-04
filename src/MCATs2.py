"""
Parameter Sweep Framework with JIT Optimization and Comprehensive Data Analysis
For Magnetically Controlled Active Turbulence

Features:
- Efficient parameter sweeps with multiprocessing
- JIT-compiled bottleneck functions
- Automatic data storage (HDF5 + JSON metadata)
- Rich analysis and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from numba import jit, prange
import multiprocessing as mp
from functools import partial
import pandas as pd
import seaborn as sns

# ============================================================================
# JIT-OPTIMIZED COMPUTATIONAL KERNELS
# ============================================================================

@jit(nopython=True, parallel=True)
def compute_alignment_torques_jit(positions, theta, L, R_align, N):
    """
    JIT-compiled alignment torque calculation.
    Major speedup for large N.
    """
    alignment_torque = np.zeros(N)
    
    for i in prange(N):
        count = 0
        for j in range(N):
            if i != j:
                # Periodic distance
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dx = dx - L * np.round(dx / L)
                dy = dy - L * np.round(dy / L)
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < R_align:
                    alignment_torque[i] += np.sin(theta[j] - theta[i])
                    count += 1
        
        if count > 0:
            alignment_torque[i] /= count
    
    return alignment_torque


@jit(nopython=True, parallel=True)
def compute_repulsion_forces_jit(positions, L, R_rep, F_rep, N):
    """
    JIT-compiled repulsion force calculation.
    """
    repulsion_force = np.zeros((N, 2))
    
    for i in prange(N):
        for j in range(N):
            if i != j:
                # Periodic distance
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dx = dx - L * np.round(dx / L)
                dy = dy - L * np.round(dy / L)
                dist = np.sqrt(dx*dx + dy*dy)
                
                if 0 < dist < R_rep:
                    force_mag = F_rep * (1 - dist / R_rep)
                    repulsion_force[i, 0] += force_mag * dx / dist
                    repulsion_force[i, 1] += force_mag * dy / dist
    
    return repulsion_force


@jit(nopython=True)
def compute_order_parameters_jit(theta):
    """Fast order parameter calculation."""
    N = len(theta)
    
    # Nematic order
    cos_2theta = np.cos(2 * theta)
    sin_2theta = np.sin(2 * theta)
    nematic_order = np.sqrt((np.mean(cos_2theta))**2 + (np.mean(sin_2theta))**2)
    
    # Polar order
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    polar_order = np.sqrt((np.mean(cos_theta))**2 + (np.mean(sin_theta))**2)
    
    return nematic_order, polar_order


# ============================================================================
# OPTIMIZED SIMULATION CLASS
# ============================================================================

class OptimizedActiveTurbulence:
    """
    Optimized version with JIT-compiled bottlenecks.
    """
    
    def __init__(self, N=500, L=200.0, dt=0.01, seed=42):
        np.random.seed(seed)
        
        self.N = N
        self.L = L
        self.dt = dt
        
        # Physical parameters
        self.v0 = 15.0
        self.D_r = 0.5
        self.D_t = 0.1
        
        # Interaction parameters
        self.R_align = 10.0
        self.omega_align = 0.5
        self.R_rep = 3.0
        self.F_rep = 50.0
        
        # Magnetic parameters
        self.B = 0.0
        self.theta_B = 0.0
        self.kappa = 0.0
        
        # State
        self.positions = np.random.uniform(0, L, (N, 2))
        self.theta = np.random.uniform(-np.pi, np.pi, N)
        self.time = 0.0
        
        # Metrics storage
        self.metrics = {
            'time': [],
            'nematic_order': [],
            'polar_order': [],
            'velocity_variance': [],
            'angular_momentum': []
        }
    
    def set_magnetic_field(self, B, theta_B=0.0):
        self.B = B
        self.theta_B = theta_B
        self.kappa = 0.01 * B**2
    
    def step(self):
        """Single time step with JIT-optimized interactions."""
        # Use JIT functions for bottlenecks
        alignment_torque = compute_alignment_torques_jit(
            self.positions, self.theta, self.L, self.R_align, self.N
        )
        repulsion_force = compute_repulsion_forces_jit(
            self.positions, self.L, self.R_rep, self.F_rep, self.N
        )
        
        # Orientation update
        magnetic_torque = -self.kappa * np.sin(self.theta - self.theta_B)
        noise_rot = np.sqrt(2 * self.D_r / self.dt) * np.random.randn(self.N)
        
        d_theta = (magnetic_torque + 
                   self.omega_align * alignment_torque + 
                   noise_rot) * self.dt
        
        self.theta += d_theta
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # Position update
        p_hat = np.column_stack([np.cos(self.theta), np.sin(self.theta)])
        noise_trans = np.sqrt(2 * self.D_t / self.dt) * np.random.randn(self.N, 2)
        
        d_pos = (self.v0 * p_hat + repulsion_force + noise_trans) * self.dt
        self.positions += d_pos
        self.positions = self.positions % self.L
        
        self.time += self.dt
    
    def compute_metrics(self):
        """Compute comprehensive metrics."""
        nematic_S, polar_S = compute_order_parameters_jit(self.theta)
        
        # Velocity field variance
        vx = self.v0 * np.cos(self.theta)
        vy = self.v0 * np.sin(self.theta)
        vel_var = np.var(vx) + np.var(vy)
        
        # Angular momentum (measure of rotation)
        r_cm = np.mean(self.positions, axis=0)
        dr = self.positions - r_cm
        v = np.column_stack([vx, vy])
        L_z = np.mean(dr[:, 0] * v[:, 1] - dr[:, 1] * v[:, 0])
        
        return {
            'nematic_order': nematic_S,
            'polar_order': polar_S,
            'velocity_variance': vel_var,
            'angular_momentum': np.abs(L_z)
        }
    
    def run(self, T, save_interval=0.5):
        """Run with periodic metric recording."""
        n_steps = int(T / self.dt)
        save_every = int(save_interval / self.dt)
        
        for step in range(n_steps):
            self.step()
            
            if step % save_every == 0:
                metrics = self.compute_metrics()
                self.metrics['time'].append(self.time)
                for key, val in metrics.items():
                    self.metrics[key].append(val)
        
        return self.get_final_metrics()
    
    def get_final_metrics(self):
        """Return final state metrics averaged over last 20%."""
        n_samples = len(self.metrics['time'])
        n_avg = max(1, n_samples // 5)
        
        return {
            'final_nematic_order': np.mean(self.metrics['nematic_order'][-n_avg:]),
            'final_polar_order': np.mean(self.metrics['polar_order'][-n_avg:]),
            'final_velocity_variance': np.mean(self.metrics['velocity_variance'][-n_avg:]),
            'final_angular_momentum': np.mean(self.metrics['angular_momentum'][-n_avg:]),
            'std_nematic_order': np.std(self.metrics['nematic_order'][-n_avg:]),
            'time_series': self.metrics
        }


# ============================================================================
# PARAMETER SWEEP FRAMEWORK
# ============================================================================

class ParameterSweep:
    """
    Manages parameter sweeps with data storage and analysis.
    """
    
    def __init__(self, output_dir='sweep_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate unique run ID
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"Initialized sweep: {self.run_id}")
        print(f"Output directory: {self.run_dir}")
    
    def run_single_simulation(self, params):
        """Run a single simulation with given parameters."""
        # Unpack parameters
        kappa, N, omega_align, R_align, seed = params
        
        # Create simulation
        sim = OptimizedActiveTurbulence(N=int(N), L=200.0, dt=0.01, seed=seed)
        
        # Set parameters
        # Convert kappa back to B: B = sqrt(kappa / 0.01)
        B = np.sqrt(kappa / 0.01)
        sim.set_magnetic_field(B)
        sim.omega_align = omega_align
        sim.R_align = R_align
        
        # Run simulation
        results = sim.run(T=20.0, save_interval=0.5)
        
        # Add parameters to results
        results['params'] = {
            'kappa': kappa,
            'B': B,
            'N': N,
            'omega_align': omega_align,
            'R_align': R_align,
            'seed': seed
        }
        
        return results
    
    def run_sweep(self, param_grid, n_cores=None):
        """
        Run parameter sweep in parallel.
        
        param_grid: dict with parameter names and value lists
            Example: {
                'kappa': np.linspace(0, 10, 20),
                'N': [250, 500, 1000],
                'omega_align': [0.5],
                'R_align': [10.0]
            }
        """
        # Create all parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        combinations = list(product(*param_values))
        n_sims = len(combinations)
        
        print(f"\nRunning {n_sims} simulations...")
        print(f"Parameter grid:")
        for name, vals in param_grid.items():
            print(f"  {name}: {len(vals)} values")
        
        # Add random seeds for each combination
        params_list = []
        for i, combo in enumerate(combinations):
            param_dict = dict(zip(param_names, combo))
            # Ensure required parameters exist
            params = (
                param_dict.get('kappa', 0),
                param_dict.get('N', 500),
                param_dict.get('omega_align', 0.5),
                param_dict.get('R_align', 10.0),
                42 + i  # unique seed
            )
            params_list.append(params)
        
        # Run in parallel
        if n_cores is None:
            n_cores = mp.cpu_count() - 1
        
        print(f"Using {n_cores} cores")
        
        with mp.Pool(n_cores) as pool:
            results = pool.map(self.run_single_simulation, params_list)
        
        print("Simulations complete! Processing results...")
        
        # Store results
        self.results = results
        self.save_results()
        
        return results
    
    def save_results(self):
        """Save results to HDF5 and JSON."""
        # Save to HDF5 (efficient for large arrays)
        h5_file = self.run_dir / 'results.h5'
        
        with h5py.File(h5_file, 'w') as f:
            for i, result in enumerate(self.results):
                grp = f.create_group(f'sim_{i:04d}')
                
                # Save parameters
                for key, val in result['params'].items():
                    grp.attrs[key] = val
                
                # Save final metrics
                for key in ['final_nematic_order', 'final_polar_order', 
                           'final_velocity_variance', 'final_angular_momentum',
                           'std_nematic_order']:
                    grp.attrs[key] = result[key]
                
                # Save time series
                ts = result['time_series']
                for key, vals in ts.items():
                    grp.create_dataset(key, data=np.array(vals))
        
        print(f"Saved HDF5: {h5_file}")
        
        # Save summary as JSON (human-readable)
        summary = []
        for result in self.results:
            summary.append({
                'params': result['params'],
                'final_nematic_order': float(result['final_nematic_order']),
                'final_polar_order': float(result['final_polar_order']),
                'final_velocity_variance': float(result['final_velocity_variance']),
                'std_nematic_order': float(result['std_nematic_order'])
            })
        
        json_file = self.run_dir / 'summary.json'
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved JSON: {json_file}")
    
    def load_results(self, run_id=None):
        """Load results from a previous run."""
        if run_id is None:
            run_id = self.run_id
        
        h5_file = self.output_dir / run_id / 'results.h5'
        
        results = []
        with h5py.File(h5_file, 'r') as f:
            for sim_name in f.keys():
                grp = f[sim_name]
                
                result = {
                    'params': dict(grp.attrs),
                    'time_series': {}
                }
                
                # Load scalars
                for key in ['final_nematic_order', 'final_polar_order',
                           'final_velocity_variance', 'final_angular_momentum',
                           'std_nematic_order']:
                    result[key] = grp.attrs[key]
                
                # Load time series
                for key in grp.keys():
                    result['time_series'][key] = grp[key][:]
                
                results.append(result)
        
        self.results = results
        return results
    
    def create_dataframe(self):
        """Convert results to pandas DataFrame for easy analysis."""
        data = []
        for result in self.results:
            row = result['params'].copy()
            row['final_nematic_order'] = result['final_nematic_order']
            row['final_polar_order'] = result['final_polar_order']
            row['final_velocity_variance'] = result['final_velocity_variance']
            row['final_angular_momentum'] = result['final_angular_momentum']
            row['std_nematic_order'] = result['std_nematic_order']
            data.append(row)
        
        return pd.DataFrame(data)
    
    def analyze_and_plot(self):
        """Generate comprehensive analysis plots."""
        df = self.create_dataframe()
        
        # Save DataFrame as CSV
        csv_file = self.run_dir / 'results.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved CSV: {csv_file}")
        
        # Create figures
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Phase diagram: kappa vs final nematic order
        ax1 = plt.subplot(2, 3, 1)
        self._plot_phase_diagram(df, ax1)
        
        # 2. Time series for different kappa values
        ax2 = plt.subplot(2, 3, 2)
        self._plot_time_series_comparison(ax2)
        
        # 3. Velocity variance vs kappa
        ax3 = plt.subplot(2, 3, 3)
        self._plot_velocity_variance(df, ax3)
        
        # 4. N dependence (if varied)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_N_dependence(df, ax4)
        
        # 5. Order parameter distribution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_order_distribution(df, ax5)
        
        # 6. Critical transition analysis
        ax6 = plt.subplot(2, 3, 6)
        self._plot_critical_transition(df, ax6)
        
        plt.tight_layout()
        fig_file = self.run_dir / 'analysis.png'
        plt.savefig(fig_file, dpi=150)
        print(f"Saved figure: {fig_file}")
        
        return fig
    
    def _plot_phase_diagram(self, df, ax):
        """Plot kappa vs nematic order."""
        if 'kappa' in df.columns:
            grouped = df.groupby('kappa').agg({
                'final_nematic_order': ['mean', 'std']
            }).reset_index()
            
            kappa = grouped['kappa']
            mean_S = grouped['final_nematic_order']['mean']
            std_S = grouped['final_nematic_order']['std']
            
            ax.errorbar(kappa, mean_S, yerr=std_S, fmt='o-', 
                       linewidth=2, markersize=8, capsize=5)
            ax.set_xlabel('κ (1/s)', fontsize=12)
            ax.set_ylabel('Nematic Order S', fontsize=12)
            ax.set_title('Phase Diagram', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='S=0.5')
            ax.legend()
    
    def _plot_time_series_comparison(self, ax):
        """Plot time series for selected kappa values."""
        # Select 4-5 evenly spaced simulations
        n_show = min(5, len(self.results))
        indices = np.linspace(0, len(self.results)-1, n_show, dtype=int)
        
        for idx in indices:
            result = self.results[idx]
            ts = result['time_series']
            kappa = result['params']['kappa']
            ax.plot(ts['time'], ts['nematic_order'], 
                   label=f"κ={kappa:.2f}", linewidth=2)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Nematic Order S', fontsize=12)
        ax.set_title('Order Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_velocity_variance(self, df, ax):
        """Plot velocity variance vs kappa."""
        if 'kappa' in df.columns:
            ax.scatter(df['kappa'], df['final_velocity_variance'], 
                      alpha=0.6, s=50)
            ax.set_xlabel('κ (1/s)', fontsize=12)
            ax.set_ylabel('Velocity Variance', fontsize=12)
            ax.set_title('Turbulent Intensity', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_N_dependence(self, df, ax):
        """Plot effect of particle number."""
        if 'N' in df.columns and df['N'].nunique() > 1:
            for N in df['N'].unique():
                subset = df[df['N'] == N]
                ax.plot(subset['kappa'], subset['final_nematic_order'], 
                       'o-', label=f'N={int(N)}', linewidth=2)
            ax.set_xlabel('κ (1/s)', fontsize=12)
            ax.set_ylabel('Nematic Order S', fontsize=12)
            ax.set_title('System Size Effect', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'N not varied', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
    
    def _plot_order_distribution(self, df, ax):
        """Distribution of final order parameters."""
        ax.hist(df['final_nematic_order'], bins=20, alpha=0.7, 
               label='Nematic', edgecolor='black')
        ax.hist(df['final_polar_order'], bins=20, alpha=0.7,
               label='Polar', edgecolor='black')
        ax.set_xlabel('Order Parameter', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Order Distribution', fontsize=14, fontweight='bold')
        ax.legend()
    
    def _plot_critical_transition(self, df, ax):
        """Analyze critical transition point."""
        if 'kappa' in df.columns:
            # Find transition by looking at steepest change
            grouped = df.groupby('kappa')['final_nematic_order'].mean()
            kappa_vals = grouped.index.values
            S_vals = grouped.values
            
            # Compute derivative
            if len(kappa_vals) > 3:
                dS_dk = np.gradient(S_vals, kappa_vals)
                
                ax.plot(kappa_vals, dS_dk, 'b-', linewidth=2)
                ax.axhline(0, color='k', linestyle='--', alpha=0.3)
                
                # Mark maximum
                max_idx = np.argmax(np.abs(dS_dk))
                ax.plot(kappa_vals[max_idx], dS_dk[max_idx], 'ro', 
                       markersize=10, label=f'κ_c ≈ {kappa_vals[max_idx]:.2f}')
                
                ax.set_xlabel('κ (1/s)', fontsize=12)
                ax.set_ylabel('dS/dκ', fontsize=12)
                ax.set_title('Critical Transition', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_kappa_sweep():
    """
    Example: Sweep kappa (magnetic alignment strength).
    """
    print("\n" + "="*60)
    print("EXAMPLE: KAPPA PARAMETER SWEEP")
    print("="*60)
    
    sweep = ParameterSweep(output_dir='kappa_sweep')
    
    # Define parameter grid
    param_grid = {
        'kappa': np.linspace(0, 10, 15),  # 15 kappa values
        'N': [500],                        # Fixed N
        'omega_align': [0.5],              # Fixed alignment
        'R_align': [10.0]                  # Fixed range
    }
    
    # Run sweep
    results = sweep.run_sweep(param_grid, n_cores=4)
    
    # Analyze
    sweep.analyze_and_plot()
    
    # Print summary
    df = sweep.create_dataframe()
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(df[['kappa', 'final_nematic_order', 'final_velocity_variance']].to_string())
    
    return sweep


def example_multi_parameter_sweep():
    """
    Example: Sweep multiple parameters simultaneously.
    """
    print("\n" + "="*60)
    print("EXAMPLE: MULTI-PARAMETER SWEEP")
    print("="*60)
    
    sweep = ParameterSweep(output_dir='multi_param_sweep')
    
    # Define parameter grid
    param_grid = {
        'kappa': np.linspace(0, 8, 10),    # 10 kappa values
        'N': [250, 500, 1000],             # 3 system sizes
        'omega_align': [0.3, 0.5, 0.8],    # 3 alignment strengths
        'R_align': [10.0]                  # Fixed range
    }
    
    # This creates 10 x 3 x 3 = 90 simulations
    
    results = sweep.run_sweep(param_grid, n_cores=6)
    sweep.analyze_and_plot()
    
    return sweep


if __name__ == "__main__":
    # Run kappa sweep
    sweep1 = example_kappa_sweep()
    
    # Uncomment for multi-parameter sweep (takes longer!)
    # sweep2 = example_multi_parameter_sweep()
    
    plt.show()