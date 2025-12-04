"""
Analysis: Nematic Order as Function of Density and Time
[JAX JIT-optimized version for faster computation]

Addresses your key questions:
1. How does nematic order S evolve with time for different densities?
2. What is S as a function of density at fixed B?
3. How does bacterial growth (increasing N over time) affect magnetic control?

Author: Michael A. Incorvaia
Date: November 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import jax
import jax.numpy as jnp
from functools import partial

from MCATs1 import ActiveTurbulenceSimulation
from parameters2 import (
    DENSITY_RANGES, MAGNETIC_FIELDS, GROWTH, APPLICATIONS,
    particle_count_to_area_fraction, calculate_population,
    EXPERIMENTAL_DENSITIES, TYPICAL_DENSITIES
)


# ============================================================================
# JAX JIT-compiled utility functions for performance-critical operations
# ============================================================================

@jax.jit
def jax_find_alignment_time(S_array, threshold=0.8):
    """
    JAX-compiled function to find when order exceeds threshold.
    
    Parameters:
    -----------
    S_array : jnp.ndarray
        Order parameter time series
    threshold : float
        Threshold value
        
    Returns:
    --------
    index : int
        Index where S first exceeds threshold (-1 if never)
    """
    # Create boolean mask for where S exceeds threshold
    exceeds = S_array > threshold
    # Find first True index, return -1 if none found
    indices = jnp.where(exceeds, size=1)[0]
    return jnp.where(indices.size > 0, indices[0], jnp.array(-1, dtype=jnp.int32))


@jax.jit
def jax_calculate_growth(N0, t_min, doubling_time):
    """
    JAX-compiled exponential growth calculation.
    
    Parameters:
    -----------
    N0 : float
        Initial population
    t_min : float
        Time in minutes
    doubling_time : float
        Doubling time in minutes
        
    Returns:
    --------
    N : float
        Population at time t
    """
    growth_rate = jnp.log(2.0) / doubling_time
    return N0 * jnp.exp(growth_rate * t_min)


@jax.jit
def jax_calculate_magnetic_field_ramped(t_min, total_time_min, B_max):
    """
    JAX-compiled ramped magnetic field protocol.
    
    Parameters:
    -----------
    t_min : float
        Current time
    total_time_min : float
        Total simulation time
    B_max : float
        Maximum field
        
    Returns:
    --------
    B : float
        Magnetic field at time t
    """
    return B_max * (t_min / total_time_min)


@jax.jit
def jax_calculate_magnetic_field_step(t_min, doubling_time, B_max):
    """
    JAX-compiled step magnetic field protocol.
    
    Parameters:
    -----------
    t_min : float
        Current time
    doubling_time : float
        Doubling time
    B_max : float
        Maximum field
        
    Returns:
    --------
    B : float
        Magnetic field at time t
    """
    generations = t_min / doubling_time
    B = 5.0 * generations  # 5 mT per generation
    return jnp.minimum(B, B_max)


@jax.jit
def jax_batch_normalize_vectors(vectors):
    """
    JAX-compiled batch vector normalization.
    Useful for normalizing order parameter calculations.
    
    Parameters:
    -----------
    vectors : jnp.ndarray
        Shape (N, 2) array of 2D vectors
        
    Returns:
    --------
    normalized : jnp.ndarray
        Normalized vectors
    """
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)


@jax.jit
def jax_calculate_nematic_order_from_angles(angles):
    """
    JAX-compiled nematic order parameter calculation from angles.
    S = < |cos(2(θ_i - θ_mean))| >
    
    Parameters:
    -----------
    angles : jnp.ndarray
        Array of particle angles
        
    Returns:
    --------
    S : float
        Nematic order parameter
    """
    # Calculate mean angle
    mean_angle = jnp.arctan2(
        jnp.mean(jnp.sin(2 * angles)),
        jnp.mean(jnp.cos(2 * angles))
    ) / 2.0
    
    # Calculate average alignment
    alignments = jnp.abs(jnp.cos(2 * (angles - mean_angle)))
    S = jnp.mean(alignments)
    
    return S


@partial(jax.jit, static_argnames=('n_points',))
def jax_interpolate_S_series(times_old, S_old, times_new, n_points=None):
    """
    JAX-compiled interpolation of order parameter time series.
    
    Parameters:
    -----------
    times_old : jnp.ndarray
        Original time points
    S_old : jnp.ndarray
        Original S values
    times_new : jnp.ndarray
        New time points for interpolation
    n_points : int
        Size hint for compilation
        
    Returns:
    --------
    S_new : jnp.ndarray
        Interpolated S values
    """
    # Simple linear interpolation
    S_new = jnp.interp(times_new, times_old, S_old)
    return S_new


@jax.jit
def jax_batch_cap_values(values, max_val):
    """
    JAX-compiled value capping.
    
    Parameters:
    -----------
    values : jnp.ndarray
        Array of values
    max_val : float
        Maximum value
        
    Returns:
    --------
    capped : jnp.ndarray
        Capped values
    """
    return jnp.minimum(values, max_val)


# ============================================================================
# Original analysis functions (unchanged logic, but using JAX where beneficial)
# ============================================================================

def measure_order_vs_time_multiple_densities(N_values, B, T=30.0, save_interval=0.5):
    """
    Measure S(t) for multiple densities at fixed magnetic field.
    
    Question: Does alignment happen faster at lower/higher densities?
    
    Parameters:
    -----------
    N_values : list
        List of particle counts to test
    B : float
        Magnetic field strength (mT)
    T : float
        Total simulation time (s)
        
    Returns:
    --------
    results : dict
        {'N': [...], 'times': [...], 'S_vs_t': [[...], ...], 'phi': [...]}
    """
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1: Nematic Order vs Time for Multiple Densities")
    print(f"Magnetic Field: B = {B} mT")
    print(f"{'='*70}\n")
    
    results = {
        'N': N_values,
        'times': None,
        'S_vs_t': [],
        'phi': [],
    }
    
    for N in N_values:
        phi = particle_count_to_area_fraction(N)
        print(f"Running N = {N} (φ = {phi:.3f})...")
        
        sim = ActiveTurbulenceSimulation(N=N, L=200.0, dt=0.01)
        sim.set_magnetic_field(B)
        sim.run(T=T, save_interval=save_interval)
        
        times = [snap['time'] for snap in sim.trajectory_history]
        S_values = sim.order_parameter_history
        
        if results['times'] is None:
            results['times'] = times
        
        results['S_vs_t'].append(S_values)
        results['phi'].append(phi)
        
        print(f"  Final S = {S_values[-1]:.3f}\n")
    
    return results


def measure_order_vs_density_at_fixed_B(N_values, B, T_equilibrate=20.0):
    """
    Measure final S as function of density N at fixed field B.
    
    Question: What's the phase boundary in (N, B) space?
    
    Parameters:
    -----------
    N_values : list
        Particle counts to test
    B : float
        Magnetic field strength (mT)
    T_equilibrate : float
        Time to reach steady state (s)
        
    Returns:
    --------
    results : pd.DataFrame
        Columns: N, phi, S_final, tau_align
    """
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2: Nematic Order vs Density at B = {B} mT")
    print(f"{'='*70}\n")
    
    data = []
    
    for N in N_values:
        phi = particle_count_to_area_fraction(N)
        print(f"N = {N:4d} (φ = {phi:.4f})...", end=' ')
        
        sim = ActiveTurbulenceSimulation(N=N, L=200.0, dt=0.01)
        sim.set_magnetic_field(B)
        sim.run(T=T_equilibrate, save_interval=0.5)
        
        S_final = sim.calculate_nematic_order()
        
        # Use JAX JIT-compiled function to find alignment time
        S_array_jax = jnp.array(sim.order_parameter_history)
        tau_idx = jax_find_alignment_time(S_array_jax, threshold=0.8)
        
        tau_align = None
        if tau_idx >= 0:
            tau_align = sim.trajectory_history[int(tau_idx)]['time']
        
        data.append({
            'N': N,
            'phi': phi,
            'S_final': S_final,
            'tau_align': tau_align if tau_align else np.nan,
        })
        
        print(f"S = {S_final:.3f}, τ = {tau_align if tau_align else 'N/A'}")
    
    return pd.DataFrame(data)


def simulate_bacterial_growth_with_magnetic_control(
    N0=100, 
    doubling_time=20, 
    total_time_min=60,
    B_protocol='ramped',
    B_max=30.0
):
    """
    Simulate bacterial population growth with increasing magnetic field.
    
    Scenario: Bacteria multiply (e.g., skin infection), but we ramp up
    magnetic field to maintain control.
    
    Question: Can magnetic field "keep up" with exponential growth?
    
    Parameters:
    -----------
    N0 : int
        Initial population
    doubling_time : float
        Bacterial doubling time (minutes)
    total_time_min : float
        Total time (minutes)
    B_protocol : str
        'ramped': B increases linearly with time
        'step': B increases in steps matching population doublings
        'constant': B held at B_max throughout
    B_max : float
        Maximum magnetic field (mT)
        
    Returns:
    --------
    results : dict
        {'time_min': [...], 'N': [...], 'B': [...], 'S': [...], 'phi': [...]}
    """
    print(f"\n{'='*70}")
    print(f"ANALYSIS 3: Bacterial Growth + Magnetic Control")
    print(f"Initial N = {N0}, Doubling time = {doubling_time} min")
    print(f"B protocol: {B_protocol}, B_max = {B_max} mT")
    print(f"{'='*70}\n")
    
    # Time points (every 5 minutes) - vectorize with JAX
    time_points_min = np.arange(0, total_time_min + 5, 5)
    
    # JAX-compile batch calculations for growth and field
    time_points_jax = jnp.array(time_points_min)
    
    results = {
        'time_min': [],
        'N': [],
        'B': [],
        'S': [],
        'phi': [],
    }
    
    for t_min in time_points_min:
        # Calculate population at this time using JAX
        N_jax = float(jax_calculate_growth(jnp.array(N0), jnp.array(t_min), 
                                           jnp.array(doubling_time)))
        N = int(min(N_jax, 5000))  # Cap at reasonable max
        
        # Determine magnetic field based on protocol using JAX
        if B_protocol == 'ramped':
            B = float(jax_calculate_magnetic_field_ramped(jnp.array(t_min), 
                                                         jnp.array(total_time_min),
                                                         jnp.array(B_max)))
        elif B_protocol == 'step':
            B = float(jax_calculate_magnetic_field_step(jnp.array(t_min), 
                                                       jnp.array(doubling_time),
                                                       jnp.array(B_max)))
        elif B_protocol == 'constant':
            B = B_max
        else:
            B = 0.0
        
        phi = particle_count_to_area_fraction(N)
        
        print(f"t = {t_min:3.0f} min: N = {N:4d} (φ={phi:.3f}), B = {B:5.1f} mT...", 
              end=' ')
        
        # Run short simulation to measure order parameter
        sim = ActiveTurbulenceSimulation(N=N, L=200.0, dt=0.01, seed=int(t_min))
        sim.set_magnetic_field(B)
        sim.run(T=10.0, save_interval=1.0)  # 10s equilibration
        
        S = sim.calculate_nematic_order()
        
        results['time_min'].append(t_min)
        results['N'].append(N)
        results['B'].append(B)
        results['S'].append(S)
        results['phi'].append(phi)
        
        print(f"S = {S:.3f}")
    
    return results


def plot_analysis_1(results, save_path='figures1/analysis/S_vs_time_multiple_N.png'):
    """Plot S(t) for multiple densities."""
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[2, 1])
    
    # Panel A: S vs time
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['N'])))
    
    for i, (N, S_vs_t, phi) in enumerate(zip(results['N'], 
                                              results['S_vs_t'], 
                                              results['phi'])):
        ax1.plot(results['times'], S_vs_t, '-', color=colors[i], 
                linewidth=2, label=f'N={N} (φ={phi:.3f})')
    
    ax1.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Nematic Order S', fontsize=12)
    ax1.set_title('Order Parameter Evolution: Effect of Density', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Final S vs density
    ax2 = fig.add_subplot(gs[1])
    final_S = [S_vals[-1] for S_vals in results['S_vs_t']]
    ax2.plot(results['phi'], final_S, 'o-', color='#2ecc71', 
            markersize=10, linewidth=2)
    ax2.axhline(0.8, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Area Fraction φ', fontsize=12)
    ax2.set_ylabel('Final Order S', fontsize=12)
    ax2.set_title('Equilibrium Order vs Density', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    
    return fig


def plot_analysis_2(df, B, save_path='figures1/analysis/S_vs_density.png'):
    """Plot S vs density as scatter + fit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: S vs phi
    ax1.plot(df['phi'], df['S_final'], 'o-', color='#3498db', 
            markersize=10, linewidth=2)
    ax1.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Order threshold')
    ax1.axhline(0.0, color='k', linestyle='--', alpha=0.3)
    
    # Overlay application windows
    for app_name, specs in APPLICATIONS.items():
        phi_min, phi_max = specs['phi_range']
        ax1.axvspan(phi_min, phi_max, alpha=0.1, color=specs['color'], 
                   label=app_name.replace('_', ' ').title())
    
    ax1.set_xlabel('Area Fraction φ', fontsize=12)
    ax1.set_ylabel('Nematic Order S', fontsize=12)
    ax1.set_title(f'Order Parameter vs Density (B = {B} mT)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Alignment time vs density
    valid_tau = df[df['tau_align'].notna()]
    if len(valid_tau) > 0:
        ax2.plot(valid_tau['phi'], valid_tau['tau_align'], 's-', 
                color='#e74c3c', markersize=8, linewidth=2)
        ax2.set_xlabel('Area Fraction φ', fontsize=12)
        ax2.set_ylabel('Alignment Time τ (s)', fontsize=12)
        ax2.set_title('Response Time vs Density', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No alignment observed\n(B too weak)', 
                ha='center', va='center', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def plot_analysis_3(results, save_path='figures1/analysis/growth_with_control.png'):
    """Plot bacterial growth with magnetic control."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time = results['time_min']
    
    # Panel A: Population growth
    ax = axes[0, 0]
    ax.semilogy(time, results['N'], 'o-', color='#2ecc71', 
               markersize=8, linewidth=2)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Population N', fontsize=12)
    ax.set_title('Bacterial Population Growth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel B: Magnetic field protocol
    ax = axes[0, 1]
    ax.plot(time, results['B'], 's-', color='#e74c3c', 
           markersize=8, linewidth=2)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Magnetic Field B (mT)', fontsize=12)
    ax.set_title('Control Protocol', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel C: Order parameter evolution
    ax = axes[1, 0]
    ax.plot(time, results['S'], 'o-', color='#3498db', 
           markersize=8, linewidth=2)
    ax.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Order threshold')
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Nematic Order S', fontsize=12)
    ax.set_title('Collective Order Under Growth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Panel D: Phase space trajectory
    ax = axes[1, 1]
    scatter = ax.scatter(results['phi'], results['B'], c=results['S'], 
                        s=100, cmap='RdYlGn', vmin=0, vmax=1, 
                        edgecolor='k', linewidth=1)
    
    # Add arrows showing time evolution
    for i in range(len(time)-1):
        ax.annotate('', xy=(results['phi'][i+1], results['B'][i+1]),
                   xytext=(results['phi'][i], results['B'][i]),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
    
    ax.set_xlabel('Area Fraction φ', fontsize=12)
    ax.set_ylabel('Magnetic Field B (mT)', fontsize=12)
    ax.set_title('Trajectory in (φ, B) Space', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Order S', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def main():
    """Run all analyses."""
    print("\n" + "="*70)
    print("DENSITY & TIME ANALYSIS SUITE [JAX JIT-OPTIMIZED]")
    print("="*70)
    
    # Analysis 1: S(t) for multiple densities at B = 20 mT
    results1 = measure_order_vs_time_multiple_densities(
        N_values=[100, 200, 500, 1000, 2000],
        B=20.0,
        T=30.0
    )
    plot_analysis_1(results1)
    
    # Analysis 2: S(density) at different field strengths
    for B in [10, 20, 30]:
        df = measure_order_vs_density_at_fixed_B(
            N_values=DENSITY_RANGES['N_phase_diagram'],
            B=B
        )
        df.to_csv(f'data1/S_vs_density_B{B}.csv', index=False)
        plot_analysis_2(df, B, save_path=f'figures1/analysis/S_vs_density_B{B}.png')
    
    # Analysis 3: Bacterial growth with magnetic control
    for protocol in ['ramped', 'step', 'constant']:
        results3 = simulate_bacterial_growth_with_magnetic_control(
            N0=100,
            doubling_time=20,  # Fast-growing bacteria (e.g., skin infection)
            total_time_min=60,
            B_protocol=protocol,
            B_max=30.0
        )
        
        # Save data
        df3 = pd.DataFrame(results3)
        df3.to_csv(f'data1/growth_dynamics_{protocol}.csv', index=False)
        
        # Plot
        plot_analysis_3(results3, 
                       save_path=f'figures1/analysis/growth_control_{protocol}.png')
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  data1/S_vs_density_B*.csv")
    print("  data1/growth_dynamics_*.csv")
    print("  figures1/analysis/*.png")


if __name__ == "__main__":
    main()
