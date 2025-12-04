"""
Advanced Analysis: Timescale, Kappa Dependence, and 2D Phase Diagrams
[JAX JIT-optimized version]

Addresses key gameplan questions:
1. Time at which order is achieved at which magnetic field strength (τ vs B)
2. How quickly particles align at different kappa values (alignment kinetics)
3. Timescale universality - can predict order achievement for any field
4. 2D phase diagram as function of N and B
5. Nematic order vs density for different use cases (drug delivery, infections, etc.)

Author: Michael A. Incorvaia
Date: November 2025
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Try to import JAX (optional, fallback to numpy if not available)
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX not available, using numpy fallback")
    JAX_AVAILABLE = False
    jnp = np

from src.MCATs1 import ActiveTurbulenceSimulation
from src.parameters2 import (
    DENSITY_RANGES, MAGNETIC_FIELDS, GROWTH, APPLICATIONS,
    particle_count_to_area_fraction, calculate_population,
    EXPERIMENTAL_DENSITIES, TYPICAL_DENSITIES
)


# ============================================================================
# JAX JIT-compiled utility functions for timescale analysis
# (with numpy fallbacks)
# ============================================================================

if JAX_AVAILABLE:
    @jax.jit
    def jax_find_alignment_time(S_array, threshold=0.8):
        """JAX-compiled exponential fit to find characteristic alignment time."""
        S_final = S_array[-1]
        target = S_final * threshold
        exceeds = S_array > target
        indices = jnp.where(exceeds, size=1)[0]
        return jnp.where(indices.size > 0, indices[0], jnp.array(-1, dtype=jnp.int32))
    
    @jax.jit
    def jax_calculate_kappa(B):
        """JAX-compiled kappa calculation: κ = 0.01 * B²"""
        return 0.01 * B**2
else:
    def jax_find_alignment_time(S_array, threshold=0.8):
        """Numpy fallback for alignment time finding."""
        S_final = S_array[-1]
        target = S_final * threshold
        exceeds = S_array > target
        indices = np.where(exceeds)[0]
        return indices[0] if len(indices) > 0 else -1
    
    def jax_calculate_kappa(B):
        """Numpy fallback for kappa calculation."""
        return 0.01 * B**2


# ============================================================================
# Analysis 1: Alignment Time vs Magnetic Field Strength (τ vs B)
# ============================================================================

def analyze_alignment_timescale_vs_B(N=500, B_values=None, T=30.0):
    """
    Measure time to achieve order as a function of magnetic field strength.
    
    Questions answered:
    - How long does alignment take at different field strengths?
    - Can we predict τ(B) relationship?
    - Is there a power law or exponential dependence?
    
    Parameters:
    -----------
    N : int
        Number of particles
    B_values : list
        Magnetic field strengths to test
    T : float
        Simulation time (seconds)
        
    Returns:
    --------
    results : dict
        {'B': [...], 'tau': [...], 'kappa': [...], 'S_final': [...]}
    """
    if B_values is None:
        B_values = np.linspace(5, 30, 11)  # 5 to 30 mT
    
    print(f"\n{'='*70}")
    print("ANALYSIS 1: Alignment Time vs Magnetic Field Strength")
    print(f"Testing B from {B_values[0]:.1f} to {B_values[-1]:.1f} mT")
    print(f"{'='*70}\n")
    
    results = {
        'B': [],
        'tau': [],
        'tau_time': [],  # In seconds
        'kappa': [],
        'S_final': [],
        'alignment_ratio': []  # τ * κ (should be ~constant if τ ~ κ⁻¹)
    }
    
    for B in B_values:
        print(f"B = {B:5.1f} mT...", end=' ')
        
        sim = ActiveTurbulenceSimulation(N=N, L=200.0, dt=0.01, seed=42)
        sim.set_magnetic_field(B)
        sim.run(T=T, save_interval=0.5)
        
        # Extract time series
        times = np.array([snap['time'] for snap in sim.trajectory_history])
        S_values = np.array(sim.order_parameter_history)
        
        # Find alignment time
        if JAX_AVAILABLE:
            S_jax = jnp.array(S_values)
            tau_idx = int(jax_find_alignment_time(S_jax, threshold=0.8))
        else:
            tau_idx = jax_find_alignment_time(S_values, threshold=0.8)
        
        if tau_idx >= 0 and tau_idx < len(times):
            tau_time = times[tau_idx]
        else:
            tau_time = np.nan
        
        # Calculate kappa
        if JAX_AVAILABLE:
            kappa = float(jax_calculate_kappa(jnp.array(B)))
        else:
            kappa = jax_calculate_kappa(B)
        
        # Final order
        S_final = S_values[-1]
        
        # Calculate alignment ratio τ * κ
        if not np.isnan(tau_time) and tau_time > 0:
            alignment_ratio = tau_time * kappa
        else:
            alignment_ratio = np.nan
        
        results['B'].append(B)
        results['tau'].append(tau_idx)
        results['tau_time'].append(tau_time)
        results['kappa'].append(kappa)
        results['S_final'].append(S_final)
        results['alignment_ratio'].append(alignment_ratio)
        
        print(f"τ = {tau_time:6.2f}s, κ = {kappa:6.3f}s⁻¹, S = {S_final:.3f}")
    
    return results


def plot_timescale_analysis_1(results, save_path='figures1/analysis/timescale_vs_B.png'):
    """Plot alignment time vs magnetic field strength."""
    print(f"\nPlotting timescale analysis...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Panel A: Alignment time vs B
    ax1 = fig.add_subplot(gs[0, 0])
    valid_idx = ~np.isnan(results['tau_time'])
    B_valid = np.array(results['B'])[valid_idx]
    tau_valid = np.array(results['tau_time'])[valid_idx]
    
    if len(tau_valid) > 0:
        ax1.semilogy(B_valid, tau_valid, 'o-', color='#e74c3c', markersize=10, linewidth=2)
        ax1.set_xlabel('Magnetic Field B (mT)', fontsize=12)
        ax1.set_ylabel('Alignment Time τ (s)', fontsize=12)
        ax1.set_title('Time to Achieve Order vs Field Strength', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, which='both')
        
        # Try to fit power law: τ ~ B^n
        if len(B_valid) > 2 and np.all(B_valid > 0) and np.all(tau_valid > 0):
            # Fit log-log
            coeffs = np.polyfit(np.log(B_valid), np.log(tau_valid), 1)
            power_law_exponent = coeffs[0]
            ax1.text(0.95, 0.95, f'Power law: τ ∝ B^{power_law_exponent:.2f}',
                    transform=ax1.transAxes, ha='right', va='top',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel B: Kappa vs B
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['B'], results['kappa'], 's-', color='#3498db', markersize=10, linewidth=2)
    ax2.set_xlabel('Magnetic Field B (mT)', fontsize=12)
    ax2.set_ylabel('Alignment Strength κ (1/s)', fontsize=12)
    ax2.set_title('Magnetic Coupling Strength vs Field', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add quadratic fit line
    B_fit = np.linspace(0, max(results['B']), 100)
    kappa_fit = 0.01 * B_fit**2
    ax2.plot(B_fit, kappa_fit, '--', color='gray', alpha=0.5, linewidth=1, label='κ = 0.01B²')
    ax2.legend()
    
    # Panel C: Final order vs B
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['B'], results['S_final'], 'o-', color='#2ecc71', markersize=10, linewidth=2)
    ax3.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Order threshold (0.8)')
    ax3.set_xlabel('Magnetic Field B (mT)', fontsize=12)
    ax3.set_ylabel('Final Nematic Order S', fontsize=12)
    ax3.set_title('Equilibrium Order vs Field Strength', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(-0.1, 1.1)
    
    # Panel D: τ * κ (should be constant if τ ~ κ⁻¹)
    ax4 = fig.add_subplot(gs[1, 1])
    alignment_ratio = np.array(results['alignment_ratio'])
    valid_ratio = ~np.isnan(alignment_ratio)
    
    if np.sum(valid_ratio) > 0:
        # FIX: Changed 'd=' to proper matplotlib format
        ax4.plot(np.array(results['B'])[valid_ratio], alignment_ratio[valid_ratio], 
                'o-', color='#9b59b6', markersize=10, linewidth=2)
        ax4.set_xlabel('Magnetic Field B (mT)', fontsize=12)
        ax4.set_ylabel('τ × κ (dimensionless)', fontsize=12)
        ax4.set_title('Universality Check: τ × κ Product', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        if np.sum(valid_ratio) > 1:
            mean_product = np.mean(alignment_ratio[valid_ratio])
            ax4.axhline(mean_product, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Mean ≈ {mean_product:.2f}')
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No valid alignment data', 
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


# ============================================================================
# Analysis 2: 2D Phase Diagram (N vs B)
# ============================================================================

def analyze_2d_phase_diagram(N_values=None, B_values=None, T=20.0):
    """
    Generate 2D phase diagram showing nematic order as function of (N, B).
    
    Questions answered:
    - How does critical field B_c depend on density?
    - What's the full phase space (N, B) → S?
    - Operating windows for different applications?
    
    Parameters:
    -----------
    N_values : list
        Particle counts to test
    B_values : list
        Magnetic field strengths to test
    T : float
        Equilibration time per simulation
        
    Returns:
    --------
    results : dict
        {'N': [...], 'B': [...], 'S': [...], 'grid': 2D array}
    """
    if N_values is None:
        # Original smaller range for testing:
        # N_values = [15, 25, 50, 75, 100, 200, 500, 1000, 2000]
        # Expanded range to explore bioreactor mixing and microfluidic valve regimes:
        N_values = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000]
    if B_values is None:
        # Full sweep across all magnetic fields up to 30 mT (11 points)
        B_values = np.linspace(0, 30, 11)  # 0 to 30 mT, 11 points
    
    print(f"\n{'='*70}")
    print("ANALYSIS 2: 2D Phase Diagram (Density vs Field Strength)")
    print(f"N: {N_values}, B: {list(B_values)}")
    print(f"{'='*70}\n")
    
    # Initialize grid
    S_grid = np.zeros((len(N_values), len(B_values)))
    
    results = {
        'N': N_values,
        'B': list(B_values),
        'N_arr': [],
        'B_arr': [],
        'S': [],
        'phi': []
    }
    
    for i, N in enumerate(N_values):
        phi = particle_count_to_area_fraction(N)
        print(f"N = {N:4d} (φ = {phi:.3f}): ", end='')
        
        for j, B in enumerate(B_values):
            print(f"B={B:5.1f}...", end=' ')
            
            sim = ActiveTurbulenceSimulation(N=N, L=200.0, dt=0.01, seed=42)
            sim.set_magnetic_field(B)
            sim.run(T=T, save_interval=1.0)
            
            # Final order
            S_final = sim.calculate_nematic_order()
            S_grid[i, j] = S_final
            
            results['N_arr'].append(N)
            results['B_arr'].append(B)
            results['S'].append(S_final)
            results['phi'].append(phi)
        
        print()
    
    results['grid'] = S_grid
    return results


def plot_2d_phase_diagram(results, save_path='figures1/analysis/phase_diagram_2d.png'):
    """Plot 2D phase diagram as heatmap."""
    print(f"\nPlotting 2D phase diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    B_values = np.array(results['B'])
    im = ax.imshow(results['grid'], cmap='RdYlGn', aspect='auto', origin='lower',
                   extent=[B_values[0], B_values[-1], 
                          0, len(results['N'])-1],
                   vmin=0, vmax=1)
    
    # Labels
    ax.set_xlabel('Magnetic Field B (mT)', fontsize=13)
    ax.set_ylabel('Particle Count N', fontsize=13)
    ax.set_title('Phase Diagram: Nematic Order S(N, B)', fontsize=15, fontweight='bold')
    
    # Y-axis ticks
    ax.set_yticks(range(len(results['N'])))
    ax.set_yticklabels([f"N={N}" for N in results['N']])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Nematic Order S')
    cbar.ax.set_ylabel('Nematic Order S', fontsize=12)
    
    # Add contour lines for S = 0.8 (order threshold)
    B_mesh, N_mesh = np.meshgrid(B_values, np.arange(len(results['N'])))
    contours = ax.contour(B_mesh, N_mesh, results['grid'],
                          levels=[0.8], colors='blue', linewidths=2, linestyles='dashed')
    ax.clabel(contours, inline=True, fontsize=10, fmt='S=0.8')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


# ============================================================================
# Analysis 3: Application Windows with Density Labels
# ============================================================================

def plot_application_windows(results_density, save_path='figures1/analysis/application_windows.png'):
    """
    Plot nematic order vs density with application use cases labeled.
    
    Shows which density ranges correspond to:
    - Drug delivery (high density)
    - Skin infections (medium-low density)
    - Biofilms (very high density)
    - etc.
    """
    print(f"\nPlotting application windows...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get unique B values (take first 3 for plotting)
    # B_unique = sorted(set(results_density['B_arr']))[:3]
    #here i can run the simulation and decide which B values to plot
    B_unique = sorted(set(results_density['B_arr']))[:7]
    
    for ax_idx, B in enumerate(B_unique):
        ax = axes[ax_idx]
        
        # Filter data for this B value
        mask = np.array(results_density['B_arr']) == B
        phi_vals = np.array(results_density['phi'])[mask]
        S_vals = np.array(results_density['S'])[mask]
        N_vals = np.array(results_density['N_arr'])[mask]
        
        # Sort by phi
        sort_idx = np.argsort(phi_vals)
        phi_vals = phi_vals[sort_idx]
        S_vals = S_vals[sort_idx]
        N_vals = N_vals[sort_idx]
        
        # Plot main curve
        ax.plot(phi_vals, S_vals, 'o-', color='#3498db', markersize=12, linewidth=3)
        ax.axhline(0.8, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Order threshold')
        
        # Add application windows
        app_colors = {
            'drug_delivery': '#2ecc71',
            'wound_infection_control': '#e74c3c',
            'bioreactor_mixing': '#9b59b6',
            'microfluidic_valve': '#f39c12'
        }
        
        for app_name, specs in APPLICATIONS.items():
            if 'phi_range' in specs:
                phi_min, phi_max = specs['phi_range']
                color = app_colors.get(app_name, specs.get('color', 'gray'))
                ax.axvspan(phi_min, phi_max, alpha=0.15, color=color)
                
                # Label in the middle of window
                mid_phi = (phi_min + phi_max) / 2
                ax.text(mid_phi, -0.15, app_name.replace('_', ' ').title(),
                       ha='center', va='top', fontsize=9, style='italic', rotation=15)
        
        ax.set_xlabel('Area Fraction φ', fontsize=12)
        ax.set_ylabel('Nematic Order S', fontsize=12)
        ax.set_title(f'Density-Order Relation\n(B = {B:.1f} mT)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.1)
        ax.legend(loc='lower left')
        
        # Add density labels on top
        for phi, N in zip(phi_vals, N_vals):
            ax.text(phi, 1.05, f'{N}', ha='center', fontsize=8, alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


# ============================================================================
# Master analysis function
# ============================================================================

def main():
    """Run all advanced analyses."""
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS: TIMESCALE, KAPPA, AND PHASE DIAGRAMS")
    print("="*70)
    
    # Analysis 1: Timescale vs B
    print("\n[1/3] ANALYSIS 1: Alignment Timescale vs Magnetic Field")
    timescale_results = analyze_alignment_timescale_vs_B(
        N=500,
        B_values=np.linspace(5, 30, 11),  # 5 to 30 mT
        T=30.0
    )
    plot_timescale_analysis_1(timescale_results)
    
    # Save timescale data
    df_timescale = pd.DataFrame({
        'B_mT': timescale_results['B'],
        'tau_seconds': timescale_results['tau_time'],
        'kappa_inv_s': timescale_results['kappa'],
        'S_final': timescale_results['S_final'],
        'tau_times_kappa': timescale_results['alignment_ratio']
    })
    os.makedirs('data1', exist_ok=True)
    df_timescale.to_csv('data1/timescale_vs_B.csv', index=False)
    print("Saved: data1/timescale_vs_B.csv")
    
    # Analysis 2: 2D Phase Diagram
    print("\n[2/3] ANALYSIS 2: 2D Phase Diagram (N vs B)")
    # Commented out old smaller range:
    # N_values=[100, 300, 500, 1000, 2000],
    # B_values=np.linspace(0, 30, 7),
    phase_results = analyze_2d_phase_diagram(
        N_values=[100, 500, 1000, 2000, 5000, 10000, 15000, 20000],
        B_values=np.linspace(0, 30, 11),  # Full range 0-30 mT with 11 points
        T=20.0
    )
    plot_2d_phase_diagram(phase_results)
    
    # Save phase diagram data
    df_phase = pd.DataFrame({
        'N': phase_results['N_arr'],
        'B_mT': phase_results['B_arr'],
        'S': phase_results['S'],
        'phi': phase_results['phi']
    })
    df_phase.to_csv('data1/phase_diagram_2d.csv', index=False)
    print("Saved: data1/phase_diagram_2d.csv")
    
    # Analysis 3: Application Windows
    print("\n[3/3] ANALYSIS 3: Application Windows with Density Labels")
    plot_application_windows(phase_results)
    
    print("\n" + "="*70)
    print("ALL ADVANCED ANALYSES COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  figures1/analysis/timescale_vs_B.png")
    print("  figures1/analysis/phase_diagram_2d.png")
    print("  figures1/analysis/application_windows.png")
    print("  data1/timescale_vs_B.csv")
    print("  data1/phase_diagram_2d.csv")
    print("  data1/application_windows.csv")

if __name__ == "__main__":
    main()