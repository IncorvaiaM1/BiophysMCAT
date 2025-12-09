"""
Master script to run all analyses for magnetic bacterial turbulence project.
[JAX JIT-optimized version for faster computation]

This script orchestrates all simulations and generates all figures needed
for the poster and 2-page report.

Author: Michael A. Incorvaia
Date: November 2025
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))

# JAX imports for optimization
import jax
import jax.numpy as jnp
from functools import partial

# Create output directories
DIRS = [
    'data2',
    'figures2/MCATs',
    'figures2/results',
    'figures2/analysis',
    'figures2/applications',
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# JAX JIT-compiled utility functions for performance-critical operations
# ============================================================================

@jax.jit
def jax_calculate_average_order(S_values_array):
    """
    JAX-compiled calculation of average order parameter.
    
    Parameters:
    -----------
    S_values_array : jnp.ndarray
        Time series of order parameters
        
    Returns:
    --------
    avg_S : float
        Average order parameter
    """
    return jnp.mean(S_values_array)


@jax.jit
def jax_calculate_order_std(S_values_array):
    """
    JAX-compiled calculation of order parameter standard deviation.
    
    Parameters:
    -----------
    S_values_array : jnp.ndarray
        Time series of order parameters
        
    Returns:
    --------
    std_S : float
        Standard deviation of order
    """
    return jnp.std(S_values_array)


@jax.jit
def jax_find_steady_state_index(S_values_array, window_size=5):
    """
    JAX-compiled detection of steady state in order parameter.
    Uses sliding window approach - steady state when variance is low.
    
    Parameters:
    -----------
    S_values_array : jnp.ndarray
        Time series of order parameters
    window_size : int
        Size of sliding window for variance calculation
        
    Returns:
    --------
    index : int
        Index where steady state begins
    """
    # Calculate sliding window variances
    n = S_values_array.shape[0]
    # Pad array
    padded = jnp.pad(S_values_array, (window_size // 2, window_size // 2), mode='edge')
    
    # Simple approach: find point where values stabilize (low derivative)
    diffs = jnp.abs(jnp.diff(S_values_array))
    threshold = jnp.mean(diffs) * 0.1  # Threshold at 10% of mean change
    
    # Find first index where differences are consistently small
    below_threshold = diffs < threshold
    # Return point where we see stability
    indices = jnp.where(below_threshold, size=1)[0]
    return jnp.where(indices.size > 0, indices[0], jnp.array(n // 2, dtype=jnp.int32))


@jax.jit
def jax_batch_process_B_values(B_array):
    """
    JAX-compiled batch processing of magnetic field values.
    Useful for parameter sweeps.
    
    Parameters:
    -----------
    B_array : jnp.ndarray
        Array of magnetic field values
        
    Returns:
    --------
    processed : jnp.ndarray
        Processed field values
    """
    # Ensure values are in reasonable range
    return jnp.clip(B_array, 0, 100)  # Clip to 0-100 mT


@jax.jit
def jax_calculate_response_time(S_values_array, threshold=0.5):
    """
    JAX-compiled calculation of system response time.
    Time to reach a fraction of final order value.
    
    Parameters:
    -----------
    S_values_array : jnp.ndarray
        Time series of order parameters
    threshold : float
        Fraction of final value to reach
        
    Returns:
    --------
    response_index : int
        Index where threshold is first exceeded
    """
    final_S = S_values_array[-1]
    target = final_S * threshold
    
    exceeds = S_values_array > target
    indices = jnp.where(exceeds, size=1)[0]
    return jnp.where(indices.size > 0, indices[0], jnp.array(-1, dtype=jnp.int32))


@jax.jit
def jax_normalize_S_values(S_values_array):
    """
    JAX-compiled normalization of order parameter values to [0, 1].
    
    Parameters:
    -----------
    S_values_array : jnp.ndarray
        Raw order parameter values
        
    Returns:
    --------
    normalized : jnp.ndarray
        Normalized values
    """
    S_min = jnp.min(S_values_array)
    S_max = jnp.max(S_values_array)
    S_range = S_max - S_min
    return (S_values_array - S_min) / (S_range + 1e-10)


@jax.jit
def jax_estimate_relaxation_time(S_values_array, S_final=None):
    """
    JAX-compiled exponential relaxation time estimation.
    Fits S(t) ≈ S_final * (1 - exp(-t/tau))
    
    Parameters:
    -----------
    S_values_array : jnp.ndarray
        Time series of order parameters
    S_final : float, optional
        Final order value (default: use last value)
        
    Returns:
    --------
    tau_estimate : float
        Estimated relaxation time (in sample indices)
    """
    if S_final is None:
        S_final = S_values_array[-1]
    
    # Find where S reaches 63.2% of final (tau in exponential decay)
    target = S_final * 0.632
    exceeds = S_values_array > target
    indices = jnp.where(exceeds, size=1)[0]
    
    tau = jnp.where(indices.size > 0, indices[0], jnp.array(len(S_values_array) // 2, dtype=jnp.float32))
    return tau


@partial(jax.jit, static_argnames=('num_bins',))
def jax_histogram_order_values(S_values_array, num_bins=10):
    """
    JAX-compiled histogram of order parameter distribution.
    
    Parameters:
    -----------
    S_values_array : jnp.ndarray
        Order parameter values
    num_bins : int
        Number of histogram bins
        
    Returns:
    --------
    hist : jnp.ndarray
        Histogram counts
    """
    return jnp.histogram(S_values_array, bins=num_bins)[0]


# ============================================================================
# Original orchestration functions (with JAX optimizations integrated)
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_basic_examples():
    """Run the original example simulations with JAX optimizations."""
    print_header("STEP 1: Basic Simulation Examples")
    
    from MCATs1 import run_baseline_turbulence, run_magnetic_suppression, run_field_sweep
    
    print("[1/3] Running baseline turbulence (B=0)...")
    sim1 = run_baseline_turbulence()
    
    print("\n[2/3] Running magnetic suppression (B=25 mT)...")
    sim2 = run_magnetic_suppression()
    
    print("\n[3/3] Running field sweep...")
    B_vals, S_vals = run_field_sweep()
    
    # Use JAX for rapid analysis of results
    print("\nAnalyzing results with JAX acceleration...")
    S_jax = jnp.array(S_vals)
    avg_S = float(jax_calculate_average_order(S_jax))
    std_S = float(jax_calculate_order_std(S_jax))
    print(f"  Field sweep: <S> = {avg_S:.3f} ± {std_S:.3f}")
    
    # Save VMD trajectory
    print("\nSaving VMD trajectory...")
    sim2.save_vmd_trajectory('data/aligned_state.xyz')
    
    return {
        'baseline': sim1,
        'suppressed': sim2,
        'field_sweep': (B_vals, S_vals),
        'analysis': {
            'avg_S': avg_S,
            'std_S': std_S
        }
    }


def run_density_time_analysis():
    """Run density and time-dependent analyses."""
    print_header("STEP 2: Density & Time Analysis")
    
    # Import and run the analysis module
    # Now uses JAX JIT-optimized version if available
    try:
        import nematic_vs_density_time2 as nematic_module
        print("  [Using JAX-optimized analysis module: nematic_vs_density_time2]")
    except ImportError:
        import nematic_vs_density_time as nematic_module
        print("  [Using standard analysis module: nematic_vs_density_time]")
    
    nematic_module.main()


def run_phase_diagram():
    """Generate full 2D phase diagram with JAX acceleration."""
    print_header("STEP 3: Phase Diagram Generation")
    
    print("This will take 30-60 minutes...")
    print("Running parameter sweep over (N, B) space with JAX optimizations...")
    
    # Will be implemented in separate script
    print("(Placeholder - implement run_phase_diagram.py)")


def run_correlation_analysis():
    """Calculate spatial correlations with JAX acceleration."""
    print_header("STEP 4: Correlation Length Analysis")
    
    print("Measuring C(Δr) for different field strengths with JAX...")
    
    # Will be implemented
    print("(Placeholder - implement correlation_analysis.py)")


def run_application_analysis():
    """Map results to real-world applications."""
    print_header("STEP 5: Application Mapping")
    
    print("Overlaying application windows on phase diagrams...")
    
    # Will be implemented
    print("(Placeholder - implement application_windows.py)")


def generate_poster_figures():
    """Compile all figures needed for poster."""
    print_header("STEP 6: Generate Poster Figures")
    
    print("Creating publication-quality figures for poster...")
    
    # Will create a script that pulls from all analyses
    print("(Placeholder - implement generate_poster_figures.py)")


def print_summary(start_time, results_summary=None):
    """Print summary of generated files."""
    elapsed = time.time() - start_time
    
    print_header("ANALYSIS COMPLETE!")
    
    print(f"Total runtime: {elapsed/60:.1f} minutes\n")
    
    if results_summary:
        print("Analysis Results:")
        if 'analysis' in results_summary:
            analysis = results_summary['analysis']
            print(f"  Average order parameter: {analysis.get('avg_S', 'N/A')}")
            print(f"  Order std deviation: {analysis.get('std_S', 'N/A')}\n")
    
    print("Generated files:")
    print("\n--- Data Files ---")
    print("  data/aligned_state.xyz")
    print("  data/S_vs_density_B*.csv")
    print("  data/growth_dynamics_*.csv")
    
    print("\n--- Figures: Basic Simulations ---")
    print("  turbulence_baseline.png")
    print("  turbulence_suppressed.png")
    print("  field_sweep_snapshots.png")
    print("  phase_diagram.png")
    
    print("\n--- Figures: Density & Time Analysis ---")
    print("  figures/analysis/S_vs_time_multiple_N.png")
    print("  figures/analysis/S_vs_density_B*.png")
    print("  figures/analysis/growth_control_*.png")
    
    print("\n--- Next Steps ---")
    print("  1. Review all figures in figures/ directory")
    print("  2. Select best figures for poster (aim for 4-6 main panels)")
    print("  3. Import into PowerPoint/Illustrator")
    print("  4. Add text from docs/*.md files")
    print("  5. Write 2-pager using data/*.csv results")
    
    print("\n" + "="*80)


def main():
    """Run complete analysis pipeline with JAX optimizations."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("  MAGNETIC BACTERIAL TURBULENCE - MASTER ANALYSIS SCRIPT")
    print("  [JAX JIT-optimized version]")
    print("  Michael A. Incorvaia")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Warm up JAX (compile functions on first use)
    print("\nWarming up JAX JIT compiler...")
    _ = jax_calculate_average_order(jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    print("  JAX compiled and ready.\n")
    
    # Run analyses in sequence
    results_basic = None
    
    try:
        # Step 1: Basic examples (fast, ~5 min)
        results_basic = run_basic_examples()
        
        # Step 2: Density & time analysis (moderate, ~20-30 min)
        run_density_time_analysis()
        
        # Step 3: Phase diagram (slow, ~30-60 min) - OPTIONAL
        # Uncomment if you have time and computational resources
        run_phase_diagram()
        
        # Step 4: Correlations (moderate, ~15 min) - OPTIONAL
        run_correlation_analysis()
        
        # Step 5: Application mapping (fast, ~5 min)
        run_application_analysis()
        
        # Step 6: Compile poster figures
        generate_poster_figures()
        
    except KeyboardInterrupt:
        print("\n\n*** Analysis interrupted by user ***")
        print("Partial results may be available in data/ and figures/ directories")
        return
    
    except Exception as e:
        print(f"\n\n*** ERROR: {e} ***")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary
    print_summary(start_time, results_basic)


if __name__ == "__main__":
    # Check if running with arguments to run specific analyses
    if len(sys.argv) > 1:
        analysis = sys.argv[1]
        
        if analysis == 'basic':
            results = run_basic_examples()
            print_summary(time.time(), results)
        elif analysis == 'density':
            run_density_time_analysis()
        elif analysis == 'phase':
            run_phase_diagram()
        elif analysis == 'correlation':
            run_correlation_analysis()
        elif analysis == 'app':
            run_application_analysis()
        elif analysis == 'poster':
            generate_poster_figures()
        else:
            print(f"Unknown analysis: {analysis}")
            print("Options: basic, density, phase, correlation, app, poster")
    else:
        # Run everything
        main()
