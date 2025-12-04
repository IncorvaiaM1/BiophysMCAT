"""
Master script to run all analyses for magnetic bacterial turbulence project.

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

# Create output directories
DIRS = [
    'data',
    'figures/MCATs',
    'figures/results',
    'figures/analysis',
    'figures/applications',
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_basic_examples():
    """Run the original example simulations."""
    print_header("STEP 1: Basic Simulation Examples")
    
    from MCATs1 import run_baseline_turbulence, run_magnetic_suppression, run_field_sweep
    
    print("[1/3] Running baseline turbulence (B=0)...")
    sim1 = run_baseline_turbulence()
    
    print("\n[2/3] Running magnetic suppression (B=25 mT)...")
    sim2 = run_magnetic_suppression()
    
    print("\n[3/3] Running field sweep...")
    B_vals, S_vals = run_field_sweep()
    
    # Save VMD trajectory
    print("\nSaving VMD trajectory...")
    sim2.save_vmd_trajectory('data/aligned_state.xyz')
    
    return {
        'baseline': sim1,
        'suppressed': sim2,
        'field_sweep': (B_vals, S_vals)
    }


def run_density_time_analysis():
    """Run density and time-dependent analyses."""
    print_header("STEP 2: Density & Time Analysis")
    
    # Import and run the analysis module
    import nematic_vs_density_time2
    nematic_vs_density_time2.main()


def run_phase_diagram():
    """Generate full 2D phase diagram."""
    print_header("STEP 3: Phase Diagram Generation")
    
    print("This will take 30-60 minutes...")
    print("Running parameter sweep over (N, B) space...")
    
    # Will be implemented in separate script
    print("(Placeholder - implement run_phase_diagram.py)")


def run_correlation_analysis():
    """Calculate spatial correlations."""
    print_header("STEP 4: Correlation Length Analysis")
    
    print("Measuring C(Δr) for different field strengths...")
    
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


def print_summary(start_time):
    """Print summary of generated files."""
    elapsed = time.time() - start_time
    
    print_header("ANALYSIS COMPLETE!")
    
    print(f"Total runtime: {elapsed/60:.1f} minutes\n")
    
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
    """Run complete analysis pipeline."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("  MAGNETIC BACTERIAL TURBULENCE - MASTER ANALYSIS SCRIPT")
    print("  Michael A. Incorvaia")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Run analyses in sequence
    try:
        # Step 1: Basic examples (fast, ~5 min)
        results_basic = run_basic_examples()
        
        # Step 2: Density & time analysis (moderate, ~20-30 min)
        run_density_time_analysis()
        
        # Step 3: Phase diagram (slow, ~30-60 min) - OPTIONAL
        # Uncomment if you have time and computational resources
        # run_phase_diagram()
        
        # Step 4: Correlations (moderate, ~15 min) - OPTIONAL
        # run_correlation_analysis()
        
        # Step 5: Application mapping (fast, ~5 min)
        # run_application_analysis()
        
        # Step 6: Compile poster figures
        # generate_poster_figures()
        
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
    print_summary(start_time)


if __name__ == "__main__":
    # Check if running with arguments to run specific analyses
    if len(sys.argv) > 1:
        analysis = sys.argv[1]
        
        if analysis == 'basic':
            run_basic_examples()
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