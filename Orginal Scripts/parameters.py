"""
Central parameter configuration for magnetic bacterial turbulence simulations.
All physical constants and simulation settings in one place.

Author: Michael A. Incorvaia 
Date: November 2025
"""

import numpy as np

# ============================================================================
# PHYSICAL PARAMETERS (Based on B. subtilis)
# ============================================================================

PHYSICAL = {
    # Self-propulsion
    'v0': 15.0,                    # Swimming speed (μm/s)
    
    # Diffusion coefficients
    'D_r': 0.5,                    # Rotational diffusion (rad²/s)
    'D_t': 0.1,                    # Translational diffusion (μm²/s)
    
    # Interaction ranges
    'R_align': 10.0,               # Alignment interaction range (μm)
    'R_rep': 3.0,                  # Repulsion range (μm, ~body length)
    
    # Interaction strengths
    'omega_align': 0.5,            # Alignment strength (1/s)
    'F_rep': 50.0,                 # Repulsion force (μm/s²)
    
    # Magnetic coupling
    'kappa_prefactor': 0.01,       # κ = kappa_prefactor * B² (1/s per mT²)
    
    # Bacterial dimensions (for area fraction calculations)
    'L_bacteria': 7.0,             # Body length (μm)
    'R_bacteria': 0.8,             # Body radius (μm)
}

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

SIMULATION = {
    'dt': 0.01,                    # Time step (seconds)
    'L': 200.0,                    # Box size (μm)
    'N_default': 500,              # Default number of particles
    'seed': 42,                    # Random seed for reproducibility
}

# ============================================================================
# MAGNETIC FIELD RANGES
# ============================================================================

MAGNETIC_FIELDS = {
    'B_sweep': [0, 5, 10, 15, 20, 25, 30],          # Standard sweep (mT)
    'B_fine': np.linspace(0, 30, 31).tolist(),      # Fine-grained sweep
    'B_transition': np.linspace(10, 25, 16).tolist(), # Focus on transition
}

# ============================================================================
# DENSITY RANGES
# ============================================================================

# Particle numbers for density studies
DENSITY_RANGES = {
    # Standard particle counts
    'N_sweep': [100, 200, 500, 1000, 2000],
    
    # Fine-grained for phase diagram
    'N_phase_diagram': [100, 150, 200, 300, 500, 750, 1000, 1500, 2000],
    
    # From bacterial literature (cells/mL converted to simulation units)
    # Based on: https://pmc.ncbi.nlm.nih.gov/articles/PMC7126130/
    'biological_densities': {
        'dilute_culture': 1e8,      # Early exponential phase (cells/mL)
        'mid_culture': 1e9,         # Mid exponential phase
        'dense_culture': 1e10,      # Late exponential/stationary
        'biofilm': 1e11,            # Dense biofilm (cells/mL)
    },
}

def cells_per_ml_to_particle_count(density_cells_ml, volume_ml=1e-6):
    """
    Convert biological cell density to simulation particle count.
    
    Parameters:
    -----------
    density_cells_ml : float
        Cell density in cells/mL
    volume_ml : float
        Simulation volume in mL (default: 1 μL = 1e-6 mL)
        
    Returns:
    --------
    N : int
        Number of particles for simulation
    """
    # For quasi-2D simulation with height h ≈ 60 μm (from paper)
    L = SIMULATION['L']  # μm
    h = 60.0  # μm
    volume_um3 = L * L * h
    volume_ml = volume_um3 * 1e-15  # Convert μm³ to mL
    
    N = int(density_cells_ml * volume_ml)
    return max(N, 10)  # Ensure at least 10 particles

def particle_count_to_area_fraction(N, L=None):
    """
    Convert particle count to area fraction φ.
    
    φ = N * π * R² / L²
    
    Parameters:
    -----------
    N : int
        Number of particles
    L : float
        Box size (μm), defaults to SIMULATION['L']
        
    Returns:
    --------
    phi : float
        Area fraction
    """
    if L is None:
        L = SIMULATION['L']
    
    R = PHYSICAL['R_bacteria']
    A_particle = np.pi * R**2
    A_box = L**2
    
    phi = N * A_particle / A_box
    return phi

def area_fraction_to_particle_count(phi, L=None):
    """
    Convert area fraction φ to particle count N.
    
    Parameters:
    -----------
    phi : float
        Target area fraction
    L : float
        Box size (μm)
        
    Returns:
    --------
    N : int
        Number of particles
    """
    if L is None:
        L = SIMULATION['L']
    
    R = PHYSICAL['R_bacteria']
    A_particle = np.pi * R**2
    A_box = L**2
    
    N = int(phi * A_box / A_particle)
    return max(N, 10)

# ============================================================================
# BACTERIAL GROWTH PARAMETERS
# ============================================================================

GROWTH = {
    # Doubling times (minutes) - from literature
    'doubling_time_fast': 20,      # E.g., E. coli in rich medium
    'doubling_time_medium': 40,    # B. subtilis typical
    'doubling_time_slow': 90,      # Nutrient-limited
    
    # Growth rates (1/min)
    'mu_fast': np.log(2) / 20,     # 0.0347 min⁻¹
    'mu_medium': np.log(2) / 40,   # 0.0173 min⁻¹
    'mu_slow': np.log(2) / 90,     # 0.0077 min⁻¹
}

def calculate_population(N0, time_minutes, doubling_time):
    """
    Calculate bacterial population at time t.
    
    N(t) = N0 * 2^(t / t_double)
    
    Parameters:
    -----------
    N0 : int
        Initial population
    time_minutes : float
        Time elapsed (minutes)
    doubling_time : float
        Doubling time (minutes)
        
    Returns:
    --------
    N : int
        Population at time t
    """
    generations = time_minutes / doubling_time
    N = int(N0 * (2 ** generations))
    return N

# ============================================================================
# APPLICATION-SPECIFIC PARAMETERS
# ============================================================================

APPLICATIONS = {
    'drug_delivery': {
        'density_range': (1e10, 1e11),     # cells/mL
        'response_time_max': 10,            # seconds
        'B_max': 50,                        # mT (MRI-safe)
        'phi_range': (0.15, 0.35),
        'color': '#e74c3c',                 # Red
        'description': 'Targeted drug delivery to tumors',
    },
    
    'microfluidic_valve': {
        'density_range': (1e11, 5e11),
        'response_time_max': 1,
        'B_max': 100,                       # No human exposure constraint
        'phi_range': (0.3, 0.5),
        'color': '#3498db',                 # Blue
        'description': 'On-demand microfluidic flow control',
    },
    
    'bioreactor_mixing': {
        'density_range': (1e9, 5e10),
        'response_time_max': 60,
        'B_max': 30,
        'phi_range': (0.1, 0.3),
        'color': '#2ecc71',                 # Green
        'description': 'Enhanced mixing in bioreactors',
    },
    
    'biosensor_swarm': {
        'density_range': (1e8, 1e9),
        'response_time_max': 30,
        'B_max': 20,
        'phi_range': (0.05, 0.15),
        'color': '#f39c12',                 # Orange
        'description': 'Distributed chemical/toxin sensing',
    },
    
    'wound_infection_control': {
        'density_range': (1e9, 1e10),       # Typical wound biofilm density
        'response_time_max': 5,             # Rapid response needed
        'B_max': 30,
        'phi_range': (0.15, 0.30),
        'color': '#9b59b6',                 # Purple
        'description': 'Wound biofilm disruption',
    },
}

# ============================================================================
# EXPERIMENTAL DATA FROM LITERATURE
# ============================================================================

# From: https://scijournals.onlinelibrary.wiley.com/doi/abs/10.1002/jctb.5837
# Bacterial density during wastewater treatment
EXPERIMENTAL_DENSITIES = {
    'chlorella_vulgaris': {
        'day_0': 2.58e8,
        'day_1': 1.83e8,
        'day_3': 1.65e8,
        'day_4': 1.02e8,
        'day_5': 0.98e8,
    },
    'chlorococcum_sp': {
        'day_0': 2.62e8,
        'day_1': 1.53e8,
        'day_3': 1.44e8,
        'day_4': 1.04e8,
        'day_5': 0.85e8,
    },
    # Note: These decrease over time (wastewater treatment, not growth)
}

# Typical bacterial densities from literature (cells/mL)
TYPICAL_DENSITIES = {
    'early_log_phase': 1e7,
    'mid_log_phase': 1e8,
    'late_log_phase': 1e9,
    'stationary_phase': 1e10,
    'skin_infection': 1e8,          # Typical wound infection
    'biofilm': 1e11,                # Dense biofilm
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_parameter_summary():
    """Print a summary of all parameters for documentation."""
    print("="*70)
    print("PARAMETER SUMMARY")
    print("="*70)
    
    print("\n--- Physical Parameters ---")
    for key, val in PHYSICAL.items():
        print(f"  {key:20s} = {val}")
    
    print("\n--- Simulation Settings ---")
    for key, val in SIMULATION.items():
        print(f"  {key:20s} = {val}")
    
    print("\n--- Magnetic Field Sweeps ---")
    print(f"  Standard: {MAGNETIC_FIELDS['B_sweep']}")
    
    print("\n--- Density Ranges ---")
    print(f"  Particle counts: {DENSITY_RANGES['N_sweep']}")
    
    print("\n--- Applications ---")
    for app_name, specs in APPLICATIONS.items():
        print(f"\n  {app_name}:")
        print(f"    Density: {specs['density_range']} cells/mL")
        print(f"    φ range: {specs['phi_range']}")
        print(f"    Max response time: {specs['response_time_max']} s")
        print(f"    Max field: {specs['B_max']} mT")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print_parameter_summary()
    
    # Example conversions
    print("\n--- Example Conversions ---")
    N = 500
    phi = particle_count_to_area_fraction(N)
    print(f"N = {N} particles → φ = {phi:.4f}")
    
    density = 1e10  # cells/mL
    N_bio = cells_per_ml_to_particle_count(density)
    print(f"Density = {density:.1e} cells/mL → N = {N_bio} particles")
    
    # Growth example
    N0 = 100
    t = 60  # minutes (1 hour)
    doubling = 20  # minutes
    Nt = calculate_population(N0, t, doubling)
    print(f"\nGrowth: N0={N0}, t={t} min, t_double={doubling} min")
    print(f"  → N(t) = {Nt} (factor of {Nt/N0:.1f})")