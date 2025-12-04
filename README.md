# BiophysMCAT
Magnetically Controlled Active Turbulence
# Magnetically Controlled Bacterial Turbulence

**Authors**: Michael A. Incorvaia & David Gonzalez  
**Institution**: Georgia Tech, College of Sciences - Physics  
**Date**: November 2025

Computational study of magnetic control of active turbulence, based on [Beppu & Timonen (2024) *Commun. Phys.*](https://www.nature.com/articles/s42005-024-01707-5)

---

## 📋 Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas

# Run complete analysis suite
python scripts/run_all_analyses.py

# Or run individual analyses
python scripts/run_all_analyses.py basic      # Basic simulations only
python scripts/run_all_analyses.py density    # Density & time analysis
```

**Expected runtime**: 30-45 minutes for full suite

---

## 🗂️ Repository Structure

```
magnetic-bacterial-turbulence/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── src/                               # Core simulation code
│   ├── simulation.py                  # Langevin dynamics (refactored from original)
│   └── parameters.py                  # Centralized parameter configuration
│
├── analysis/                          # Analysis scripts (generate DATA)
│   └── nematic_vs_density_time.py     # Density & time-dependent studies
│
├── scripts/                           # Master control scripts
│   └── run_all_analyses.py            # Run complete analysis pipeline
│
├── data/                              # Generated data files (.csv, .npz)
│   ├── S_vs_density_B*.csv
│   ├── growth_dynamics_*.csv
│   └── aligned_state.xyz              # VMD trajectory
│
├── figures/                           # Generated figures (.png)
│   ├── simulation/                    # Basic snapshots
│   ├── results/                       # Phase diagrams, etc.
│   ├── analysis/                      # Density/time analysis
│   └── applications/                  # Application mapping
│
├── docs/                              # Text for poster/report
│   └── (to be created)
│
└── poster/                            # Final deliverables
    └── (PowerPoint/PDF)
```

---

## 🎯 Key Research Questions Addressed

### 1. **Nematic Order vs Time**
*How quickly does magnetic alignment occur at different densities?*

**Analysis**: `analysis/nematic_vs_density_time.py` → Analysis 1  
**Output**: `figures/analysis/S_vs_time_multiple_N.png`  
**Key Finding**: Higher density → slower alignment (more hydrodynamic interactions)

---

### 2. **Nematic Order vs Density**
*What's the critical density for turbulence? How does it depend on field strength?*

**Analysis**: `analysis/nematic_vs_density_time.py` → Analysis 2  
**Output**: `figures/analysis/S_vs_density_B*.png`  
**Key Finding**: Critical area fraction φ_c ≈ 0.2-0.3 for turbulence onset

---

### 3. **Bacterial Growth + Magnetic Control**
*Can magnetic fields maintain control as bacteria multiply (e.g., infection)?*

**Analysis**: `analysis/nematic_vs_density_time.py` → Analysis 3  
**Output**: `figures/analysis/growth_control_*.png`  
**Key Finding**: 
- **Ramped protocol**: B increases linearly with time → maintains S ≈ 0.8
- **Step protocol**: B increases at each doubling → better control
- **Constant B**: Eventually overwhelmed by exponential growth

**Real-world relevance**: Skin infections (doubling time ~20 min)

---

## 📊 Generated Figures for Poster

### **Panel 1: Introduction/Schematic**
- Cartoon of bacteria in ferrofluid with magnetic field
- *(Manually create in PowerPoint or use Figure 1 from paper)*

### **Panel 2: Mathematical Model**
- Langevin equations + parameter table
- Generated from `src/parameters.py`

### **Panel 3: Simulation Examples**
- `turbulence_baseline.png` (B=0 mT, chaotic)
- `turbulence_suppressed.png` (B=25 mT, aligned)
- Side-by-side comparison

### **Panel 4: Results - Density Effects**
- **Figure 4A**: `figures/analysis/S_vs_time_multiple_N.png`  
  Order parameter evolution for N=100, 200, 500, 1000, 2000
  
- **Figure 4B**: `figures/analysis/S_vs_density_B20.png`  
  Final order vs density at B=20 mT with application windows overlaid

### **Panel 5: Results - Growth Dynamics**
- **Figure 5A-D**: `figures/analysis/growth_control_ramped.png`  
  Four-panel figure showing:
  - Population growth (exponential)
  - Magnetic field protocol
  - Order parameter evolution
  - Trajectory in (φ, B) phase space

### **Panel 6: Applications**
- Table comparing applications (drug delivery, microfluidics, bioreactors, etc.)
- Operating windows mapped to simulation results

### **Panel 7: Conclusions**
- Key findings (3-5 bullet points)
- Future directions

---

## 🔬 Experimental Data Integration

### **Bacterial Densities from Literature**

From [Chlorella/Chlorococcum wastewater study](https://scijournals.onlinelibrary.wiley.com/doi/abs/10.1002/jctb.5837):

| Species | Day 0 | Day 5 |
|---------|-------|-------|
| *Chlorella vulgaris* | 2.58×10⁸ | 0.98×10⁸ cells/mL |
| *Chlorococcum sp.* | 2.62×10⁸ | 0.85×10⁸ cells/mL |

*(Note: These decrease over time due to wastewater treatment, not growth)*

### **Typical Bacterial Densities**

Defined in `src/parameters.py`:

```python
TYPICAL_DENSITIES = {
    'early_log_phase': 1e7,      # cells/mL
    'mid_log_phase': 1e8,
    'late_log_phase': 1e9,
    'stationary_phase': 1e10,
    'skin_infection': 1e8,       # Typical wound
    'biofilm': 1e11,             # Dense biofilm
}
```

### **Doubling Times**

From [bacterial growth literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC7126130/):

- **Fast** (20 min): *E. coli*, skin infections
- **Medium** (40 min): *B. subtilis* typical
- **Slow** (90 min): Nutrient-limited growth

---

## 🧮 Key Parameters

All parameters centralized in `src/parameters.py`:

### **Physical (Based on *B. subtilis*)**
- Swimming speed: `v0 = 15 μm/s`
- Rotational diffusion: `D_r = 0.5 rad²/s`
- Body length: `L = 7 μm`, radius: `R = 0.8 μm`

### **Magnetic Coupling**
- `κ(B) = 0.01 × B²` (calibrated to paper)
- At B=30 mT: κ ≈ 9 rad/s

### **Simulation**
- Time step: `dt = 0.01 s`
- Box size: `L = 200 μm`
- Periodic boundaries

---

## 📈 How to Use Results in Your Report

### **For the 2-Page Report**

#### **Introduction** (¼ page)
- Active turbulence background
- Magnetic control mechanism (ferrofluid creates torques)
- Why it matters: drug delivery, microfluidics, infection control

#### **Methods** (¼ page)
- Langevin dynamics equations (from `src/parameters.py`)
- Simulation details (N, L, dt, boundary conditions)
- Analysis methods (order parameter, correlation functions)

#### **Results** (1 page)
1. **Basic control demonstration**: B=0 vs B=30 mT comparison
2. **Density effects**: S(φ) curves at different B
3. **Growth dynamics**: Maintaining control during bacterial multiplication
4. **Application mapping**: Operating windows for different use cases

#### **Discussion/Conclusions** (½ page)
- Key findings:
  - Critical field B_c ≈ 15-20 mT for order
  - Alignment time τ ~ B⁻² (power law)
  - Density-dependent control: need stronger B for higher φ
  - Ramped B protocol can maintain control during growth
- Real-world implications:
  - Drug delivery: N ≈ 10¹⁰ cells/mL, B ≈ 20 mT, τ < 10s
  - Skin infection control: doubling time 20 min, need adaptive B
- Future work: 3D simulations, experimental validation

---

## 🚀 Next Steps

### **For Poster Completion**:
1. ✅ Run `python scripts/run_all_analyses.py` (30-45 min)
2. ✅ Review all figures in `figures/` directory
3. ⬜ Select 4-6 best figures for poster
4. ⬜ Create schematic diagram (bacteria + ferrofluid + B field)
5. ⬜ Write text sections in `docs/` (use this README as template)
6. ⬜ Assemble in PowerPoint using poster template
7. ⬜ Write 2-pager using data from `data/*.csv`

### **Optional Extensions** (if time):
- ⬜ Full 2D phase diagram (N vs B)
- ⬜ Correlation length analysis C(Δr)
- ⬜ Field switch-off dynamics (instability growth)
- ⬜ Vorticity analysis (turbulence quantification)
- ⬜ Comparison with different bacteria species (different doubling times)

---

## 📚 Key References

1. **Beppu & Timonen (2024)**: Main inspiration - magnetic control using ferrofluid  
   [*Commun. Phys.* 7, 216](https://www.nature.com/articles/s42005-024-01707-5)

2. **Alert et al. (2022)**: Active turbulence review  
   [*Annu. Rev. Condens. Matter Phys.* 13, 143](https://doi.org/10.1146/annurev-conmatphys-082321-035957)

3. **Wensink et al. (2012)**: Meso-scale turbulence in bacterial suspensions  
   [*PNAS* 109, 14308](https://doi.org/10.1073/pnas.1202032109)

4. **Bacterial density data**: Wastewater treatment study  
   [*J. Chem. Technol. Biotechnol.*](https://scijournals.onlinelibrary.wiley.com/doi/abs/10.1002/jctb.5837)

5. **Bacterial growth**: Growth kinetics review  
   [PMC7126130](https://pmc.ncbi.nlm.nih.gov/articles/PMC7126130/)

---

## 🤝 Collaboration Notes

**Division of Labor** (suggested):
- **Michael**: Simulation code, parameter tuning, VMD visualization
- **David**: Data analysis, figure generation, application mapping
- **Both**: Poster design, report writing, presentation prep

**Weekly Goals**:
- Week 1: ✅ Get simulations running, basic figures
- Week 2: ⏳ Density & growth analysis (current focus)
- Week 3: Poster assembly, report writing
- Week 4: Practice presentation, final revisions

---

## 📧 Contact

Michael Incorvaia: [email]  
David Gonzalez: [email]  

Georgia Tech Physics Department  
Project for: [Course Name/Number]  
Advisor: [Professor Name]

---

## 📝 License

This project is for academic purposes. Code is provided as-is for educational use.

When using this code, please cite:
- Beppu & Timonen (2024) for the physical model
- This repository for the computational implementation