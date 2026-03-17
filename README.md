# Scaling Laws in Biologically-Motivated 3D Voronoi Foams

**Malcolm Wigder** · Rice University

> Companion code and data for *"Scaling Laws in Biologically-Motivated Three-Dimensional Voronoi Foams"*

---

## Overview

This project simulates biological cell division in 3D by iterative Voronoi tessellation. Starting from a single seed in a periodic cubic domain, the largest cells divide preferentially at each generation — mimicking the size-biased division rule observed in epithelial tissues. Five division-bias levels (10%–90%) are swept across 20 independent runs each, and a matched Poisson-Voronoi baseline is generated at every generation for apples-to-apples comparison.

The four main questions answered:

| Question | Finding |
|---|---|
| Does the 3D Aboav-Weaire relation hold? | Yes — but with **inverted sign** ($b \approx -33 < 0$), opposite to the Poisson reference ($b > 0$) |
| Does Lewis's Law hold? | No — $R^2 < 0.20$ uniformly across all bias levels and generations |
| Does volume uniformity converge? | Yes — to a **bias-independent floor** of $\mathrm{CV} \approx 0.15$–$0.20$ |
| What is the network-level imprint of division? | Assortativity rises above the Poisson baseline ($r \approx 0.29$ vs $0.17$); clustering decays to it |

An exact graph identity links the sign of the Aboav-Weaire $b$ parameter to the sign of Newman degree assortativity, explaining both the 2D–3D inversion and the universality across bias levels analytically.

---

## Repository Structure

```
biased-voronoi-foam/
├── biased_voronoi3d_generation.py   # Simulation: runs all bias levels, outputs HDF5
├── biased_voronoi3d_analysis.py     # Analysis & figure generation from HDF5
├── figures/                         # All paper figures (PDF)
│   ├── aboav_weaire_scatter.pdf
│   ├── aboav_weaire_params.pdf
│   ├── aboav_weaire_rmse.pdf
│   ├── cell_count_growth.pdf
│   ├── degree_distributions.pdf
│   ├── lewis_scatter.pdf
│   ├── lewis_r2.pdf
│   ├── lewis_c.pdf
│   ├── cv_over_time.pdf
│   ├── cv_decay_fits.pdf
│   ├── network_stats.pdf
│   ├── clustering_by_degree.pdf
│   └── genealogical_clustering.pdf
├── foam_letter_MW.tex               # LaTeX manuscript source
├── foam_letter_MW.pdf               # Compiled paper
├── foam_refs.bib                    # Bibliography
└── .gitignore
```

---

## Data

The simulation output (`foam_data_periodic.h5`, ~200 MB) is too large for GitHub and is hosted on Zenodo:

> **DOI: [10.5281/zenodo.19058387](https://doi.org/10.5281/zenodo.19058387)**

Download the HDF5 file and place it in the repo root before running the analysis script.

**HDF5 structure:**
```
/bias_0.10/run_00/generation_00/{seeds, volumes, centroids, degrees, edges}
/bias_0.25/...
...
/poisson/bias_0.10/run_00/generation_00/{...}
...
```

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install numpy scipy h5py networkx matplotlib tqdm
```

### 2. Run the simulation

```bash
python biased_voronoi3d_generation.py
```

This runs all 5 bias levels × 20 independent runs in parallel (using all available CPU cores) and writes `foam_data_periodic.h5`. Expect several hours on a modern multi-core machine.

**Key parameters** (editable at the top of the script):

| Parameter | Default | Description |
|---|---|---|
| `L` | 100 | Domain side length (arbitrary units) |
| `GRID` | 150 | Voxels per axis (150³ = 3.375M voxels) |
| `MAX_CELLS` | 2000 | Cells per run at termination |
| `BIAS_LEVELS` | [0.10, 0.25, 0.50, 0.75, 0.90] | Fraction of cells dividing per generation |
| `N_RUNS` | 20 | Independent runs per bias level |
| `DAUGHTER_R` | 0.22 | Daughter seed offset: $r = 0.22 \cdot V^{1/3}$ |

### 3. Generate figures

```bash
python biased_voronoi3d_analysis.py
```

Reads `foam_data_periodic.h5` and writes all figures to `figures/`. Figures map to the paper as follows:

| Figure in paper | File |
|---|---|
| Fig. 2 — Cell count growth | `cell_count_growth.pdf` |
| Fig. 3 — Degree distributions | `degree_distributions.pdf` |
| Fig. 4 — Aboav-Weaire scatter | `aboav_weaire_scatter.pdf` |
| Fig. 5 — AW parameters vs generation | `aboav_weaire_params.pdf` |
| Fig. 6 — AW RMSE | `aboav_weaire_rmse.pdf` |
| Fig. 7 — Lewis scatter | `lewis_scatter.pdf` |
| Fig. 8 — Lewis diagnostics | `lewis_r2.pdf`, `lewis_c.pdf` |
| Fig. 9 — CV over time | `cv_over_time.pdf` |
| Fig. 10 — CV decay fits | `cv_decay_fits.pdf` |
| Fig. 11 — Network statistics | `network_stats.pdf` |

---

## Methods Summary

**Simulation domain:** Cubic $[0, L]^3$ with $L = 100$, discretised to a $150^3$ voxel grid (~1700 voxels/cell at 2000 cells). Periodic boundary conditions throughout — periodic distance is used for Voronoi assignment, daughter seeds wrap modulo $L$, and cell centroids are computed via circular mean.

**Division rule:** At each generation the top fraction (bias level) of cells by volume each place two daughter seeds symmetrically around the parent centroid at distance $r = 0.22 \cdot V^{1/3}$ along a uniformly random axis. The full Voronoi diagram is recomputed from scratch each generation.

**Baseline:** A matched Poisson-Voronoi foam is generated at every generation by drawing the same number of seeds uniformly at random, isolating the effect of the division rule from generic 3D Voronoi geometry.

**Adjacency graph:** Cells sharing a polyhedral face are connected. Node attributes: voxel volume, centroid coordinates. Adjacency detection spans the periodic boundary.

---

## Citation

If you use this code or data, please cite:

```
Malcolm Wigder, "Scaling Laws in Biologically-Motivated Three-Dimensional
Voronoi Foams," (2025).
```

---

## Acknowledgments

Numerical analysis and figure production were assisted by Claude Code (Anthropic, 2025). The author takes full responsibility for all scientific content, results, and interpretations.
