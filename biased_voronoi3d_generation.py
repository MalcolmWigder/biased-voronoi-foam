"""
Biased 3D Voronoi foam simulation across multiple bias levels,
with matched Poisson-Voronoi baselines.

For each bias level and run:
  - Start with a single seed
  - At each generation: compute 3D Voronoi, select top (bias fraction) cells
    by volume, divide each into two daughters placed symmetrically at
    distance r = 0.22 * V^(1/3) from the centroid along a random axis
  - Save seeds, volumes, centroids, degrees, and edge list per generation
  - Generate a matched Poisson baseline (same cell count, uniform random seeds)

Runs are parallelised across CPU cores.

Output: foam_data.h5
  /bias_0.10/run_00/generation_00/{seeds, volumes, centroids, degrees, edges}
  /poisson/bias_0.10/run_00/generation_00/{seeds, volumes, centroids, degrees, edges}
  ...
"""

import numpy as np
import h5py
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

# ── Parameters ────────────────────────────────────────────────────────────────

L           = 100           # domain side length (arbitrary units)
GRID        = 150           # voxels per axis (150^3 = 3.375M voxels)
MAX_CELLS   = 2000          # stop a run once cell count reaches this
MAX_GENS    = 200           # safety cap
BIAS_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
N_RUNS      = 20
DAUGHTER_R  = 0.22          # r = DAUGHTER_R * V^(1/3)
OUTPUT_FILE = "foam_data_periodic.h5"
N_WORKERS   = min(os.cpu_count() or 1, N_RUNS * len(BIAS_LEVELS))

# Voxel grid is computed lazily per-process (important for spawn-based multiprocessing)
_vox_coords = None
_vox_vol    = None

def _ensure_grid():
    global _vox_coords, _vox_vol
    if _vox_coords is None:
        step = L / GRID
        _vox_coords = (
            np.mgrid[0:GRID, 0:GRID, 0:GRID].reshape(3, -1).T * step + step / 2
        )
        _vox_vol = step ** 3


# ── Core functions ─────────────────────────────────────────────────────────────

def voronoi_labels(seeds):
    """Assign each voxel to its nearest seed under periodic BCs. Returns flat int array (GRID^3,)."""
    _ensure_grid()
    _, labels = KDTree(seeds, boxsize=L).query(_vox_coords)
    return labels


def cell_properties(labels, seeds):
    """
    Compute volumes and centroids.  Centroids use the circular mean so that
    cells straddling the periodic boundary are handled correctly.
    """
    _ensure_grid()
    n      = len(seeds)
    counts = np.bincount(labels, minlength=n)
    nonzero = counts > 0

    # Circular mean: map positions to angles on [0, 2π), average sin/cos,
    # then convert back.  Correct for any wrap-around cell.
    angles    = 2 * np.pi * _vox_coords / L          # (N_vox, 3)
    centroids = np.zeros((n, 3))
    for d in range(3):
        sin_sum = np.bincount(labels, weights=np.sin(angles[:, d]), minlength=n)
        cos_sum = np.bincount(labels, weights=np.cos(angles[:, d]), minlength=n)
        centroids[nonzero, d] = (
            np.arctan2(-sin_sum[nonzero] / counts[nonzero],
                       -cos_sum[nonzero] / counts[nonzero]) + np.pi
        ) * L / (2 * np.pi)

    centroids[~nonzero] = seeds[~nonzero]

    return counts * _vox_vol, centroids


def adjacency_edges(labels):
    """
    Find all pairs of cells sharing a voxel face, including wrap-around faces
    at the periodic boundary (last layer vs. first layer on each axis).
    """
    grid  = labels.reshape(GRID, GRID, GRID)
    parts = []
    for axis in range(3):
        # Interior faces
        sl_a = [slice(None)] * 3; sl_a[axis] = slice(None, -1)
        sl_b = [slice(None)] * 3; sl_b[axis] = slice(1, None)
        a = grid[tuple(sl_a)].ravel()
        b = grid[tuple(sl_b)].ravel()
        diff = a != b
        if diff.any():
            pairs = np.column_stack([a[diff], b[diff]])
            pairs = np.where(pairs[:, :1] < pairs[:, 1:], pairs, pairs[:, ::-1])
            parts.append(pairs)

        # Periodic wrap face: last layer vs. first layer
        sl_last  = [slice(None)] * 3; sl_last[axis]  = -1
        sl_first = [slice(None)] * 3; sl_first[axis] = 0
        a_wrap = grid[tuple(sl_last)].ravel()
        b_wrap = grid[tuple(sl_first)].ravel()
        diff_w = a_wrap != b_wrap
        if diff_w.any():
            pairs_w = np.column_stack([a_wrap[diff_w], b_wrap[diff_w]])
            pairs_w = np.where(pairs_w[:, :1] < pairs_w[:, 1:], pairs_w, pairs_w[:, ::-1])
            parts.append(pairs_w)

    if not parts:
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(np.vstack(parts), axis=0).astype(np.int32)


def cell_degrees(edges, n_cells):
    deg = np.zeros(n_cells, dtype=np.int32)
    if len(edges):
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
    return deg


def divide(seeds, volumes, centroids, bias):
    n        = len(seeds)
    n_divide = max(1, int(np.ceil(bias * n)))
    order    = np.argsort(volumes)[::-1]
    div_idx  = order[:n_divide]
    kep_idx  = order[n_divide:]

    r    = DAUGHTER_R * volumes[div_idx] ** (1 / 3)
    axes = np.random.randn(n_divide, 3)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    offsets   = r[:, np.newaxis] * axes
    daughters = np.vstack([
        centroids[div_idx] + offsets,
        centroids[div_idx] - offsets,
    ])

    new_seeds = np.vstack([seeds[kep_idx], daughters])
    return new_seeds % L


# ── Poisson baseline ──────────────────────────────────────────────────────────

def poisson_baseline(n_cells, rng_seed):
    rng = np.random.RandomState(rng_seed)
    seeds = rng.uniform(0, L, size=(n_cells, 3))

    labels    = voronoi_labels(seeds)
    vols, cen = cell_properties(labels, seeds)
    edges     = adjacency_edges(labels)
    degs      = cell_degrees(edges, len(seeds))

    return dict(
        seeds     = seeds,
        volumes   = vols,
        centroids = cen,
        degrees   = degs,
        edges     = edges,
    )


# ── Single run (called in worker process) ─────────────────────────────────────

def run_simulation(args):
    """
    Run one biased simulation + matched Poisson baselines.
    Returns (bias, run_idx, biased_results, poisson_results).
    """
    bias, run_idx, rng_seed = args

    np.random.seed(rng_seed)
    seeds   = np.random.uniform(0, L, size=(1, 3))
    biased_results  = []
    poisson_results = []

    for gen in range(MAX_GENS + 1):
        labels    = voronoi_labels(seeds)
        vols, cen = cell_properties(labels, seeds)
        edges     = adjacency_edges(labels)
        degs      = cell_degrees(edges, len(seeds))

        biased_results.append(dict(
            seeds     = seeds.copy(),
            volumes   = vols,
            centroids = cen,
            degrees   = degs,
            edges     = edges,
        ))

        poisson_seed = rng_seed * 10000 + gen * 100 + 77
        poisson_results.append(poisson_baseline(len(seeds), poisson_seed))

        if len(seeds) >= MAX_CELLS:
            break

        seeds = divide(seeds, vols, cen, bias)

    return (bias, run_idx, biased_results, poisson_results)


# ── HDF5 I/O ───────────────────────────────────────────────────────────────────

_H5_OPTS = dict(compression="gzip", compression_opts=4)

def save_generation(grp, data):
    grp.create_dataset("seeds",     data=data["seeds"],     **_H5_OPTS)
    grp.create_dataset("volumes",   data=data["volumes"],   **_H5_OPTS)
    grp.create_dataset("centroids", data=data["centroids"], **_H5_OPTS)
    grp.create_dataset("degrees",   data=data["degrees"],   **_H5_OPTS)
    grp.create_dataset("edges",     data=data["edges"],     **_H5_OPTS)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Build job list
    jobs = []
    for bias in BIAS_LEVELS:
        for run in range(N_RUNS):
            rng_seed = int(bias * 1000) + run
            jobs.append((bias, run, rng_seed))

    total_jobs = len(jobs)
    print(f"Launching {total_jobs} runs across {N_WORKERS} workers "
          f"({len(BIAS_LEVELS)} bias levels x {N_RUNS} runs, stop at {MAX_CELLS} cells)")

    # Run in parallel, collect results
    all_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_simulation, job): job for job in jobs}
        with tqdm(total=total_jobs, desc="runs", unit="run",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                             "[{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            for future in as_completed(futures):
                bias, run_idx, biased, poisson = future.result()
                n_gens = len(biased) - 1
                final_cells = len(biased[-1]["seeds"])
                all_results.append((bias, run_idx, biased, poisson))
                pbar.set_postfix_str(
                    f"done: bias={bias:.2f} run={run_idx} "
                    f"→ {n_gens} gens, {final_cells} cells"
                )
                pbar.update(1)

    # Write to HDF5 serially (fast compared to simulation)
    print("Writing HDF5...")
    all_results.sort(key=lambda r: (r[0], r[1]))

    with h5py.File(OUTPUT_FILE, "w") as f:
        f.attrs.update(dict(L=L, grid=GRID, daughter_r=DAUGHTER_R,
                            max_cells=MAX_CELLS))

        for bias, run_idx, biased, poisson in all_results:
            n_gens = len(biased) - 1

            bias_grp    = f.require_group(f"bias_{bias:.2f}")
            poisson_grp = f.require_group(f"poisson/bias_{bias:.2f}")

            run_grp = bias_grp.require_group(f"run_{run_idx:02d}")
            run_grp.attrs.update(dict(bias=bias, generations=n_gens))
            for gen, data in enumerate(biased):
                save_generation(
                    run_grp.require_group(f"generation_{gen:02d}"), data
                )

            p_run_grp = poisson_grp.require_group(f"run_{run_idx:02d}")
            p_run_grp.attrs.update(dict(bias=bias, generations=n_gens))
            for gen, data in enumerate(poisson):
                save_generation(
                    p_run_grp.require_group(f"generation_{gen:02d}"), data
                )

        # Set group-level attrs from first run of each bias
        for bias in BIAS_LEVELS:
            for prefix in [f"bias_{bias:.2f}", f"poisson/bias_{bias:.2f}"]:
                grp = f[prefix]
                n_gens = int(grp["run_00"].attrs["generations"])
                grp.attrs.update(dict(bias=bias, generations=n_gens))

    print(f"Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
