"""
foam_deep_anal.py

Reads foam_data.h5 and produces figures for the three research questions:

  RQ1 — 3D Aboav-Weaire Law:  m(n) = a + b/n across generations and bias levels
  RQ2 — 3D Lewis's Law:       V(n) ~ k(n - c), tracking R^2 and c across generations
  RQ3 — Volume uniformity:    CV_t decay and floor across bias levels

HDF5 structure:
  /bias_0.10/run_00/generation_00/{volumes, degrees, edges, seeds, centroids}
  /poisson/bias_0.10/run_00/generation_00/{...}

Figures saved to ./figures/
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.stats import linregress, pearsonr
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

DATA_FILE         = "foam_data_periodic.h5"
FIG_DIR           = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

POISSON_MEAN_DEGREE = 15.54   # Lazar et al. 2013
POISSON_COLOR       = "#888888"
BAND_ALPHA          = 0.20

# ── Clean helper functions ─────────────────────────────────────────────────────

def plot_band(ax, x, mean, std, color, label, alpha=BAND_ALPHA, lw=1.8):
    """Plot mean line + fill_between band (mean ± std)."""
    mean = np.asarray(mean, dtype=float)
    std  = np.asarray(std,  dtype=float)
    ax.plot(x, mean, color=color, lw=lw, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def get_poisson_runs(f, bias_key):
    """Return list of run groups from f['poisson/{bias_key}']. Returns [] if not found."""
    try:
        grp = f[f"poisson/{bias_key}"]
        return list(grp.values())
    except KeyError:
        return []


def mean_std(values):
    """Return (mean, std) of a list, ignoring NaN."""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


# ── Core computation helpers ───────────────────────────────────────────────────

def load_generation(run_grp, gen):
    """Return dict of arrays for one generation."""
    g = run_grp[f"generation_{gen:02d}"]
    return dict(
        volumes   = g["volumes"][:],
        degrees   = g["degrees"][:],
        centroids = g["centroids"][:],
        edges     = g["edges"][:],
        seeds     = g["seeds"][:],
    )


def all_bias_keys(f):
    return sorted(k for k in f.keys() if k.startswith("bias_"))


def bias_val(key):
    return float(key.split("_")[1])


def bias_color(bias, all_biases):
    idx = sorted(all_biases).index(bias)
    return cm.plasma(idx / max(len(all_biases) - 1, 1))


def _mn_data(degrees, edges):
    """
    Compute m(n) = mean neighbour degree for cells of each degree n.
    Returns (n_vals, m_vals) arrays filtered to 4 <= n <= 30,
    or None if there are too few data points.
    """
    if len(edges) == 0 or len(degrees) < 10:
        return None
    deg_sum = np.zeros(len(degrees), dtype=np.float64)
    np.add.at(deg_sum, edges[:, 0], degrees[edges[:, 1]])
    np.add.at(deg_sum, edges[:, 1], degrees[edges[:, 0]])
    node_deg = degrees.astype(np.float64)
    m_node   = np.where(node_deg > 0, deg_sum / node_deg, 0.0)
    unique_n = np.unique(node_deg[node_deg > 0])
    if len(unique_n) == 0:
        return None
    m_vals = np.array([m_node[node_deg == n].mean() for n in unique_n])
    mask   = (unique_n >= 4) & (unique_n <= 30)
    n_vals, m_vals = unique_n[mask], m_vals[mask]
    return (n_vals, m_vals) if len(n_vals) >= 4 else None



def aboav_weaire_fit(degrees, edges):
    """
    Compute m(n) and fit m(n) = a + b/n (legacy interface for param tracking).
    Returns (n_vals, m_vals, a, b, rmse, r2) or None.
    """
    res = _mn_data(degrees, edges)
    if res is None:
        return None
    n_vals, m_vals = res
    try:
        (a, b), _ = curve_fit(lambda n, a, b: a + b / n, n_vals, m_vals,
                               p0=[15.0, -30.0], maxfev=4000)
        fitted = a + b / n_vals
        rmse   = np.sqrt(np.mean((m_vals - fitted) ** 2))
        ss_res = np.sum((m_vals - fitted) ** 2)
        ss_tot = np.sum((m_vals - m_vals.mean()) ** 2)
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    except (RuntimeError, ValueError):
        a, b, rmse, r2 = np.nan, np.nan, np.nan, np.nan
    return n_vals, m_vals, a, b, rmse, r2






def lewis_fit(volumes, degrees, c):
    """OLS of V on (n - c). Returns slope k and R^2."""
    x = degrees.astype(np.float64) - c
    if x.std() == 0:
        return np.nan, np.nan
    slope, intercept, r, *_ = linregress(x, volumes)
    return slope, r ** 2


def best_lewis_c(volumes, degrees, c_range=range(9)):
    """Scan c in c_range, return best (c, k, R^2)."""
    best = (-1, np.nan, -1.0)
    for c in c_range:
        k, r2 = lewis_fit(volumes, degrees, c)
        if np.isfinite(r2) and r2 > best[2]:
            best = (c, k, r2)
    return best


# ── RQ1: Aboav-Weaire ─────────────────────────────────────────────────────────

def _aw_binned_mean_std(run_list, gen):
    """
    For a list of run groups, load generation gen and aggregate m(n) per
    degree bin across runs.  Returns (n_vals, m_mean, m_std) or (None, None, None).
    Uses try/except KeyError so missing generations are silently skipped.
    """
    # Collect per-degree m values across all runs
    from collections import defaultdict
    per_n = defaultdict(list)
    for run in run_list:
        try:
            data = load_generation(run, gen)
        except KeyError:
            continue
        res = aboav_weaire_fit(data["degrees"], data["edges"])
        if res is None:
            continue
        n_vals, m_vals, *_ = res
        for n, m in zip(n_vals, m_vals):
            per_n[n].append(m)

    if not per_n:
        return None, None, None

    ns = np.array(sorted(per_n.keys()))
    ms_mean = np.array([np.mean(per_n[n]) for n in ns])
    ms_std  = np.array([np.std(per_n[n])  for n in ns])
    return ns, ms_mean, ms_std


def _poisson_aw_reference(f, all_bias_keys_list):
    """
    Compute Poisson AW reference values (a, b) pooled across all biases
    at their respective final generations.  Returns (a_ref, b_ref) as
    single scalars — used as horizontal reference bands on param plots.
    """
    a_pool, b_pool = [], []
    for bkey in all_bias_keys_list:
        gens      = f[bkey].attrs["generations"]
        pois_runs = get_poisson_runs(f, bkey)
        for run in pois_runs:
            try:
                data = load_generation(run, gens)
            except KeyError:
                continue
            res = aboav_weaire_fit(data["degrees"], data["edges"])
            if res:
                _, _, a, b, rmse, r2 = res
                if np.isfinite(a): a_pool.append(a)
                if np.isfinite(b): b_pool.append(b)
    a_ref = float(np.mean(a_pool)) if a_pool else np.nan
    b_ref = float(np.mean(b_pool)) if b_pool else np.nan
    a_std = float(np.std(a_pool))  if a_pool else np.nan
    b_std = float(np.std(b_pool))  if b_pool else np.nan
    return a_ref, a_std, b_ref, b_std


def plot_aboav_weaire(f):
    """
    Figure 1: m(n) errorbar (mean ± std across runs) at final generation,
              one panel per bias.  Single gray Poisson reference on each panel.
    Figure 2: a and b vs generation (mean ± std band).
              One horizontal gray band = Poisson asymptotic value ± std.
    Figure 3: RMSE vs generation (mean ± std band).  No Poisson.
    """
    all_biases  = [bias_val(k) for k in all_bias_keys(f)]
    bkeys       = list(all_bias_keys(f))
    n_bias      = len(all_biases)

    # Pre-compute single Poisson AW reference (pooled across all biases)
    a_ref, a_ref_std, b_ref, b_ref_std = _poisson_aw_reference(f, bkeys)

    # ── Fig 1: m(n) combined — all bias levels + Poisson on one panel ─────────
    MARKERS = ['o', 's', '^', 'D', 'v']
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    # Poisson reference: compute for print summary only (not plotted on scatter)
    mid_key   = "bias_0.50" if "bias_0.50" in f else bkeys[len(bkeys) // 2]
    pois_runs = get_poisson_runs(f, mid_key)
    pns, pms_mean, _ = _aw_binned_mean_std(pois_runs, f[mid_key].attrs["generations"])

    # Biased foam: data points, one marker shape per bias level
    # Also accumulate pooled (n, m) for universal fits
    pooled_n, pooled_m = defaultdict(list), defaultdict(list)

    for marker, bkey in zip(MARKERS, bkeys):
        bias     = bias_val(bkey)
        gens     = f[bkey].attrs["generations"]
        color    = bias_color(bias, all_biases)
        run_list = list(f[bkey].values())

        ns, ms_mean, ms_std = _aw_binned_mean_std(run_list, gens)
        if ns is None:
            continue

        ax1.errorbar(ns, ms_mean, yerr=ms_std, fmt=marker, ms=5, capsize=3,
                     color=color, label=f"bias={bias:.2f}", zorder=2, alpha=0.85)

        for n, m in zip(ns, ms_mean):
            pooled_n[n].append(m)

    # Pooled mean m(n) across all bias levels (solid black)
    all_ns = np.array(sorted(pooled_n.keys()))
    all_ms = np.array([np.mean(pooled_n[n]) for n in all_ns])
    ax1.plot(all_ns, all_ms, "-", color="black", lw=2.0, zorder=3,
             label="pooled mean $m(n)$")

    # a + b/n fit to pooled mean (dashed black) — trim min/max degree bins
    fit_mask = (all_ns > all_ns.min()) & (all_ns < all_ns.max())
    try:
        (a_u, b_u), _ = curve_fit(lambda n, a, b: a + b / n,
                                   all_ns[fit_mask], all_ms[fit_mask],
                                   p0=[15.0, -30.0], maxfev=4000)
        fitted_u = a_u + b_u / all_ns[fit_mask]
        ss_res   = np.sum((all_ms[fit_mask] - fitted_u) ** 2)
        ss_tot   = np.sum((all_ms[fit_mask] - all_ms[fit_mask].mean()) ** 2)
        r2_u     = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
        n_fit_u  = np.linspace(all_ns.min(), all_ns.max(), 300)
        ax1.plot(n_fit_u, a_u + b_u / n_fit_u, "--", color="black", lw=2.0,
                 zorder=4,
                 label=f"$a+b/n$ ($a={a_u:.1f},\\,b={b_u:+.1f},\\,R^2={r2_u:.3f}$)")
    except RuntimeError:
        a_u = b_u = r2_u = np.nan

    print(f"\nAboav-Weaire pooled mean fit:")
    print(f"  a+b/n:  a={a_u:.4f}  b={b_u:.4f}  R2={r2_u:.4f}")

    ax1.set_xlabel("degree $n$")
    ax1.set_ylabel("$m(n)$")
    ax1.legend(fontsize=7, frameon=False)
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "aboav_weaire_scatter.pdf", dpi=150)
    plt.close(fig1)

    # ── Fig 2 & 3: a, b, RMSE vs generation with ±1 std band ──────────────────
    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))
    fig3, ax_rmse      = plt.subplots(figsize=(6, 4))

    for bkey in tqdm(bkeys, desc="Aboav-Weaire param curves"):
        bias      = bias_val(bkey)
        gens      = f[bkey].attrs["generations"]
        color     = bias_color(bias, all_biases)
        gen_range = np.arange(gens + 1)
        run_list  = list(f[bkey].values())

        a_mean_v, a_std_v = [], []
        b_mean_v, b_std_v = [], []
        r_mean_v, r_std_v = [], []

        for gen in gen_range:
            a_list, b_list, r_list = [], [], []
            for run in run_list:
                try:
                    data = load_generation(run, gen)
                except KeyError:
                    continue
                res = aboav_weaire_fit(data["degrees"], data["edges"])
                if res:
                    _, _, a, b, rmse, r2 = res
                    if np.isfinite(a):    a_list.append(a)
                    if np.isfinite(b):    b_list.append(b)
                    if np.isfinite(rmse): r_list.append(rmse)
            mu_a, sd_a = mean_std(a_list)
            mu_b, sd_b = mean_std(b_list)
            mu_r, sd_r = mean_std(r_list)
            a_mean_v.append(mu_a); a_std_v.append(sd_a)
            b_mean_v.append(mu_b); b_std_v.append(sd_b)
            r_mean_v.append(mu_r); r_std_v.append(sd_r)

        label = f"bias={bias:.2f}"
        plot_band(ax_a,    gen_range, a_mean_v, a_std_v, color, label)
        plot_band(ax_b,    gen_range, b_mean_v, b_std_v, color, label)
        plot_band(ax_rmse, gen_range, r_mean_v, r_std_v, color, label)

    # Single Poisson reference band on a and b plots only
    if np.isfinite(a_ref):
        ax_a.axhline(a_ref, color=POISSON_COLOR, lw=1.4, ls="--",
                     label=f"Poisson ($a={a_ref:.1f}$)")
        ax_a.axhspan(a_ref - a_ref_std, a_ref + a_ref_std,
                     color=POISSON_COLOR, alpha=0.10)
    if np.isfinite(b_ref):
        ax_b.axhline(b_ref, color=POISSON_COLOR, lw=1.4, ls="--",
                     label=f"Poisson ($b={b_ref:.1f}$)")
        ax_b.axhspan(b_ref - b_ref_std, b_ref + b_ref_std,
                     color=POISSON_COLOR, alpha=0.10)

    ax_a.set(xlabel="generation", ylabel="$a$")
    ax_b.set(xlabel="generation", ylabel="$b$")
    ax_a.legend(fontsize=7); ax_b.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "aboav_weaire_params.pdf", dpi=150)
    plt.close(fig2)

    ax_rmse.set(xlabel="generation", ylabel="RMSE")
    ax_rmse.legend(fontsize=7)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "aboav_weaire_rmse.pdf", dpi=150)
    plt.close(fig3)


# ── RQ2: Lewis's Law ───────────────────────────────────────────────────────────

def plot_lewis(f):
    """
    Figure 4: R^2 of best Lewis fit vs generation (mean ± std band), Poisson dashed.
    Figure 5: best-fit offset c vs generation (mean ± std band), Poisson dashed.
    Figure 6: V vs (n - c) scatter at final generation, one panel per bias
              (reduced alpha=0.25, marker size=4).
    """
    all_biases = [bias_val(k) for k in all_bias_keys(f)]

    fig4, ax_r2 = plt.subplots(figsize=(7, 4))
    fig5, ax_c  = plt.subplots(figsize=(7, 4))

    # Pre-compute single Poisson Lewis R² reference (pooled across all biases
    # at final generation — Poisson should be consistently high)
    pois_r2_pool = []
    for bkey in all_bias_keys(f):
        gens      = f[bkey].attrs["generations"]
        pois_runs = get_poisson_runs(f, bkey)
        for run in pois_runs:
            try:
                data = load_generation(run, gens)
            except KeyError:
                continue
            if len(data["volumes"]) < 20:
                continue
            _, _, r2 = best_lewis_c(data["volumes"], data["degrees"])
            if np.isfinite(r2):
                pois_r2_pool.append(r2)
    pois_r2_ref = float(np.mean(pois_r2_pool)) if pois_r2_pool else np.nan
    pois_r2_std = float(np.std(pois_r2_pool))  if pois_r2_pool else np.nan

    for bkey in tqdm(all_bias_keys(f), desc="Lewis param curves"):
        bias      = bias_val(bkey)
        gens      = f[bkey].attrs["generations"]
        color     = bias_color(bias, all_biases)
        run_list  = list(f[bkey].values())
        gen_range = np.arange(gens + 1)

        r2_mean_v, r2_std_v = [], []
        c_mean_v,  c_std_v  = [], []

        for gen in gen_range:
            r2_run, c_run = [], []
            for run in run_list:
                try:
                    data = load_generation(run, gen)
                except KeyError:
                    continue
                vols = data["volumes"]; degs = data["degrees"]
                if len(vols) < 20:
                    continue
                c, k, r2 = best_lewis_c(vols, degs)
                if np.isfinite(r2):
                    r2_run.append(r2); c_run.append(c)

            mu_r2, sd_r2 = mean_std(r2_run)
            mu_c,  sd_c  = mean_std(c_run)
            r2_mean_v.append(mu_r2); r2_std_v.append(sd_r2)
            c_mean_v.append(mu_c);   c_std_v.append(sd_c)

        label = f"bias={bias:.2f}"
        plot_band(ax_r2, gen_range, r2_mean_v, r2_std_v, color, label)
        plot_band(ax_c,  gen_range, c_mean_v,  c_std_v,  color, label)

    # Single Poisson R² reference band on R² plot only
    if np.isfinite(pois_r2_ref):
        ax_r2.axhline(pois_r2_ref, color=POISSON_COLOR, lw=1.4, ls="--",
                      label=f"Poisson ($R^2={pois_r2_ref:.2f}$)")
        ax_r2.axhspan(pois_r2_ref - pois_r2_std, pois_r2_ref + pois_r2_std,
                      color=POISSON_COLOR, alpha=0.10)

    ax_r2.set(xlabel="generation", ylabel="$R^2$", ylim=(0, 1))
    ax_r2.legend(fontsize=7)
    fig4.tight_layout()
    fig4.savefig(FIG_DIR / "lewis_r2.pdf", dpi=150)
    plt.close(fig4)

    ax_c.set(xlabel="generation", ylabel="best-fit $c$")
    ax_c.legend(fontsize=7)
    fig5.tight_layout()
    fig5.savefig(FIG_DIR / "lewis_c.pdf", dpi=150)
    plt.close(fig5)

    # ── Fig 6: scatter V vs (n - c) at final generation ───────────────────────
    n_bias = len(all_biases)
    fig6, axes = plt.subplots(1, n_bias, figsize=(4 * n_bias, 4), sharey=False)
    if n_bias == 1:
        axes = [axes]

    for ax, bkey in zip(axes, all_bias_keys(f)):
        bias  = bias_val(bkey)
        gens  = f[bkey].attrs["generations"]
        color = bias_color(bias, all_biases)
        all_x, all_v = [], []
        best_c = 0

        for run in f[bkey].values():
            try:
                data = load_generation(run, gens)
            except KeyError:
                continue
            vols = data["volumes"]; degs = data["degrees"]
            if len(vols) < 20:
                continue
            c, k, r2 = best_lewis_c(vols, degs)
            best_c   = c
            all_x.extend(degs - c); all_v.extend(vols)

        if all_x:
            x = np.array(all_x); v = np.array(all_v)
            ax.scatter(x, v, s=4, alpha=0.25, color=color)
            xl = np.linspace(x.min(), x.max(), 100)
            slope, intercept, *_ = linregress(x, v)
            ax.plot(xl, slope * xl + intercept, "k--", lw=1.5)
        ax.set_xlabel("$n - c$")

    axes[0].set_ylabel("volume $V$")
    fig6.tight_layout()
    fig6.savefig(FIG_DIR / "lewis_scatter.pdf", dpi=150)
    plt.close(fig6)


# ── RQ3: Volume uniformity ─────────────────────────────────────────────────────

def plot_cv(f):
    """
    Figure 7: CV_t vs generation (mean ± std band for biased foam),
              Poisson baseline as gray dashed band (upper bound).
    Figure 8: exponential decay fit panels per bias (mean ± std band on data),
              fit on mean curve.
    """
    all_biases = [bias_val(k) for k in all_bias_keys(f)]

    fig7, ax7  = plt.subplots(figsize=(7, 4))
    fig8, axes8 = plt.subplots(1, len(all_biases),
                                figsize=(4 * len(all_biases), 4), sharey=True)
    if len(all_biases) == 1:
        axes8 = [axes8]

    # Pre-compute a single Poisson CV reference by pooling all bias/run/gen
    # observations and computing a representative steady-state mean ± std.
    # We use all final-generation Poisson CVs (matched cell count ~ MAX_CELLS).
    pois_cv_pool = []
    for bkey in all_bias_keys(f):
        gens      = f[bkey].attrs["generations"]
        pois_runs = get_poisson_runs(f, bkey)
        for run in pois_runs:
            try:
                data = load_generation(run, gens)
            except KeyError:
                continue
            vols = data["volumes"]
            if len(vols) > 1:
                pois_cv_pool.append(vols.std() / vols.mean())
    pois_cv_ref = float(np.mean(pois_cv_pool)) if pois_cv_pool else np.nan
    pois_cv_std = float(np.std(pois_cv_pool))  if pois_cv_pool else np.nan

    for ax8, bkey in zip(axes8, all_bias_keys(f)):
        bias      = bias_val(bkey)
        gens      = f[bkey].attrs["generations"]
        gen_range = np.arange(gens + 1)
        color     = bias_color(bias, all_biases)
        run_list  = list(f[bkey].values())

        # Biased foam: CV per generation, per run
        cv_mean_v, cv_std_v = [], []
        for gen in gen_range:
            cv_run = []
            for run in run_list:
                try:
                    data = load_generation(run, gen)
                except KeyError:
                    continue
                vols = data["volumes"]
                if len(vols) > 1:
                    cv_run.append(vols.std() / vols.mean())
            mu, sd = mean_std(cv_run)
            cv_mean_v.append(mu); cv_std_v.append(sd)

        cv_mean_arr = np.array(cv_mean_v)
        cv_std_arr  = np.array(cv_std_v)

        # fig7: CV over time with ± std band
        plot_band(ax7, gen_range, cv_mean_arr, cv_std_arr, color,
                  label=f"bias={bias:.2f}")

        # fig8: decay fit panel with ±1 std band around mean data
        ax8.plot(gen_range, cv_mean_arr, "o-", color=color, ms=4, label="data (mean)")
        ax8.fill_between(gen_range,
                         cv_mean_arr - cv_std_arr,
                         cv_mean_arr + cv_std_arr,
                         color=color, alpha=BAND_ALPHA)

        valid = np.isfinite(cv_mean_arr) & (gen_range >= 3)
        if valid.sum() >= 4:
            try:
                def exp_decay(t, A, lam, B):
                    return A * np.exp(-lam * t) + B
                lo, hi = [0, 0, 0], [10, 10, 10]
                A0 = float(np.clip(cv_mean_arr[valid][0],  lo[0] + 1e-6, hi[0]))
                B0 = float(np.clip(cv_mean_arr[valid][-1], lo[2] + 1e-6, hi[2]))
                p0 = (A0, 0.1, B0)
                popt, _ = curve_fit(exp_decay, gen_range[valid], cv_mean_arr[valid],
                                    p0=p0, bounds=(lo, hi), maxfev=5000)
                A, lam, B = popt
                t_fit = np.linspace(0, gens, 200)
                ax8.plot(t_fit, exp_decay(t_fit, *popt), "k--", lw=1.5,
                         label=f"fit: $B={B:.3f}$, $\\lambda={lam:.2f}$")
                ax8.axhline(B, color="gray", ls=":", lw=1)
            except RuntimeError:
                pass

        ax8.set_xlabel("generation")
        ax8.legend(fontsize=7)

    axes8[0].set_ylabel("CV$_t$")
    fig8.tight_layout()
    fig8.savefig(FIG_DIR / "cv_decay_fits.pdf", dpi=150)
    plt.close(fig8)

    # Single Poisson CV reference on the main CV-over-time plot
    if np.isfinite(pois_cv_ref):
        ax7.axhline(pois_cv_ref, color=POISSON_COLOR, lw=1.4, ls="--",
                    label=f"Poisson ($\\mathrm{{CV}}={pois_cv_ref:.2f}$)")
        ax7.axhspan(pois_cv_ref - pois_cv_std, pois_cv_ref + pois_cv_std,
                    color=POISSON_COLOR, alpha=0.10)

    ax7.set(xlabel="generation", ylabel="CV$_t = \\sigma/\\mu$")
    ax7.legend(fontsize=8)
    fig7.tight_layout()
    fig7.savefig(FIG_DIR / "cv_over_time.pdf", dpi=150)
    plt.close(fig7)


# ── Sanity checks ──────────────────────────────────────────────────────────────

def plot_sanity(f):
    """
    Figure 9:  Degree distribution at gen 0 (should be near Poisson-Voronoi
               mean ~15.54) and at final generation across bias levels.
    Figure 10: Cell count vs generation for each bias (mean ± std band).
    """
    all_biases = [bias_val(k) for k in all_bias_keys(f)]

    # ── Fig 9: degree distributions ───────────────────────────────────────────
    fig9, (ax_early, ax_late) = plt.subplots(1, 2, figsize=(10, 4))

    for bkey in all_bias_keys(f):
        bias  = bias_val(bkey)
        gens  = f[bkey].attrs["generations"]
        color = bias_color(bias, all_biases)

        for gen, ax, title in [(0, ax_early, "generation 0"),
                                (gens, ax_late, f"generation {gens}")]:
            all_deg = []
            for run in f[bkey].values():
                try:
                    data = load_generation(run, gen)
                except KeyError:
                    continue
                all_deg.extend(data["degrees"].tolist())
            if all_deg:
                bins = np.arange(min(all_deg), max(all_deg) + 2) - 0.5
                ax.hist(all_deg, bins=bins, density=True, histtype="step",
                        color=color, label=f"bias={bias:.2f}")

    for ax in [ax_early, ax_late]:
        ax.axvline(POISSON_MEAN_DEGREE, color="k", ls="--", lw=1,
                   label=f"Poisson mean ({POISSON_MEAN_DEGREE})")
        ax.set(xlabel="degree $n$", ylabel="density")
        ax.legend(fontsize=7)

    fig9.tight_layout()
    fig9.savefig(FIG_DIR / "degree_distributions.pdf", dpi=150)
    plt.close(fig9)

    # ── Fig 10: cell count growth with ±1 std band ────────────────────────────
    fig10, ax10 = plt.subplots(figsize=(7, 4))
    for bkey in all_bias_keys(f):
        bias      = bias_val(bkey)
        gens      = f[bkey].attrs["generations"]
        color     = bias_color(bias, all_biases)
        run_list  = list(f[bkey].values())
        gen_range = np.arange(gens + 1)

        cnt_mean_v, cnt_std_v = [], []
        for gen in gen_range:
            n_list = []
            for run in run_list:
                try:
                    n_list.append(len(load_generation(run, gen)["volumes"]))
                except KeyError:
                    continue
            mu, sd = mean_std(n_list)
            cnt_mean_v.append(mu); cnt_std_v.append(sd)

        plot_band(ax10, gen_range, cnt_mean_v, cnt_std_v, color,
                  label=f"bias={bias:.2f}")

    ax10.set(xlabel="generation", ylabel="mean cell count")
    ax10.legend(fontsize=8)
    fig10.tight_layout()
    fig10.savefig(FIG_DIR / "cell_count_growth.pdf", dpi=150)
    plt.close(fig10)


# ── Graph theory & genealogy ───────────────────────────────────────────────────

def compute_assortativity(degrees, edges):
    """
    Newman degree assortativity coefficient r (Pearson correlation of degrees
    across edges).  r > 0: assortative (high-degree nodes neighbor high-degree).
    r < 0: disassortative (classical 2D Aboav-Weaire limit).
    """
    if len(edges) == 0 or len(degrees) < 4:
        return np.nan
    j = degrees[edges[:, 0]].astype(float)
    k = degrees[edges[:, 1]].astype(float)
    mean_jk  = np.mean(j * k)
    mean_deg = np.mean((j + k) / 2)
    mean_sq  = np.mean((j ** 2 + k ** 2) / 2)
    denom    = mean_sq - mean_deg ** 2
    if denom < 1e-10:
        return np.nan
    return (mean_jk - mean_deg ** 2) / denom


def compute_local_clustering(degrees, edges):
    """
    Local clustering coefficient per node via sparse A³ diagonal.
    C_i = triangles(i) / (k_i*(k_i-1)/2).
    Returns array of length len(degrees).
    """
    N = len(degrees)
    if len(edges) == 0:
        return np.zeros(N)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A    = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A2   = A.dot(A)
    tri  = np.array(A2.multiply(A).sum(axis=1)).flatten() / 2
    deg  = degrees.astype(float)
    poss = deg * (deg - 1) / 2
    return np.where(poss > 0, tri / poss, 0.0)


def track_birth_generations(f, bkey, run_grp, threshold=0.01):
    """
    For each seed in the final generation, find the earliest generation at
    which that seed position appears (within `threshold` distance).
    Surviving seeds keep their exact position; daughters are displaced by
    r = 0.22*V^(1/3) ~ 1.7 units, so threshold=0.01 cleanly separates them.
    Returns birth_gen int array indexed to final-generation cells, or None.
    """
    gens = f[bkey].attrs["generations"]
    seed_arrays = {}
    for g in range(gens + 1):
        try:
            seed_arrays[g] = load_generation(run_grp, g)["seeds"]
        except KeyError:
            pass
    if gens not in seed_arrays:
        return None
    final_seeds = seed_arrays[gens]
    birth_gen   = np.full(len(final_seeds), gens, dtype=int)
    for g in sorted(seed_arrays):
        if g == gens:
            continue
        tree     = cKDTree(seed_arrays[g])
        dists, _ = tree.query(final_seeds, k=1)
        matched  = dists < threshold
        birth_gen[matched] = np.minimum(birth_gen[matched], g)
    return birth_gen


def plot_network_stats(f):
    """
    Figure: Newman assortativity r and mean clustering coefficient C̄
    vs generation for each bias level, with Poisson reference bands.
    Saved as figures/network_stats.pdf
    """
    all_biases = [bias_val(k) for k in all_bias_keys(f)]
    bkeys      = list(all_bias_keys(f))

    # Poisson references (pooled across bias levels at final generation)
    pois_r_pool, pois_c_pool = [], []
    for bkey in bkeys:
        gens = f[bkey].attrs["generations"]
        for run in get_poisson_runs(f, bkey):
            try:
                data = load_generation(run, gens)
            except KeyError:
                continue
            r = compute_assortativity(data["degrees"], data["edges"])
            C = compute_local_clustering(data["degrees"], data["edges"])
            if np.isfinite(r):  pois_r_pool.append(r)
            if len(C) > 0:      pois_c_pool.append(float(C.mean()))
    pois_r_ref, pois_r_std = mean_std(pois_r_pool)
    pois_c_ref, pois_c_std = mean_std(pois_c_pool)

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(12, 4))

    for bkey in tqdm(bkeys, desc="Network stats"):
        bias      = bias_val(bkey)
        gens      = f[bkey].attrs["generations"]
        color     = bias_color(bias, all_biases)
        run_list  = list(f[bkey].values())
        gen_range = np.arange(gens + 1)

        r_mean_v, r_std_v = [], []
        c_mean_v, c_std_v = [], []

        for gen in gen_range:
            r_run, c_run = [], []
            for run in run_list:
                try:
                    data = load_generation(run, gen)
                except KeyError:
                    continue
                if len(data["degrees"]) < 10:
                    continue
                r = compute_assortativity(data["degrees"], data["edges"])
                C = compute_local_clustering(data["degrees"], data["edges"])
                if np.isfinite(r): r_run.append(r)
                if len(C) > 0:     c_run.append(float(C.mean()))
            mu_r, sd_r = mean_std(r_run)
            mu_c, sd_c = mean_std(c_run)
            r_mean_v.append(mu_r); r_std_v.append(sd_r)
            c_mean_v.append(mu_c); c_std_v.append(sd_c)

        label = f"bias={bias:.2f}"
        plot_band(ax_r, gen_range, r_mean_v, r_std_v, color, label)
        plot_band(ax_c, gen_range, c_mean_v, c_std_v, color, label)

    for ax, ref, std, lbl in [
        (ax_r, pois_r_ref, pois_r_std, f"Poisson ($r={pois_r_ref:.3f}$)"),
        (ax_c, pois_c_ref, pois_c_std, f"Poisson ($\\bar{{C}}={pois_c_ref:.3f}$)"),
    ]:
        if np.isfinite(ref):
            ax.axhline(ref, color=POISSON_COLOR, lw=1.4, ls="--", label=lbl)
            ax.axhspan(ref - std, ref + std, color=POISSON_COLOR, alpha=0.10)

    ax_r.axhline(0, color="k", lw=0.6, ls=":")
    ax_r.set(xlabel="generation", ylabel="assortativity $r$")
    ax_c.set(xlabel="generation", ylabel="mean clustering $\\bar{C}$")
    ax_r.legend(fontsize=7); ax_c.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "network_stats.pdf", dpi=150)
    plt.close(fig)


def plot_clustering_by_degree(f):
    """
    Figure: local clustering coefficient C(n) vs degree n at the final
    generation for each bias level.
    Saved as figures/clustering_by_degree.pdf
    """
    all_biases = [bias_val(k) for k in all_bias_keys(f)]
    bkeys      = list(all_bias_keys(f))

    fig, ax = plt.subplots(figsize=(7, 4))

    for bkey in bkeys:
        bias  = bias_val(bkey)
        gens  = f[bkey].attrs["generations"]
        color = bias_color(bias, all_biases)
        per_n = defaultdict(list)

        for run in f[bkey].values():
            try:
                data = load_generation(run, gens)
            except KeyError:
                continue
            C    = compute_local_clustering(data["degrees"], data["edges"])
            for n, c_val in zip(data["degrees"], C):
                if 4 <= n <= 30:
                    per_n[n].append(float(c_val))

        if not per_n:
            continue
        ns     = np.array(sorted(per_n.keys()))
        c_mean = np.array([np.mean(per_n[n]) for n in ns])
        c_std  = np.array([np.std(per_n[n])  for n in ns])
        plot_band(ax, ns, c_mean, c_std, color, f"bias={bias:.2f}", lw=1.5)

    ax.set(xlabel="degree $n$", ylabel="clustering $C(n)$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clustering_by_degree.pdf", dpi=150)
    plt.close(fig)


def plot_genealogical(f):
    """
    Genealogical clustering analysis using seed position tracking.

    Three panels:
      Left:   mean birth generation vs degree n  (do high-n cells live longer?)
      Centre: mean neighbour birth_gen vs own birth_gen  (do old cells neighbour old?)
      Right:  genealogical edge correlation r per bias level  (scalar summary)

    Saved as figures/genealogical_clustering.pdf
    """
    all_biases = [bias_val(k) for k in all_bias_keys(f)]
    bkeys      = list(all_bias_keys(f))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_bd, ax_nb, ax_ga = axes

    for bkey in tqdm(bkeys, desc="Genealogical clustering"):
        bias  = bias_val(bkey)
        gens  = f[bkey].attrs["generations"]
        color = bias_color(bias, all_biases)

        birth_by_deg = defaultdict(list)
        gen_assort   = []
        nb_rows      = []        # (own_bg, mean_nb_bg)

        for run in f[bkey].values():
            bg = track_birth_generations(f, bkey, run)
            if bg is None:
                continue
            try:
                data = load_generation(run, gens)
            except KeyError:
                continue
            degs  = data["degrees"]
            edges = data["edges"]

            # Birth gen bucketed by degree
            for n, b in zip(degs, bg):
                if 4 <= n <= 30:
                    birth_by_deg[n].append(int(b))

            # Genealogical assortativity: Pearson r of birth_gen across edges
            if len(edges) > 0:
                bi = bg[edges[:, 0]].astype(float)
                bj = bg[edges[:, 1]].astype(float)
                if bi.std() > 0 and bj.std() > 0:
                    r_val, _ = pearsonr(bi, bj)
                    gen_assort.append(float(r_val))

            # Mean neighbour birth_gen vs own birth_gen
            if len(edges) > 0:
                adj_bg = np.zeros(len(bg), dtype=float)
                np.add.at(adj_bg, edges[:, 0], bg[edges[:, 1]])
                np.add.at(adj_bg, edges[:, 1], bg[edges[:, 0]])
                deg_f      = degs.astype(float)
                mean_nb_bg = np.where(deg_f > 0, adj_bg / deg_f, np.nan)
                for own, nb in zip(bg, mean_nb_bg):
                    if np.isfinite(nb):
                        nb_rows.append((int(own), float(nb)))

        # Panel 1: mean birth_gen vs degree
        if birth_by_deg:
            ns      = np.array(sorted(birth_by_deg.keys()))
            bg_mean = np.array([np.mean(birth_by_deg[n]) for n in ns])
            bg_std  = np.array([np.std(birth_by_deg[n])  for n in ns])
            plot_band(ax_bd, ns, bg_mean, bg_std, color,
                      f"bias={bias:.2f}", lw=1.5)

        # Panel 2: bin by own birth_gen, plot mean neighbour birth_gen
        if nb_rows:
            arr      = np.array(nb_rows)
            own_gens = np.unique(arr[:, 0].astype(int))
            nb_means = np.array([arr[arr[:, 0].astype(int) == g, 1].mean()
                                 for g in own_gens])
            ax_nb.plot(own_gens, nb_means, "o-", color=color,
                       ms=3, lw=1.2, alpha=0.8, label=f"bias={bias:.2f}")

        # Panel 3: scalar genealogical assortativity
        if gen_assort:
            mu, sd = mean_std(gen_assort)
            ax_ga.errorbar(bias, mu, yerr=sd, fmt="o", color=color,
                           capsize=4, ms=7)

    # Diagonal reference on panel 2
    all_gens = range(max(f[k].attrs["generations"] for k in bkeys) + 1)
    ax_nb.plot([0, max(all_gens)], [0, max(all_gens)],
               "k--", lw=0.8, alpha=0.4, label="$y = x$")

    ax_bd.set(xlabel="degree $n$", ylabel="mean birth generation")
    ax_nb.set(xlabel="own birth generation",
              ylabel="mean neighbour birth generation")
    ax_ga.set(xlabel="bias level",
              ylabel="edge birth-gen correlation $r$")
    ax_ga.axhline(0, color="k", lw=0.6, ls=":")

    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "genealogical_clustering.pdf", dpi=150)
    plt.close(fig)


# ── Diagnostics ────────────────────────────────────────────────────────────────

def debug_data(f):
    """Print key stats for first run only at first, middle, and last generation."""
    print("\n── Data diagnostics ──────────────────────────────────────────────")
    for bkey in all_bias_keys(f):
        gens = f[bkey].attrs["generations"]
        print(f"\n  {bkey}  (generations={gens})")
        # Only print the first run to keep output readable
        run_items = list(f[bkey].items())
        if not run_items:
            continue
        run_name, run = run_items[0]
        print(f"    {run_name}  (first run only)")
        for gen in [0, gens // 2, gens]:
            try:
                d    = load_generation(run, gen)
                vols = d["volumes"]; degs = d["degrees"]; edges = d["edges"]
                cv   = vols.std() / vols.mean() if vols.mean() > 0 else float("nan")
                print(f"      gen {gen:02d}: cells={len(vols)}, "
                      f"mean_vol={vols.mean():.1f}, cv={cv:.3f}, "
                      f"mean_deg={degs.mean():.2f}, edges={len(edges)}")
            except KeyError as e:
                print(f"      gen {gen:02d}: MISSING — {e}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    with h5py.File(DATA_FILE, "r") as f:
        print(f"Loaded {DATA_FILE}  |  bias levels: {all_bias_keys(f)}")
        debug_data(f)

        print("\n── RQ1: Aboav-Weaire ─────────────────────────────────────────")
        plot_aboav_weaire(f)

        print("\n── RQ2: Lewis's Law ──────────────────────────────────────────")
        plot_lewis(f)

        print("\n── RQ3: Volume uniformity ────────────────────────────────────")
        plot_cv(f)

        print("\n── Sanity checks ─────────────────────────────────────────────")
        plot_sanity(f)

        print("\n── Graph theory ──────────────────────────────────────────────")
        plot_network_stats(f)
        plot_clustering_by_degree(f)

        print("\n── Genealogical clustering ───────────────────────────────────")
        plot_genealogical(f)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
