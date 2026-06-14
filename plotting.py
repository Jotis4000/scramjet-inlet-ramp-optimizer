import json
import numpy as np
import matplotlib.pyplot as plt

"""
plotting.py
===========

Plotting routine. Generative help on this one.
"""

DATA_FILE = "data/results.json"

def load_results(path=DATA_FILE):
    with open(path) as handle:
        return json.load(handle)


def show():
    plt.show()


#########################################
# GEOMETRY CONSTRUCTION
#########################################

def build_geometry(thetas_deg, betas_deg, ramp_length=1.0, shock_length=None):
    """From per-ramp deflection angles and shock wave angles, build the
    compression-surface corner points and the shock ray endpoints."""
    thetas_deg = np.asarray(thetas_deg, dtype=float)
    betas_deg = np.asarray(betas_deg, dtype=float)
    phi = np.cumsum(thetas_deg)                      # wall angle of each ramp [deg]

    corners = [np.array([0.0, 0.0])]
    for i in range(len(thetas_deg)):
        ang = np.radians(phi[i])
        corners.append(corners[-1] + ramp_length * np.array([np.cos(ang), np.sin(ang)]))
    corners = np.array(corners)

    if shock_length is None:
        shock_length = 1.4 * corners[-1, 0]

    phi_upstream = np.concatenate(([0.0], phi[:-1]))  # upstream flow angle at corner
    shocks = []
    for i in range(len(thetas_deg)):
        origin = corners[i]
        ang = np.radians(phi_upstream[i] + betas_deg[i])
        tip = origin + shock_length * np.array([np.cos(ang), np.sin(ang)])
        shocks.append((origin, tip))
    return corners, shocks


def global_limits(cases, shock_length, pad=0.08):
    """Common axis window across all cases so every subplot is the same size."""
    xs, ys = [0.0], [0.0]
    for c in cases:
        if c['thetas_deg'] is None:
            continue
        corners, shocks = build_geometry(c['thetas_deg'], c['betas_deg'],
                                         shock_length=shock_length)
        xs += list(corners[:, 0]); ys += list(corners[:, 1])
        for origin, tip in shocks:
            xs += [origin[0], tip[0]]; ys += [origin[1], tip[1]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = x1 - x0, y1 - y0
    span = max(dx, dy)
    return (x0 - pad * span, x0 + span + pad * span,
            y0 - pad * span - 0.15 * span, y0 + span + pad * span)


#########################################
# PLOTTING
#########################################

def plot_geometries(results, filename="figs/inlet_geometries.png"):
    """One equally-sized panel per Mach: ramp surface and resulting shocks."""
    cases = results['cases']
    n = len(cases)
    cols = 3
    rows = int(np.ceil(n / cols))

    # common scale so all subplots match in size
    surface_lengths = [build_geometry(c['thetas_deg'], c['betas_deg'])[0][-1, 0]
                       for c in cases if c['thetas_deg'] is not None]
    shock_length = 1.4 * (max(surface_lengths) if surface_lengths else 1.0)
    xlim_lo, xlim_hi, ylim_lo, ylim_hi = global_limits(cases, shock_length)

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, cases):
        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        if c['thetas_deg'] is None:
            ax.set_title(f"M={c['M_inf']:.0f}  [no solution]", fontsize=9)
            continue

        corners, shocks = build_geometry(c['thetas_deg'], c['betas_deg'],
                                         shock_length=shock_length)
        total_len = corners[-1, 0]

        # ramp body (filled) + surface line
        body_x = np.concatenate([corners[:, 0], [corners[-1, 0], 0.0]])
        body_y = np.concatenate([corners[:, 1],
                                 [ylim_lo, ylim_lo]])
        ax.fill(body_x, body_y, color='0.82', zorder=1)
        ax.plot(corners[:, 0], corners[:, 1], '-', color='0.25', lw=2.2,
                zorder=3, label='ramps')

        # shock rays
        for k, (origin, tip) in enumerate(shocks):
            ax.plot([origin[0], tip[0]], [origin[1], tip[1]], '--',
                    color='tab:red', lw=1.3, zorder=2,
                    label='shocks' if k == 0 else None)

        feas = "feasible" if c['feasible'] else "INFEASIBLE"
        angles = "/".join(f"{a:.1f}" for a in c['thetas_deg'])
        ax.set_title(f"M={c['M_inf']:.0f}  [{feas}]\n"
                     f"ramps {angles}°,  $\\pi_t$={c['recovery']:.3f}",
                     fontsize=9)
        if c is cases[0]:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    for ax in axes[n:]:
        ax.set_axis_off()

    fig.suptitle("Optimized inlet geometry and shock structure", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(filename, dpi=130)
    return fig


def plot_performance(results, filename="figs/inlet_performance.png"):
    """Recovery, exit Mach, and exit temperature vs freestream Mach."""
    cases = results['cases']
    M_exit_range = results['constraints']['M_exit_range']
    T_exit_range = results['constraints']['T_exit_range']

    mach = np.array([c['M_inf'] for c in cases], dtype=float)
    feasible = np.array([c['feasible'] for c in cases])
    recovery = np.array([np.nan if c['recovery'] is None else c['recovery']
                         for c in cases], dtype=float)
    m_exit = np.array([np.nan if c['M_exit'] is None else c['M_exit']
                       for c in cases], dtype=float)
    t_exit = np.array([np.nan if c['T_exit'] is None else c['T_exit']
                       for c in cases], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def scatter_feasible(ax, y):
        ax.plot(mach, y, '-', color='0.7', lw=1, zorder=1)
        ax.scatter(mach[feasible], y[feasible], c='tab:green', s=45,
                   zorder=3, label='feasible')
        if np.any(~feasible):
            ax.scatter(mach[~feasible], y[~feasible], facecolors='none',
                       edgecolors='tab:red', s=55, zorder=3, label='infeasible')

    scatter_feasible(axes[0], recovery)
    axes[0].set_ylabel(r'total-pressure recovery  $\pi_t = p_{t,e}/p_{t,\infty}$')
    axes[0].set_title('Pressure recovery')

    scatter_feasible(axes[1], m_exit)
    axes[1].axhspan(M_exit_range[0], M_exit_range[1], color='tab:green',
                    alpha=0.08, label='allowed band')
    axes[1].set_ylabel('combustor-entry Mach  $M_{exit}$')
    axes[1].set_title('Exit Mach number')

    scatter_feasible(axes[2], t_exit)
    axes[2].axhspan(T_exit_range[0], T_exit_range[1], color='tab:green',
                    alpha=0.08, label='allowed band')
    axes[2].set_ylabel('exit static temperature  $T_{exit}$ [K]')
    axes[2].set_title('Exit temperature')

    for ax in axes:
        ax.set_xlabel(r'freestream Mach  $M_\infty$')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    return fig


if __name__ == "__main__":
    results = load_results()
    plot_geometries(results)
    plot_performance(results)
    show()