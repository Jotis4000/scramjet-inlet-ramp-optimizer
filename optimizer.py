import numpy as np
from scipy.optimize import minimize
from solveShock import solve_inlet

'''
optimizer.py

Optimize the ramp angles of a multi-ramp scramjet inlet at a single freestream
Mach number. Maximizes total-pressure recovery subject to constraints on the
combustor-entry Mach number and static temperature (auto-ignition).

Method: SLSQP (sequential least squares) -- it handles the per-ramp angle bounds
and the Mach/temperature inequality constraints directly in one call. A few
starting geometries are tried so we don't settle in a poor local optimum, and a
small cache keeps each geometry from being re-solved for every constraint.

Author: Panagiotis Sachinis
Year: 2026
'''

def optimize_scramjet_inlet(M_inf, T_inf, P_inf, gas, n_ramps=3,M_exit_range=(2.5, 4.5),T_exit_range=(1000.0, 1600.0),ramp_angle_range=(0.5, 18.0),verbose=True):
    
    """
    Parameters
    ----------
    M_inf, T_inf, P_inf : freestream Mach number, static temperature [K],
                          static pressure [Pa]
    gas                 : a realGas instance (passed through to solve_inlet)
    n_ramps             : number of ramps (any number)
    M_exit_range        : (min, max) allowed combustor-entry Mach number
    T_exit_range        : (min, max) allowed static temperature [K]
                          (min is the fuel auto-ignition floor)
    ramp_angle_range    : (min, max) deflection per ramp [degrees]
    verbose             : print progress while optimizing

    Returns
    -------
    dict with: feasible, thetas_deg, betas_deg, recovery (pt_exit/pt_inf),
               M_exit, T_exit, P_exit
    """

    M_min, M_max = M_exit_range
    T_min, T_max = T_exit_range
    angle_min, angle_max = ramp_angle_range

    cache = {}

    def run_inlet(thetas_deg):
        """solve_inlet for a geometry; returns the result dict, or None if the
        geometry is detached / non-convergent. Cached so the objective and all
        constraints at one geometry trigger only a single solve."""
        key = tuple(np.round(thetas_deg, 6)) # headaaaaachesss
        if key not in cache:
            try:
                cache[key] = solve_inlet(M_inf, T_inf, P_inf,
                                         np.radians(thetas_deg), gas)
            except (ValueError, RuntimeError):
                cache[key] = None
        return cache[key]

    # Objective for SLSQP is to minimize negative of pressure recovery
    def neg_recovery(thetas_deg):
        result = run_inlet(thetas_deg)
        return 1.0 if result is None else -result['PT_RATIO']

    # All constraints written as g(x) >= 0, normalized to O(1)
    def constraints(thetas_deg):
        result = run_inlet(thetas_deg)
        if result is None:
            return np.full(4, -1.0)                 # infeasible geometry, violate all
        return np.array([
            result['M_EXIT'] - M_min,               # M_exit >= M_min
            M_max - result['M_EXIT'],               # M_exit <= M_max
            (result['T_EXIT'] - T_min) / T_min,     # T_exit >= T_min (ignition)
            (T_max - result['T_EXIT']) / T_min,     # T_exit <= T_max
        ])

    bounds = [(angle_min, angle_max)] * n_ramps
    constraint_spec = {'type': 'ineq', 'fun': constraints}

    # Seed geometries to start with - kinda chosen randomly
    mid = 0.5 * (angle_min + min(angle_max, 12.0))
    seeds = [
        np.linspace(0.6 * mid, 1.4 * mid, n_ramps),       # increasing
        np.full(n_ramps, mid),                            # equal
        np.linspace(0.7 * angle_max, angle_max, n_ramps), # steep
    ]
    seeds = [np.clip(seed, angle_min, angle_max) for seed in seeds]

    def callback(thetas_deg):
        result = run_inlet(thetas_deg)
        if result is not None:
            print(f"      pi_t={result['PT_RATIO']:.4f}  "
                  f"M_exit={result['M_EXIT']:.3f}  "
                  f"T_exit={result['T_EXIT']:7.1f} K  "
                  f"angles={np.round(thetas_deg, 2)}")

    def is_feasible(result):
        return (M_min - 1e-3 <= result['M_EXIT'] <= M_max + 1e-3 and
                T_min * 0.999 <= result['T_EXIT'] <= T_max * 1.001)

    # SLSQP setup (help)
    if verbose:
        print(f"Optimizing {n_ramps}-ramp inlet at M_inf = {M_inf:.2f} "
              f"(T_inf={T_inf:.1f} K, P_inf={P_inf:.0f} Pa)")

    best = None    # (feasible, recovery, thetas_deg, result)
    for seed in seeds:
        if verbose:
            print(f"  seed = {np.round(seed, 2)} deg")
        solution = minimize(neg_recovery, seed, method='SLSQP',
                            bounds=bounds, constraints=constraint_spec,
                            callback=callback if verbose else None,
                            options={'maxiter': 100, 'ftol': 1e-6, 'eps': 1e-2})

        thetas_deg = np.clip(solution.x, angle_min, angle_max)
        result = run_inlet(thetas_deg)
        if result is None:
            continue

        candidate = (is_feasible(result), result['PT_RATIO'], thetas_deg, result)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        if verbose:
            print("no usable geometry found")
        return {'feasible': False, 'thetas_deg': None, 'betas_deg': None,
                'recovery': 0.0, 'M_exit': None, 'T_exit': None, 'P_exit': None}

    feasible, recovery, thetas_deg, result = best
    if verbose:
        tag = "feasible" if feasible else "INFEASIBLE (best effort)"
        print(f"  -> {tag}: pi_t={recovery:.4f}, angles={np.round(thetas_deg, 2)} deg")

    return {
        'feasible': feasible,
        'thetas_deg': thetas_deg,
        'betas_deg': np.asarray(result['betas']),
        'recovery': recovery,
        'M_exit': result['M_EXIT'],
        'T_exit': result['T_EXIT'],
        'P_exit': result['P_EXIT'],
    }