import cantera as ct
import numpy as np
from scipy.optimize import root_scalar, minimize_scalar
from scipy.optimize import brentq

'''
solveShock.py

Contains all functions necessary to solve for each inlet geometry.

Author: Panagiotis Sachinis
Year: 2026
'''

def stagnation_pressure(gas_solver, V, tol=1e-6, max_steps=30):
    """Real-gas stagnation pressure of the gas's CURRENT static state.

    Isentropically decelerates to rest (constant entropy, total enthalpy
    h0 = h + V^2/2) and returns the pressure where enthalpy == h0.

    Uses Newton iteration with the exact isentropic derivative
        dh/dp|_s = 1/rho
    seeded by the perfect-gas closed form. This was a headache
    """
    g = gas_solver.gas
    T_s, P_s, X_s = g.TPX                      # save static state
    s0 = g.entropy_mass
    h0 = g.enthalpy_mass + 0.5 * V**2          # conserved total enthalpy
    gamma = g.cp_mass / g.cv_mass
    M = V / g.sound_speed

    # Perfect-gas closed form as the initial guess (no equilibration, ~10-30% off).
    p = P_s * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))

    enthalpy_scale = max(0.5 * V**2, 1.0)      # positive convergence scale
    for _ in range(max_steps):
        g.SP = s0, p
        g.equilibrate('SP')
        residual = g.enthalpy_mass - h0
        if abs(residual) <= tol * enthalpy_scale:
            break
        # Newton step: p_{n+1} = p_n - f/f' with f' = dh/dp|_s = 1/rho
        p = p - residual * g.density
        p = min(max(p, P_s), P_s * 1e7)        # pt >= p_static; cap for safety
    else:
        g.TPX = T_s, P_s, X_s
        raise RuntimeError("Newton stagnation-pressure iteration did not converge.")

    g.TPX = T_s, P_s, X_s                      # restore static state
    return p

def calc_post_shock(beta, V1, state, gas, X_upstream):
    """Calculates the post shock state of the gas.
    """

    u1 = V1 * np.sin(beta)
    w1 = V1 * np.cos(beta)
    mass_flux = state['RHO'] * u1

    Mn1 = u1 / state['A']
    g = state['GAMMA']
    ratio = ((g + 1) * Mn1**2) / ((g - 1) * Mn1**2 + 2) if Mn1 > 1.0 else 1.0
    rho2 = state['RHO'] * ratio

    tolerance, max_iter = 1e-4, 200
    for i in range(max_iter):
        rho2 = max(rho2, state['RHO'])          # enforce compression (rho2 >= rho1)
        u2 = mass_flux / rho2                     # => u2 <= u1, so h2 >= h1 always
        P2 = state['P'] + mass_flux * (u1 - u2)
        h2 = state['H'] + 0.5 * (u1**2 - u2**2)

        gas.gas.X = X_upstream
        gas.gas.HP = h2, P2
        if gas.get_state()['T'] >= 300:
            gas.gas.equilibrate('HP')

        rho2_new = gas.gas.density
        if abs(rho2_new - rho2) / rho2 < tolerance:
            rho2 = rho2_new
            break
        rho2 = 0.5 * rho2 + 0.5 * rho2_new
    else:
        raise RuntimeError("Inner loop (thermodynamics) failed to converge.")

    theta_calc = beta - np.arctan(u2 / w1)
    return theta_calc, gas.get_state(), u2, w1


def solve_oblique_shock(M1, T1, P1, theta, gas_solver):
    """
    Finds the correct shock angle (beta) for a target ramp angle (theta).
    """

    # Set upstream state
    gas_solver.gas.TP = T1, P1
    X1 = gas_solver.gas.X.copy()      # composition entering THIS shock
    state1 = gas_solver.get_state()
    V1 = M1 * state1['A']

    # Define the objective function for root-finding
    def objective(beta_guess):
        theta_calc, _, _, _ = calc_post_shock(beta_guess, V1, state1, gas_solver, X1)
        return theta_calc - theta

    beta_min = np.arcsin(1.0 / M1)
    beta_max = np.pi / 2.0

    deflection = lambda b: calc_post_shock(b, V1, state1, gas_solver, X1)[0]
    peak = minimize_scalar(lambda b: -deflection(b),
                           bounds=(beta_min + 1e-4, beta_max - 1e-4),
                           method='bounded')
    beta_at_theta_max = peak.x
    theta_max = -peak.fun

    # Can you even turn that hard without detaching?
    if theta > theta_max:
        raise ValueError(
            f"Shock Detached! A ramp angle of {theta}deg exceeds the maximum "
            f"deflection theta_max = {np.degrees(theta_max):.2f}deg for Mach {M1}.")

    res = root_scalar(objective,
                      bracket=[beta_min + 1e-4, beta_at_theta_max],
                      method='brentq')
    beta_sol = res.root

    _, final_state2, u2, w1 = calc_post_shock(beta_sol, V1, state1, gas_solver, X1)

    V2 = np.sqrt(u2**2 + w1**2)
    M2 = V2 / final_state2['A']

    gamma_eff = final_state2['GAMMA']
    Pt_ratio = (final_state2['P'] / P1) * (
        (1 + 0.5 * (gamma_eff - 1) * M2**2)
        / (1 + 0.5 * (state1['GAMMA'] - 1) * M1**2)
    ) ** (gamma_eff / (gamma_eff - 1))

    return {
        'beta_deg': np.degrees(beta_sol),
        'M2': M2,
        'T2': final_state2['T'],
        'P2': final_state2['P'],
        'Pt_ratio': Pt_ratio,  # Approximation of total pressure recovery
        'gamma2': final_state2['GAMMA'],
    }

def solve_inlet(M_INF, T_INF, P_INF, thetas, gas_solver):
    """Uses the above functions to solve for the full inlet.
    Each successive ramp takes output state of previous as input (including chemical reactions)."""
    
    gas_solver.gas.TP = T_INF, P_INF            # fresh design air (see below)
    a_inf = gas_solver.get_state()['A']
    pt_inf = stagnation_pressure(gas_solver, M_INF * a_inf)

    M1, T1, P1 = M_INF, T_INF, P_INF
    betas = np.zeros(len(thetas))
    res = None
    for c, theta in enumerate(thetas):
        res = solve_oblique_shock(M1, T1, P1, theta, gas_solver)
        M1, T1, P1 = res['M2'], res['T2'], res['P2']
        betas[c] = res['beta_deg']

    if res is None:
        return {'betas': betas, 'M_EXIT': M_INF, 'T_EXIT': T_INF,
                'P_EXIT': P_INF, 'PT_RATIO': 1.0}

    a_exit = gas_solver.get_state()['A']          # gas is left at the exit static state
    pt_exit = stagnation_pressure(gas_solver, res['M2'] * a_exit)

    return {'betas': betas, 'M_EXIT': res['M2'], 'T_EXIT': res['T2'],
            'P_EXIT': res['P2'], 'PT_RATIO': pt_exit / pt_inf}