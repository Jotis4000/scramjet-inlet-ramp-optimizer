import os
import json
import numpy as np
from createGas import realGas
import atmosphere as atm
from optimizer import optimize_scramjet_inlet
import plotting

'''
Scramjet Inlet Ramp Optimizer

Author: Panagiotis Sachinis
Year: 2026

Sweeps a multi-ramp scramjet inlet across freestream Mach number, optimizing the
ramp geometry at each Mach for maximum total-pressure recovery subject to fixed
combustor-entry constraints.
'''

#########################################
# SIMULATION PARAMETERS
#########################################

H = 25000                                                               # Altitude [m]
RHO_INF, T_INF, P_INF, A_INF, NU_INF, MU_INF = atm.getAtmosphere(H)     # Atmospheric Parameters from ISA

MECHANISM = "airNASA9.yaml"                                             # airNASA9 used for larger temperature ranges
COMPOSITION = "O2:0.21, N2:0.79"                                        # Standard air composition

N_RAMPS = 3                                                             # No. of ramps in geometry
MACH_LIST = list(range(5, 11))                                          # M = 5, 6, 7, 8, 9, 10

M_EXIT_RANGE = (1.5, 3.5)                                               # Combustor Mach range [-]
T_EXIT_RANGE = (1000.0, 1600.0)                                         # Auto-ignition Temperature Floor [K]
RAMP_ANGLE_RANGE = (0.5, 18.0)                                          # Per-Ramp Deflection Constraints [deg]

DATA_FILE = "data/results.json"

def to_serializable(case):
    """Convert one optimizer result (with numpy types) into plain JSON types."""
    def listify(value):
        return None if value is None else np.asarray(value).tolist()
    return {
        'M_inf': float(case['M_inf']),
        'feasible': bool(case['feasible']),
        'thetas_deg': listify(case['thetas_deg']),
        'betas_deg': listify(case['betas_deg']),
        'recovery': None if case['recovery'] is None else float(case['recovery']),
        'M_exit': None if case['M_exit'] is None else float(case['M_exit']),
        'T_exit': None if case['T_exit'] is None else float(case['T_exit']),
        'P_exit': None if case['P_exit'] is None else float(case['P_exit']),
    }


def main():
    gas = realGas(MECHANISM, COMPOSITION)

    cases = []
    for M_inf in MACH_LIST:
        print(f"\n===== M_inf = {M_inf} =====")
        out = optimize_scramjet_inlet(
            M_inf, T_INF, P_INF, gas, n_ramps=N_RAMPS,
            M_exit_range=M_EXIT_RANGE, T_exit_range=T_EXIT_RANGE,
            ramp_angle_range=RAMP_ANGLE_RANGE, verbose=True)
        out['M_inf'] = M_inf
        cases.append(out)

    ## Assemble results
    results = {
        'altitude': H,
        'n_ramps': N_RAMPS,
        'constraints': {
            'M_exit_range': list(M_EXIT_RANGE),
            'T_exit_range': list(T_EXIT_RANGE),
            'ramp_angle_range': list(RAMP_ANGLE_RANGE),
        },
        'cases': [to_serializable(c) for c in cases],
    }
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as handle:
        json.dump(results, handle, indent=2)

    ## Print summary
    print("\n================ SUMMARY ================")
    for c in results['cases']:
        if c['thetas_deg'] is None:
            print(f"M={c['M_inf']:.0f} | no solution")
            continue
        angles = ", ".join(f"{a:5.2f}" for a in c['thetas_deg'])
        tag = "feasible" if c['feasible'] else "INFEASIBLE"
        print(f"M={c['M_inf']:4.0f} | ramps=[{angles}] deg | pi_t={c['recovery']:.4f} "
              f"| M_exit={c['M_exit']:.3f} | T_exit={c['T_exit']:6.1f} K | {tag}")
    print(f"\nSaved data to {DATA_FILE}")

    ## Plotting
    plotting.plot_geometries(results)
    plotting.plot_performance(results)
    plotting.show()


if __name__ == "__main__":
    main()