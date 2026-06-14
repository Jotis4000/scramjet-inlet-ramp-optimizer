import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

'''
atmosphere.py

Obtains the relevant atmospheric parameters for a given altitude based on the International Standard Atmosphere Model.

Author: Panagiotis Sachinis
Year: 2026
'''

g0 = 9.80665
R = 287
gamAir = 1.4

def getAtmosphere(h):
    """
    Calculates atmospheric properties using the 1976 US Standard Atmosphere model.
    Extends isothermal properties up to 100 km for LEO reentry simulations.
    
    Inputs:
    h       : Geometric altitude [m] (can be a scalar or numpy array)
    g0      : Standard gravity [m/s^2]
    R       : Specific gas constant for air [J/(kg*K)]
    gamAir  : Ratio of specific heats (gamma)
    
    Outputs:
    T, P, rho, a, mu, nu (All as numpy arrays or scalars depending on input)
    """
    # Ensure input is a numpy array for vectorized operations
    h_arr = np.atleast_1d(h)
    
    # Constants
    R_earth = 6356766.0  # Earth radius in meters
    beta = 1.458e-6      # Sutherland's constant 1 [kg/(m*s*K^0.5)]
    S = 110.4            # Sutherland's constant 2 [K]
    
    # 1. Convert Geometric Altitude (h) to Geopotential Altitude (H)
    H = (R_earth * h_arr) / (R_earth + h_arr)
    
    # 2. Define the ISA Layers (Base Altitude, Base Temp, Base Press, Lapse Rate)
    # Layers: Troposphere, Tropopause, Stratosphere 1, Stratosphere 2, Stratopause, Mesosphere 1, Mesosphere 2, Mesopause/Thermosphere
    Hb = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
    Tb = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    Pb = np.array([101325.0, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.9564, 0.3734])
    Lb = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0])
    
    # 3. Create boolean masks to identify which layer each altitude falls into
    # The last condition catches everything above 84.852 km up to 100+ km
    conds = [
        (H >= Hb[0]) & (H < Hb[1]),
        (H >= Hb[1]) & (H < Hb[2]),
        (H >= Hb[2]) & (H < Hb[3]),
        (H >= Hb[3]) & (H < Hb[4]),
        (H >= Hb[4]) & (H < Hb[5]),
        (H >= Hb[5]) & (H < Hb[6]),
        (H >= Hb[6]) & (H < Hb[7]),
        (H >= Hb[7])
    ]
    
    # 4. Calculate Temperature (T) and Pressure (P)
    # We create a function to apply the correct math based on whether Lapse Rate is 0 (isothermal) or not (gradient)
    T = np.zeros_like(H)
    P = np.zeros_like(H)
    
    for i in range(len(Hb)):
        mask = conds[i]
        if not np.any(mask):
            continue
            
        if Lb[i] == 0.0: # Isothermal Layer
            T[mask] = Tb[i]
            P[mask] = Pb[i] * np.exp(-g0 * (H[mask] - Hb[i]) / (R * Tb[i]))
        else:            # Gradient Layer
            T[mask] = Tb[i] + Lb[i] * (H[mask] - Hb[i])
            P[mask] = Pb[i] * (T[mask] / Tb[i]) ** (-g0 / (Lb[i] * R))
            
    # 5. Derived Properties
    rho = P / (R * T)                     # Density [kg/m^3]
    a = np.sqrt(gamAir * R * T)           # Speed of Sound [m/s]
    mu = (beta * T**1.5) / (T + S)        # Dynamic Viscosity [kg/(m*s)]
    nu = mu / rho                         # Kinematic Viscosity [m^2/s]
    
    # If a scalar was passed in, return scalars. Otherwise, return arrays.
    if np.isscalar(h):
        return [rho[0], T[0], P[0], a[0], nu[0], mu[0]]
    
    return [rho, T, P, a, nu, mu]