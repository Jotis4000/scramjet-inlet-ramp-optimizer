import numpy as np
import cantera as ct

'''
Author: Panagiotis Sachinis
Year: 2026

Create and manage gas object. Useful in this application to minimize calls to
Cantera which take up a lot of time and memory. Further classes may be developed
if a different gas model is required (though alterations to other scripts may
be required).
'''

class realGas:

    def __init__(self,mechanism="air.yaml",composition="O2:0.21, N2:0.79"):

        self.gas = ct.Solution(mechanism)
        self.gas.X = composition
    
    def get_state(self):
        """Returns a dictionary of current gas properties."""
        return {
            'T': self.gas.T,
            'P': self.gas.P,
            'RHO': self.gas.density,
            'H': self.gas.enthalpy_mass,
            'A': self.gas.sound_speed,
            'GAMMA': self.gas.cp_mass / self.gas.cv_mass
        }