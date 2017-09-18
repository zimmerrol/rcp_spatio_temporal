"""
    Can be used to generate BOCF timeseries. The usage of "demo.py" is more recommended offers more features.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BOCFSimulation import BOCFSimulation
import progressbar
import dill as pickle

def generate_uvws_data(N, trans, sample_rate=1, Ngrid=100):
    Nx = Ngrid
    Ny = Ngrid
    deltaT = 1e-2
    deltaX = 1.0

    D = 1.171

    sim = BOCFSimulation(Nx, Ny, D, deltaT, deltaX)

    #sim.initialize_random(42, deltaX)
    sim.initialize_double_spiral()

    bar = progressbar.ProgressBar(max_value=trans+N, redirect_stdout=True)

    for i in range(trans):
        sim.explicit_step()
        bar.update(i)

    data = np.empty((4, N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step()
        data[0, i] = sim._u
        data[1, i] = sim._v
        data[2, i] = sim._w
        data[3, i] = sim._s
        bar.update(i+trans)

    bar.finish()
    return data
