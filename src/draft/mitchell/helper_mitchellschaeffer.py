import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MitchellSimulation import MitchellSimulation
import progressbar
import dill as pickle

def generate_vh_data(N, trans, sample_rate=1, Ngrid=100):
    Nx = Ngrid
    Ny = Ngrid
    deltaT = 1e-2
    deltaX = 0.1
    D = 5e-3
    h = D/deltaX**2

    #constants according to https://books.google.de/books?id=aB34DAAAQBAJ&pg=PA134&lpg=PA134&dq=mitchell-schaefer+model&source=bl&ots=RVuc3hoJwW&sig=ukfFhjF_COsljaaznv5uB6Cn5V8&hl=de&sa=X&ved=0ahUKEwiozdj8ic7TAhURLVAKHfa3A5wQ6AEIOTAC#v=onepage&q=mitchell-schaefer%20model&f=false
    sim = MitchellSimulation(Nx, Ny, deltaT, deltaX, D, D, t_in=0.3, t_out=6.0, t_close=150, t_open=20, v_gate=0.13,)

    sim.initialize_random(42, 0.1)

    bar = progressbar.ProgressBar(max_value=trans+N, redirect_stdout=True)

    for i in range(trans):
        sim.explicit_step(chaotic=True)
        bar.update(i)

    data = np.empty((2, N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step(chaotic=True)
        data[0, i] = sim._v
        data[1, i] = sim._h
        bar.update(i+trans)

    bar.finish()
    return data

def generate_data(N, trans, sample_rate=1, Ngrid=100):
    Nx = 150
    Ny = 150
    deltaT = 1e-2
    deltaX = 0.1
    D = 1e-1
    h = D/deltaX**2

    #constants according to https://books.google.de/books?id=aB34DAAAQBAJ&pg=PA134&lpg=PA134&dq=mitchell-schaefer+model&source=bl&ots=RVuc3hoJwW&sig=ukfFhjF_COsljaaznv5uB6Cn5V8&hl=de&sa=X&ved=0ahUKEwiozdj8ic7TAhURLVAKHfa3A5wQ6AEIOTAC#v=onepage&q=mitchell-schaefer%20model&f=false
    sim = MitchellSimulation(Nx, Ny, deltaT, deltaX, D, D, t_in=0.3, t_out=6.0, t_close=150, t_open=20, v_gate=0.13,)

    sim.initialize_random(42, 0.1)

    bar = progressbar.ProgressBar(max_value=trans+N, redirect_stdout=True)

    for i in range(trans):
        sim.explicit_step(chaotic=True)
        bar.update(i)

    data = np.empty((N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step(chaotic=True)
        data[i] = sim._v
        bar.update(i+trans)

    bar.finish()
    return data