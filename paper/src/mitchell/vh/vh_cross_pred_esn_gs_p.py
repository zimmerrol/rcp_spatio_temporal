import os

id = int(os.getenv("SGE_TASK_ID", 0))
first = int(os.getenv("SGE_TASK_FIRST", 0))
last = int(os.getenv("SGE_TASK_LAST", 0))
print("ID {0}".format(id))
print("Task %d of %d tasks, starting with %d." % (id, last - first + 1, first))

print("This job was submitted from %s, it is currently running on %s" % (os.getenv("SGE_O_HOST"), os.getenv("HOSTNAME")))

print("NHOSTS: %s, NSLOTS: %s" % (os.getenv("NHOSTS"), os.getenv("NSLOTS")))

import sys
print(sys.version)

#id=1

# -*- coding: utf-8 -*-

#TODO: Use http://stackoverflow.com/questions/28821910/how-to-get-around-the-pickling-error-of-python-multiprocessing-without-being-in

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ESN import ESN
import progressbar
import dill as pickle #we require version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import copy
from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
#from pathos.multiprocessing import Pool
import multiprocessing
import ctypes
from GridSearchP import GridSearchP

from helper import *

N = 150
ndata = 10000
sigma = 5
sigma_skip = 2
eff_sigma = int(np.ceil(sigma/sigma_skip))
patch_radius = sigma // 2
n_units = 450

def mainFunction():
    global output_weights, frame_output_weights, last_states

    if (os.path.exists("../cache/raw/{0}_{1}.vh.dat.npy".format(ndata, N)) == False):
        print("generating data...")
        data = generate_vh_data(ndata, 20000, 50, Ngrid=N) #20000 was 50000 ndata
        np.save("../cache/raw/{0}_{1}.vh.dat.npy".format(ndata, N), data)
        print("generating finished")
    else:
        print("loading data...")
        data = np.load("../cache/raw/{0}_{1}.vh.dat.npy".format(ndata, N))

        """
        #switch the entries for the u->v prediction
        tmp = data[0].copy()
        data[0] = data[1].copy()
        data[1] = tmp.copy()
        """

        print("loading finished")


    training_data = data[:, :ndata-2000]
    test_data = data[:, ndata-2000:]

    """
    esn = ESN(n_input = eff_sigma*eff_sigma, n_output = 1, n_reservoir = n_units,
            weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[3e-6], solver = "lsqr", output_input_scaling=0.01)
    """
    #0.2,  0.4, .6, .8, .95,
    if (id == 1):
        gs = GridSearchP(
                param_grid={"n_reservoir": [50], "spectral_radius": [0.5, .8, .9, .95], "leak_rate": [.05, .1, .2, .5, .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36], "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001],"regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 2):
        gs = GridSearchP(
                param_grid={"n_reservoir": [100], "spectral_radius": [0.5, .8, .9, .95], "leak_rate": [.05, .1, .2, .5, .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36], "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001],"regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 3):
        gs = GridSearchP(
                param_grid={"n_reservoir": [200], "spectral_radius": [0.5, .8, .9, .95], "leak_rate": [.05, .1, .2, .5, .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36], "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001],"regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 4):
        gs = GridSearchP(
                param_grid={"n_reservoir": [400], "spectral_radius": [0.5, .8, .9, .95], "leak_rate": [.05, .1, .2, .5, .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36], "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001],"regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 5):
        gs = GridSearchP(
                param_grid={"n_reservoir": [50], "spectral_radius": [0.1, 0.2, 0.3, 0.4], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 6):
        gs = GridSearchP(
                param_grid={"n_reservoir": [100], "spectral_radius": [0.1, 0.2, 0.3, 0.4], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 7):
        gs = GridSearchP(
                param_grid={"n_reservoir": [200], "spectral_radius": [0.1, 0.2, 0.3, 0.4], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 8):
        gs = GridSearchP(
                param_grid={"n_reservoir": [400], "spectral_radius": [0.1, 0.2, 0.3, 0.4], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 9):
        gs = GridSearchP(
                param_grid={"n_reservoir": [50], "spectral_radius": [1.1, 1.2, 1.3, 1.4, 1.5], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 10):
        gs = GridSearchP(
                param_grid={"n_reservoir": [100], "spectral_radius": [1.1, 1.2, 1.3, 1.4, 1.5], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 11):
        gs = GridSearchP(
                param_grid={"n_reservoir": [200], "spectral_radius": [1.1, 1.2, 1.3, 1.4, 1.5], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    elif (id == 12):
        gs = GridSearchP(
                param_grid={"n_reservoir": [400], "spectral_radius": [1.1, 1.2, 1.3, 1.4, 1.5], "leak_rate": [.05, .1, .2, .5 , .7, .9, .95],
                            "random_seed": [42,41,40,39,38,37,36],  "sparseness": [.1, .2], "noise_level": [0.001, 0.0001, 0.00001], "regression_parameters": [[5e-2],[5e-3],[5e-4],[5e-5],[5e-6]]},
                fixed_params={"n_output": 1, "n_input": 1, "solver": "lsqr", "weight_generation": "advanced"},
                esnType=ESN)
    print("start fitting...")

    sys.stdout.flush()
    results = gs.fit(training_data[1, :, N//2, N//2].reshape((-1,1)), training_data[0, :, N//2, N//2].reshape((-1,1)), [(test_data[1, :, N//2, N//2].reshape((-1,1)), test_data[0, :, N//2, N//2].reshape((-1,1)))], printfreq=100, verbose=2, n_jobs=14)
    print("done:\r\n")
    print(results)

    print("\r\nBest result (mse =  {0}):\r\n".format(gs._best_mse))
    print(gs._best_params)

class ForceIOStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if not self.stream.isatty():
            os.fsync(self.stream.fileno())

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

if __name__== '__main__':
    sys.stdout = ForceIOStream(sys.stdout)
    sys.stderr = ForceIOStream(sys.stderr)

    mainFunction()
