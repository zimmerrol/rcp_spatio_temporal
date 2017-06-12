import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../barkley'))
sys.path.insert(1, os.path.join(sys.path[0], '../../mitchell'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import gaussian_filter
import progressbar
import dill as pickle

from ESN import ESN
from RBF import RBF
from NN import NN

from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import multiprocessing
import ctypes
from multiprocessing import process

import helper as hp
import barkley_helper as bh
import mitchell_helper as mh
import argparse

from GridSearchP import GridSearchP

N = 150
ndata = 30000
testLength = 2000
trainLength = 15000

def parse_arguments():
    global id, predictionMode, direction

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('direction', default="u", nargs=1, type=str, help="u: unblur u, v: unblurr v")
    args = parser.parse_args()

    if args.direction[0] not in ["u", "v"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        direction = args.direction[0]

    print("Prediction via: {0}".format(direction))
parse_arguments()

def setup_constants():
    global innerSize, halfInnerSize, borderSize, center, rightBorderAdd

    sge_id = int(os.getenv("SGE_TASK_ID", 0))

    halfInnerSize = int(np.floor(innerSize / 2))
    borderSize = 1
    center = N//2
    rightBorderAdd = 1 if innerSize != 2*halfInnerSize else 0
setup_constants()

def generate_data(N, trans, sample_rate, Ngrid):
    data = None

    if direction == "u":
        if not os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)):
            data = bh.generate_uv_data(N, 50000, 5, Ngrid=Ngrid)
            np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
    else:
        if not os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)):
            data = mh.generate_vh_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))

    data = data[0]

    return data

def mainFunction():
    data = generate_data(ndata, 20000, 50, Ngrid=N)

    prediction_length = 100

    sigma, sigma_skip = [(1, 1), (3, 1), (5, 1), (5, 2), (7, 1), (7, 2), (7, 3)][id-1]
    patch_radius = sigma//2
    input_size = [1, 9, 25, 9, 49, 16, 9][id-1]

    input_data = data[0, :-prediction_length, N//2-patch_radius:N//2+patch_radius+1, N//2-patch_radius:N//2+patch_radius+1][:, ::sigma_skip, ::sigma_skip].reshape((trainLength, -1))
    output_data = data[0, prediction_length:, N//2, N//2].reshape((-1, 1))

    param_grid = {"n_reservoir": [50, 200, 300, 400], "spectral_radius": [0.1, 0.5, 0.8, 0.95, 1.0, 1.1, 1.2, 1.5, 1.75, 2.5, 3.0], "leak_rate": [.05, .2, .5 , .7, .9, .95],
                  "random_seed": [42, 41, 40, 39], "sparseness": [.05, .1, .2], "noise_level": [0.0001, 0.00001], "regression_parameters": [[5e-2], [5e-3], [5e-4], [5e-5], [5e-6], [5e-7], [5e-8]]}
    fixed_params = {"n_output": 1, "n_input": input_size, "solver": "lsqr", "weight_generation": "advanced"}

    gridsearch = GridSearchP(param_grid, fixed_params, esnType=ESN)

    print("start fitting...")

    sys.stdout.flush()
    results = gridsearch.fit(input_data[:trainLength], output_data[:trainLength],
                             [(input_data[trainLength:trainLength+testLength], output_data[trainLength:trainLength+testLength])],
                             printfreq=100, verbose=2, n_jobs=16)
    print("results:\r\n")
    print(results)
    print("")

    print("\r\nBest result (mse =  {0}):\r\n".format(gridsearch._best_mse))
    print("best parameters {0}".format(gridsearch._best_params))

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

if __name__ == '__main__':
    sys.stdout = ForceIOStream(sys.stdout)
    sys.stderr = ForceIOStream(sys.stderr)

    mainFunction()
