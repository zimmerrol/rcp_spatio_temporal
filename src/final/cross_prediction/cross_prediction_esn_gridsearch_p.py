import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../barkley'))
sys.path.insert(1, os.path.join(sys.path[0], '../../mitchell'))
sys.path.insert(1, os.path.join(sys.path[0], '../../bocf'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import gaussian_filter
import progressbar
import dill as pickle

from ESN import *
from RBF import *
from NN import *
from GridSearchP import GridSearchP

#We require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
from multiprocessing import Process, Queue, Manager, Pool
import multiprocessing
import ctypes
from multiprocessing import process

from helper import *
import barkley_helper as bh
import bocf_helper as bocfh
import mitchell_helper as mh
import argparse



N = 150
ndata = 30000
testLength = 2000
trainLength = 15000

def parse_arguments():
    global id, predictionMode, direction

    id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('direction', default="vu", nargs=1, type=str, help="vu: v -> u, uv: u -> v, hv: h -> v, vh: v -> h, bocf_uv: BOCF u -> v, bocf_uw: BOCF u -> w, bocf_us: BOCF u -> s")
    args = parser.parse_args()

    if args.direction[0] not in ["vu", "uv", "hv", "vh", "bocf_uv", "bocf_uw", "bocf_us"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        direction = args.direction[0]

    print("Prediction via: {0}".format(direction))
parse_arguments()

def generate_data(N, trans, sample_rate, Ngrid):
    data = None

    if direction in ["uv", "vu"]:
        if not os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)):
            data = bh.generate_uv_data(N, 50000, 5, Ngrid=Ngrid)
            np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
    elif direction in ["bocf_uv", "bocf_uw", "bocf_us"]:
        if not os.path.exists("../../cache/bocf/raw/{0}_{1}.uvws.dat.npy".format(N, Ngrid)):
            data = bocfh.generate_uvws_data(N, 50000, 50, Ngrid=Ngrid)
            np.save("../../cache/bocf/raw/{0}_{1}.uvws.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/bocf/raw/{0}_{1}.uvws.dat.npy".format(N, Ngrid))
    else:
        if not os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)):
            data = mh.generate_vh_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))

    #at the moment we are doing a u -> v / v -> h cross prediction (index 0 -> index 1)
    if (direction in ["vu", "hv"]):
        #switch the entries for the v -> u / h -> v prediction
        tmp = data[0].copy()
        data[0] = data[1].copy()
        data[1] = tmp.copy()

    if direction in ["bocf_uv", "bocf_uw", "bocf_ws"]:
        real_data = np.array((2, N, Ngrid, Ngrid))
        real_data[0] = data[0].copy()

        if direction == "bocf_uv":
            real_data[1] = data[1].copy()
        elif direction == "bocf_uw":
            real_data[1] = data[2].copy()
        elif direction == "bocf_us":
            real_data[1] = data[3].copy()

        data = real_data

    return data

def mainFunction():
    data = generate_data(ndata, 20000, 50, Ngrid=N)

    training_data = data[:, :trainLength]
    test_data = data[:, trainLength:trainLength+testLength]

    sigma, sigma_skip = [(1, 1), (3, 1), (5, 1), (5, 2), (7, 1), (7, 2), (7, 3)][id-1]
    patch_radius = sigma//2
    input_size = [1, 9, 25, 9, 49, 16, 9][id-1]

    param_grid = {"n_reservoir": [50, 200, 400], "spectral_radius": [0.1, 0.5, 0.8, 0.95, 1.1, 1.5, 3.0], "leak_rate": [.05, .2, .5 , .7, .9, .95],
                  "random_seed": [42, 41, 40, 39],  "sparseness": [.1, .2], "noise_level": [0.0001, 0.00001], "regression_parameters": [[5e-2], [5e-3], [5e-4], [5e-5], [5e-6]]}
    fixed_params = {"n_output": 1, "n_input": input_size, "solver": "lsqr", "weight_generation": "advanced"}

    gridsearch = GridSearchP(param_grid, fixed_params, esnType=ESN)

    print("start fitting...")

    sys.stdout.flush()
    results = gridsearch.fit(
        training_data[0, :, N//2-patch_radius:N//2+patch_radius+1, N//2-patch_radius:N//2+patch_radius+1][:, ::sigma_skip, ::sigma_skip].reshape((trainLength, -1)), training_data[1, :, N//2, N//2].reshape((trainLength, 1)),
        [(test_data[0, :, N//2-patch_radius:N//2+patch_radius+1, N//2-patch_radius:N//2+patch_radius+1][:, ::sigma_skip, ::sigma_skip].reshape((testLength, -1)), test_data[1, :, N//2, N//2].reshape((testLength, 1)))],
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

if __name__== '__main__':
    sys.stdout = ForceIOStream(sys.stdout)
    sys.stderr = ForceIOStream(sys.stderr)

    mainFunction()
