import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, os.path.join(grandparentdir, "barkley"))
sys.path.insert(0, os.path.join(grandparentdir, "mitchell"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import gaussian_filter
import progressbar
import dill as pickle

from ESN import *
from RBF import *
from NN import *

from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import multiprocessing
import ctypes
from multiprocessing import process

from helper import *
import barkley_helper as bh
import mitchell_helper as mh
import argparse

from GridSearchP import GridSearchP

N = 150
ndata =30000
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

    id = int(os.getenv("SGE_TASK_ID", 0))

    innerSize, borderSize = [(4,1), (8,1), (16,1), (32,1), (64,1), (128,1),
                             (4,2), (8,2), (16,2), (32,2), (64,2), (128,2),
                             (4,3), (8,3), (16,3), (32,3), (64,3), (128,3),
                             (146,1), (146,2), (148,1)][id-1]

    halfInnerSize = int(np.floor(innerSize / 2))
    borderSize = 1
    center = N//2
    rightBorderAdd = 1 if innerSize != 2*halfInnerSize else 0
setup_constants()

def generate_data(N, trans, sample_rate, Ngrid):
    data = None

    if (direction == "u"):
        if (os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)) == False):
            data = bh.generate_uv_data(N, 50000, 5, Ngrid=Ngrid)
            np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
    else:
        if (os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)) == False):
            data = mh.generate_vh_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))

    data = data[0]

    input_y, input_x, _, _ = create_patch_indices(
        (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
        (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
        (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
        (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd))

    _, _, output_y, output_x = create_patch_indices(
        (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
        (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
        (center - (1), center + (1) + 0),
        (center - (1), center + (1) + 0))

    inputData = data[:, input_y, input_x]
    outputData = data[:, output_y, output_x]

    return inputData, outputData

def mainFunction():
    global output_weights, frame_output_weights, last_states

    inputData, outputData = generate_data(ndata, 20000, 50, Ngrid=N)

    #id: 1-21

    print(inputData.shape)

    param_grid = {"n_reservoir": [50, 200, 400], "spectral_radius": [0.1, 0.5, 0.8, 0.95, 1.0, 1.1, 1.5, 3.0], "leak_rate": [.05, .2, .5 , .7, .9, .95],
                "random_seed": [42,41,40,39],  "sparseness": [.1, .2], "noise_level": [0.0001, 0.00001], "input_density": [5, 10, 15, 20, 50, 100]/inputData.shape[1]
                "regression_parameters": [[5e4], [5e3], [5e2], [5e1], [5e0], [5e-1], [5e-2],[5e-3],[5e-4]]}
    fixed_params = {"n_output": 1, "n_input": inputData.shape[1], "solver": "lsqr", "weight_generation": "advanced"}

    gs = GridSearchP(param_grid, fixed_params, esnType=ESN)

    print("start fitting...")

    sys.stdout.flush()
    results = gs.fit(inputData[:trainLength], outputData[:trainLength],
                [(inputData[trainLength:trainLength+testLength], outputData[trainLength:trainLength+testLength])],
                printfreq=100, verbose=2, n_jobs=16)
    print("results:\r\n")
    print(results)
    print("")

    print("\r\nBest result (mse =  {0}):\r\n".format(gs._best_mse))
    print("best parameters {0}".format(gs._best_params))

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
