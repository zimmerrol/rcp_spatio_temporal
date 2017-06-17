import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, os.path.join(grandparentdir, "barkley"))
sys.path.insert(0, os.path.join(grandparentdir, "mitchell"))

import os
import numpy as np
from matplotlib import pyplot as plt
from BarkleySimulation import BarkleySimulation
import progressbar

from helper import *
import barkley_helper as bh
import mitchell_helper as mh
import argparse

N = 150
ndata = 30000
ntrain = 1500
nvalidation = 2000
ntest = 2000
data = None

parser = argparse.ArgumentParser(description='')
parser.add_argument('direction', default="u", nargs=1, type=str, help="uv: u -> v, vu: v -> u, vh: v -> h, hv: h -> v")
args = parser.parse_args()

if args.direction[0] not in ["uv", "vu", "vh", "hv"]:
    raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
else:
    direction = args.direction[0]

if (direction in ["uv", "vu"]):
    if (os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N)) == False):
        data = bh.generate_uv_data(ndata, 20000, 5, Ngrid=N)
        np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N))
else:
    if (os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N)) == False):
        data = mh.generate_vh_data(ndata, 20000, 50, Ngrid=N)
        np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N))

#at the moment we are doing a u -> v / v -> h cross prediction.
if (direction in ["vu", "hv"]):
    #switch the entries for the v -> u / h -> v prediction
    tmp = data[0].copy()
    data[0] = data[1].copy()
    data[1] = tmp.copy()

training_data = data[:, :ntrain]
test_data = data[:, ntrain + nvalidation:ntrain + nvalidation + ntest]

#use mean value as prediciton:
mean = np.mean(training_data[1])
meanpredmse = np.mean((test_data[1] - mean)**2)

#use h as value for v
hvpredmse = np.mean((test_data[1] - test_data[0])**2)

print("Using the mean of target_train as target_test prediction: ")
print("\tMSE = {0}".format(meanpredmse))

print("Using the value of source_test as the target_test prediction: ")
print("\tMSE = {0}".format(hvpredmse))
