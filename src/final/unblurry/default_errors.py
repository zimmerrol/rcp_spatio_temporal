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
from scipy.ndimage.filters import gaussian_filter

N = 150
ndata = 10000
trainLength = 8000
testLength = ndata-trainLength
data = None

parser = argparse.ArgumentParser(description='')
parser.add_argument('direction', default="u", nargs=1, type=str, help="u: unblur u, v: unblurr v")
args = parser.parse_args()

if args.direction[0] not in ["u", "v"]:
    raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
else:
    direction = args.direction[0]

if (direction == "u"):
    if (os.path.exists("../../cache/barkley/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
        data = bh.generate_data(ndata, 20000, 5, Ngrid=N)
        np.save("../../cache/barkley/raw/{0}_{1}.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/barkley/raw/{0}_{1}.dat.npy".format(ndata, N))
else:
    if (os.path.exists("../../cache/mitchell/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
        data = mh.generate_data(ndata, 20000, 50, Ngrid=N)
        np.save("../../cache/mitchell/raw/{0}_{1}.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/mitchell/raw/{0}_{1}.dat.npy".format(ndata, N))

training_data = np.empty((2, trainLength, N,N))
test_data = np.empty((2, testLength, N,N))

training_data[0] = data[:trainLength]
test_data[0] = data[trainLength:]

for t in range(trainLength):
    training_data[1, t, :, :] = gaussian_filter(training_data[0, t], sigma=8.0)

for t in range(testLength):
    test_data[1, t, :, :] = gaussian_filter(test_data[0, t], sigma=8.0)

#use mean value as prediciton:
mean = np.mean(training_data[0])
meanpredmse = np.mean((test_data[0] - mean)**2)

#use h as value for v
hvpredmse = np.mean((test_data[0] - test_data[1])**2)

print("Using the mean of target_train as target_test prediction: ")
print("\tMSE = {0}".format(meanpredmse))

print("Using the value of source_test as the target_test prediction: ")
print("\tMSE = {0}".format(hvpredmse))
