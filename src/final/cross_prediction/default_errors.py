import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../barkley'))
sys.path.insert(1, os.path.join(sys.path[0], '../../mitchell'))

import numpy as np

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
    if not os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N)):
        data = bh.generate_uv_data(ndata, 20000, 5, Ngrid=N)
        np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N))
else:
    if not os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N)):
        data = mh.generate_vh_data(ndata, 20000, 50, Ngrid=N)
        np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N))

print("Data loaded")

#at the moment we are doing a u -> v / v -> h cross prediction.
if direction in ["vu", "hv"]:
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
