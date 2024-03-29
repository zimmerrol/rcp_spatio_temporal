"""
    Calculates the default errors for the outer->inner prediction for the 3 different models.
"""

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
ndata = 30000
trainLength = 15000
testLength = 2000
data = None

parser = argparse.ArgumentParser(description='')
parser.add_argument('direction', default="u", nargs=1, type=str, help="u: predict inner of u, v: predict inner of v")
args = parser.parse_args()

if args.direction[0] not in ["u", "v"]:
    raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
else:
    direction = args.direction[0]

if (direction == "u"):
    if (os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N)) == False):
        data = bh.generate_data(ndata, 20000, 5, Ngrid=N)
        np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(ndata, N))
else:
    if (os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N)) == False):
        data = mh.generate_data(ndata, 20000, 50, Ngrid=N)
        np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N), data)
    else:
        data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(ndata, N))

data = data[0]

"""
    Creates the source (outer values) and the target (inner values) from the data time series.
"""
def create_data(innerSize, borderSize, data):
    halfInnerSize = int(np.floor(innerSize / 2))
    borderSize = 1
    center = N//2
    rightBorderAdd = 1 if innerSize != 2*halfInnerSize else 0

    input_y, input_x, output_y, output_x = create_patch_indices(
                                                (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                                (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                                (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
                                                (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd)
                                            )

    inputData = data[:, input_y, input_x]
    outputData = data[:, output_y, output_x]

    train_data_in = inputData[:trainLength]
    test_data_in = inputData[trainLength:trainLength+testLength]

    train_data_out = outputData[:trainLength]
    test_data_out = outputData[trainLength:trainLength+testLength]

    return train_data_in, train_data_out, test_data_in, test_data_out

"""
    Calcualtes the default errors for the specified values of a and b (see thesis).
"""
def default_errors(innerSize, borderSize, data):
    train_data_in, train_data_out, test_data_in, test_data_out = create_data(innerSize, borderSize, data)

    #use the mean of the inner data of the train data
    mean = np.mean(train_data_out)
    msemean = np.mean((test_data_out-mean)**2)

    #use the mean of the border values of the test data
    meanborder = np.repeat(np.mean(test_data_in, axis=1), innerSize*innerSize).reshape((testLength, innerSize*innerSize))
    msemeanborder = np.mean((test_data_out-meanborder)**2)

    print("{0}\t{1}\t{2}\t{3}".format(innerSize, borderSize, msemean, msemeanborder))


#calculate and print the default error for all used configurations
settings = [(4,1), (8,1), (16,1), (32,1), (64,1), (128,1),
            (4,2), (8,2), (16,2), (32,2), (64,2), (128,2),
            (4,3), (8,3), (16,3), (32,3), (64,3), (128,3),
            (146,1), (146,2), (148,1)]

print("inner size\t\tborder size\t\tmean train\t\tmean border")
for innerSize, borderSize in settings:
    default_errors(innerSize, borderSize, data)
