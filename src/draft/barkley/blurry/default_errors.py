import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import os
import numpy as np
from matplotlib import pyplot as plt
from BarkleySimulation import BarkleySimulation
import progressbar

from helper import *
from barkley_helper import *
from scipy.ndimage.filters import gaussian_filter

N = 150
ndata = 10000
trainLength = 8000
testLength = ndata-trainLength
data = None

if (os.path.exists("../cache/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
    print("generating data...")
    data = generate_data(ndata, 20000, 5, Ngrid=N) #20000 was 50000
    np.save("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("../cache/raw/{0}_{1}.dat.npy".format(ndata, N))

    print("loading finished")

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

print("Using the mean of u_train as prediction: ")
print("\tMSE = {0}".format(meanpredmse))

print("Using the value of u_{test, blurred} as the h prediction: ")
print("\tMSE = {0}".format(hvpredmse))

