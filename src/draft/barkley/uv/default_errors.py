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

N = 150
ndata = 10000
ntest = 2000
data = None

if (os.path.exists("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N)) == False):
    print("data missing")
    print("generating data...")
    data = generate_uv_data(ndata, 50000, 5, Ngrid=N)
    np.save("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N))
    print("loading finished")

training_data = data[:, :ndata-ntest]
test_data = data[:, ndata-ntest:]

#use mean value as prediciton:
mean = np.mean(training_data[0])
meanpredmse = np.mean((test_data[0] - mean)**2)

#use h as value for v
hvpredmse = np.mean((test_data[0] - test_data[1])**2)

print("Using the mean of u_train as prediction: ")
print("\tMSE = {0}".format(meanpredmse))

print("Using the value of v as the h prediction: ")
print("\tMSE = {0}".format(hvpredmse))
