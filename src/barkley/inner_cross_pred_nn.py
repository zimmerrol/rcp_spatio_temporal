import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN
from helper import *
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors as NN

N = 150
ndata = 10000
testLength = 1000
ddim = 10
tau = 32

"""
ddim    accuracy
10      
15      0.254207
"""

def create_1d_delay_coordinates(data, delay_dimension, tau):
    result = np.repeat(data[:, :, np.newaxis], repeats=delay_dimension, axis=2)
    print("delayed data copied")

    for n in range(1, delay_dimension):
        result[:, :, n] = np.roll(result[:, :, n], n*tau, axis=0)
    result[0:delay_dimension-1,:] = 0

    return result

if (os.path.exists("cache/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
    print("generating data...")
    data = generate_data(ndata, 50000, 5, Ngrid=N)
    np.save("cache/raw/{0}_{1}.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("cache/raw/{0}_{1}.dat.npy".format(ndata, N))
    print("loading finished")

input_y, input_x, output_y, output_x = create_patch_indices((0, N), (0, N), (1, N-1), (1, N-1))



delayed_input_data = create_1d_delay_coordinates(data[:, input_y, input_x], ddim, tau)
print(delayed_input_data.shape)

training_data_in = delayed_input_data[:ndata-testLength]    #delayed_
test_data_in = delayed_input_data[ndata-testLength:]        #delayed_

print(test_data_in.shape)

training_data_out = data[:ndata-testLength][:, output_y, output_x].reshape(-1, len(output_y))
test_data_out = data[ndata-testLength:][:, output_y, output_x].reshape(-1, len(output_y))

flat_test_data_in = test_data_in.reshape(len(test_data_out), -1)
flat_training_data_in = training_data_in.reshape(len(training_data_out), -1)

flat_test_data_out = test_data_out.reshape(-1, len(output_y))
flat_training_out = training_data_out.reshape(-1, len(output_y))

print(flat_training_data_in.shape)
print(flat_training_out.shape)

neigh = NN(2, n_jobs=16)
print("fitting")

neigh.fit(flat_training_data_in)

print("predicting...")
distances, indices = neigh.kneighbors(flat_test_data_in)
print(distances)

test_data = data[ndata-testLength:]

flat_prediction = (flat_training_out[indices[:, 0]]+flat_training_out[indices[:, 1]])/2.0
print(flat_prediction.shape)
print(indices[:, 0].shape)
#prediction = flat_prediction.reshape(2000, 148, 148)

merged_prediction = test_data.copy()
merged_prediction[:, output_y, output_x] = flat_prediction

difference = np.abs(test_data - merged_prediction)

diff = flat_test_data_out - flat_prediction
print("MSE = {0:4f}".format(np.mean(diff**2)))

show_results({"pred": merged_prediction , "orig": test_data, "diff": difference})

print("done.")
