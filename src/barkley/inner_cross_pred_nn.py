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

N = 150
ndata = 10000
testLength = 2000
ddim = 1
tau = 32

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

delayed_data = create_1d_delay_coordinates(data, ddim, tau)

delayed_training_data = delayed_data[:ndata-testLength]
delayed_test_data = delayed_data[ndata-testLength:]

input_y, input_x, output_y, output_x = create_patch_indices((0, N), (0, N), (1, N-1), (1, N-1))

training_data_in = delayed_training_data[:, input_y, input_x].reshape(-1, len(input_y), delayed_training_data.shape[2])
training_data_out = training_data[ndata-testLength:][:, output_y, output_x].reshape(-1, len(output_y))

test_data_in = delayed_test_data[:, input_y, input_x].reshape(-1, len(input_y), delayed_training_data.shape[2])
test_data_out = test_data[ndata-testLength:][:, output_y, output_x].reshape(-1, len(output_y))

flat_test_data_in = test_data_in.reshape(-1, delayed_patched_v_data.shape[2])
flat_training_data_in = training_data_in.reshape(-1, delayed_patched_v_data.shape[2])

flat_test_data_out = test_data_out.reshape(-1, len(output_y))
flat_training_out = training_data_out.reshape(-1, len(output_y))

neigh = NN(2)
print("fitting")

neigh.fit(flat_test_data_in)

print("predicting...")
distances, indices = neigh.kneighbors(flat_training_data_in)
print(distances)

flat_prediction = flat_training_out[indices[:, 0]]
#prediction = flat_prediction.reshape(2000, 148, 148)

merged_prediction = test_data.copy()
merged_prediction[:, output_y, output_x] = flat_prediction

difference = np.abs(test_data - merged_prediction)

show_results({"pred": merged_prediction , "orig": test_data, "diff": difference})

print("done.")
