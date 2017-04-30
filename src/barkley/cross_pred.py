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

def print_field(input_y, input_x, output_y, output_x):
    print_matrix = np.zeros((30,30))
    print_matrix[input_y, input_x] = 1.0
    print_matrix[output_y, output_x] = -1.0
    for y in range(30):
        string = "|"
        for x in range(30):
            if (print_matrix[y,x] == 1.0):
                string += "x"
            elif (print_matrix[y,x] == -1.0):
                string += "0"
            else:
                string += "."

        string += "|"
        print(string)

N = 160

if (os.path.exists("cache/raw/20000_{0}.dat.npy".format(N)) == False):
    print("generating data...")
    data = generate_data(20000, 50000, 5, Ngrid=N)
    np.save("cache/raw/20000_{0}.dat.npy".format(N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("cache/raw/20000_{0}.dat.npy".format(N))
    print("loading finished")

training_data = data[:18000]
test_data = data[18000:20000]

#input_y, input_x, output_y, output_x = create_patch_indices((12,17), (12,17), (13,16), (13,16)) # -> yields MSE=0.0115 with leak_rate = 0.8
input_y, input_x, output_y, output_x = create_patch_indices((40, 119), (40, 119), (50, 109), (50, 109)) # -> yields MSE=0.0873 with leak_rate = 0.3

training_data_in =  training_data[:, input_y, input_x].reshape(-1, len(input_y))
training_data_out =  training_data[:, output_y, output_x].reshape(-1, len(output_y))

test_data_in =  test_data[:, input_y, input_x].reshape(-1, len(input_y))
test_data_out =  test_data[:, output_y, output_x].reshape(-1, len(output_y))

n_units = 6000

generate_new = False
if (os.path.exists("cache/esn/cross_pred_" + str(len(input_y)) + "_" + str(len(output_y)) + "_" + str(n_units) + ".dat") == False):
    generate_new = True


if (generate_new):
    print("setting up...")
    esn = ESN(n_input = len(input_y), n_output = len(output_y), n_reservoir = n_units, #used to be 1700
            weight_generation = "advanced", leak_rate = 0.2, spectral_radius = 0.1,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-0], solver = "lsqr")#,
#            out_activation = lambda x: 0.5*(1+np.tanh(x/2)), out_inverse_activation = lambda x:2*np.arctanh(2*x-1))

    print("fitting...")

    train_error = esn.fit(training_data_in, training_data_out, verbose=1)
    print("train error: {0}".format(train_error))

    print("saving to: " + "cache/esn/cross_pred_" + str(len(input_y)) + "_" + str(len(output_y)) + "_" + str(n_units) + ".dat")
    esn.save("cache/esn/cross_pred_" + str(len(input_y)) + "_" + str(len(output_y)) + "_" + str(n_units) + ".dat")

else:
    print("loading model...")
    esn = ESN.load("cache/esn/cross_pred_" + str(len(input_y)) + "_" + str(len(output_y)) + "_" + str(n_units) + ".dat")

print("predicting...")
pred = esn.predict(test_data_in, verbose=1)
pred[pred>1.0] = 1.0
pred[pred<0.0] = 0.0

merged_prediction = test_data.copy()
merged_prediction[:, output_y, output_x] = pred

diff = pred.reshape((-1, len(output_y))) - test_data_out
mse = np.mean((diff)**2)
print("test error: {0}".format(mse))

difference = np.abs(test_data - merged_prediction)

show_results({"pred": merged_prediction , "orig": test_data, "diff": difference})

print("done.")
