import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN
import progressbar

from helper import *

N = 150

if (os.path.exists("../cache/raw/20000_{0}.uv.dat.npy".format(N)) == False):
    print("generating data...")
    data = generate_data(20000, 50000, 5, Ngrid=N)
    np.save("../cache/raw/20000_{0}.uv.dat.npy".format(N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("../cache/raw/20000_{0}.uv.dat.npy".format(N))
    print("loading finished")

print("preparing data...")


N = 5
data = data[:, :, 50:50+N, 50:50+N].reshape((2, -1, N, N))
print(data.shape)

training_data = data[:, :18000]
test_data = data[:, 18000:20000]

training_data_in = training_data[1].reshape(-1, N*N)
training_data_out = training_data[0].reshape(-1, N*N)

test_data_in = test_data[1].reshape(-1, N*N)
test_data_out = test_data[0].reshape(-1, N*N)

n_units = 2000

model_save_name = "../cache/esn/uv/cross_pred_" + str(N) + "_" + str(n_units) + ".dat"
generate_new = False
if (os.path.exists(model_save_name) == False):
    generate_new = True


if (generate_new):
    print("setting up...")
    esn = ESN(n_input = N*N, n_output = N*N, n_reservoir = n_units,
            weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-1], solver = "lsqr")#,
            #out_activation = lambda x: 0.5*(1+np.tanh(x/2)), out_inverse_activation = lambda x:2*np.arctanh(2*x-1))

    print("fitting...")

    train_error = esn.fit(training_data_in, training_data_out, verbose=1)
    print("train error: {0}".format(train_error))

    print("saving to: " + model_save_name)
    if not os.path.exists("../cache/esn/uv/"):
        os.makedirs("../cache/esn/uv/")
    esn.save(model_save_name)

else:
    print("loading model...")
    esn = ESN.load(model_save_name)

print("predicting...")
pred = esn.predict(test_data_in, verbose=1)
pred[pred>1.0] = 1.0
pred[pred<0.0] = 0.0

pred = pred.reshape(-1, N, N)

diff = pred.reshape(-1, N*N) - test_data_out
mse = np.mean((diff)**2)
print("test error: {0}".format(mse))

difference = np.abs(diff).reshape(-1, N, N)

show_results({"pred": pred, "orig": test_data[0], "diff": difference})

print("done.")
