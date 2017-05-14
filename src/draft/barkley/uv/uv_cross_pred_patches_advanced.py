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
import dill as pickle
from helper import *

N = 150
ndata = 10000
sigma = 5
n_units = 50

if (os.path.exists("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N)) == False):
    print("generating data...")
    data = generate_data(ndata, 50000, 5, Ngrid=N)
    np.save("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N))
    print("loading finished")


generate_new = False
if (os.path.exists("../cache/esn/uv/cross_pred_patches_advanced{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units)) == False):
    generate_new = True

if (generate_new):
    print("setting up...")
    esn = ESN(n_input = sigma*sigma, n_output = 1, n_reservoir = n_units,
            weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-1], solver = "lsqr")

    frameEsn = ESN(n_input = 4, n_output = 4, n_reservoir = n_units,
            weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-1], solver = "lsqr")

    last_states = np.empty(((N-2)*(N-2), n_units, 1))
    output_weights = np.empty(((N-2)*(N-2),sigma*sigma, sigma*sigma+1+n_units))

    frame_output_weights = np.empty(((N-2)*(N-2),2*2, 2*2+1+n_units))
else:
    print("loading existing model...")

    f = open("../cache/esn/uv/cross_pred_patches_advanced{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units), "rb")
    output_weights = pickle.load(f)
    frame_output_weights = pickle.load(f)
    last_states = pickle.load(f)
    esn = pickle.load(f)
    frameEsn = pickle.load(f)
    f.close()

training_data = data[:, :ndata-2000]
test_data = data[:, ndata-2000:]
prediction = np.ones((2000, N, N))

def fit_predict_pixel(y, x, running_index, prediction, last_states, output_weights, training_data, test_data):
    training_data_in = training_data[1][:, y-2:y+3, x-2:x+3].reshape(-1, 5*5)
    training_data_out = training_data[0][:, y, x].reshape(-1, 1)

    test_data_in = test_data[1][:, y-2:y+3, x-2:x+3].reshape(-1, 5*5)
    test_data_out = test_data[0][:, y, x].reshape(-1, 1)

    if (generate_new):
        train_error = esn.fit(training_data_in, training_data_out, verbose=0)
        print("train error: {0}".format(train_error))

        last_states[running_index] = esn._x
        output_weights[running_index] = esn._W_out
    else:
        esn._x = last_states[running_index]
        esn._W_out = output_weights[running_index]

    pred = esn.predict(test_data_in, verbose=0)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    diff = test_data_out-pred
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    bar.update(running_index+1)

    prediction[:, y, x] =  pred[:,0]

def fit_predict_frame_pixel(y, x, running_index, prediction, last_states, output_weights, training_data, test_data, progressCounter):
    training_data_in = training_data[1][:, y:y+2, x:x+2].reshape(-1, 2*2)
    training_data_out = training_data[0][:, y:y+2, x:x+2].reshape(-1, 2*2)

    test_data_in = test_data[1][:, y:y+2, x:x+2].reshape(-1, 2*2)
    test_data_out = test_data[0][:, y:y+2, x:x+2].reshape(-1, 2*2)

    if (generate_new):
        train_error = frameEsn.fit(training_data_in, training_data_out, verbose=0)
        print("train error: {0}".format(train_error))

        last_states[running_index] = frameEsn._x
        frame_output_weights[running_index-(N-4)*(N-4)] = frameEsn._W_out
    else:
        frameEsn._x = last_states[running_index]
        frameEsn._W_out = frame_output_weights[running_index-(N-4)*(N-4)]

    pred = frameEsn.predict(test_data_in, verbose=0)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    diff = test_data_out-pred
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    bar.update(running_index+1)

    prediction[:, ind_y, ind_x] = pred

print("fitting...")
bar = progressbar.ProgressBar(max_value=(N-4)*(N-4) + N//2*2 + (N-2)//2*2, redirect_stdout=True, poll_interval=0.0001)
bar.update(0)

progressCounter = (N-4)*(N-4)
for y in range(2, N-2):
    for x in range(2, N-2):
        fit_predict_pixel(y, x, (y-2)*(N-4) + (x-2), prediction, last_states, output_weights, training_data, test_data)

for y in range(0, N, 2):
    fit_predict_frame_pixel(y, 0, (N-4)*(N-4) + y//2*2, prediction, last_states, output_weights, training_data, test_data, progressCounter)
    progressCounter += 1

    fit_predict_frame_pixel(y, N-2, (N-4)*(N-4) + y//2*2 + 1, prediction, last_states, output_weights, training_data, test_data, progressCounter)
    progressCounter += 1

for x in range(2, N-2, 2):
    fit_predict_frame_pixel(0, x, (N-4)*(N-4) + N//2*2 + (x-2)//2*2, prediction, last_states, output_weights, training_data, test_data, progressCounter)
    progressCounter += 1

    fit_predict_frame_pixel(N-2, x,  (N-4)*(N-4) + N//2*2 + (x-2)//2*2 + 1, prediction, last_states, output_weights, training_data, test_data, progressCounter)
    progressCounter += 1

bar.finish()

if (generate_new):
    print("saving model...")

    f = open("../cache/esn/uv/cross_pred_patches_advanced{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units), "wb")
    pickle.dump(output_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(frame_output_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(last_states, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(esn, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

diff = test_data[0]-prediction
mse = np.mean((diff)**2)
print("test error: {0}".format(mse))

difference = np.abs(diff)

show_results({"pred":prediction, "orig u" : test_data[0], "orig v" : test_data[1], "diff": difference})

print("done.")
