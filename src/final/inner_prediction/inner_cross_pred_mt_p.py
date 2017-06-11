# This Python file uses the following encoding: utf-8

"""
     Performs a calculation of the inner points using just the outer points for the first variable of the model. All constants etc. must be set before
     by the corresponding *_p.py file.
"""

#We require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
from multiprocessing import Process, Queue, Pool, process #
import multiprocessing
import ctypes
import progressbar
import dill as pickle
import numpy as np

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../barkley'))
sys.path.insert(1, os.path.join(sys.path[0], '../../mitchell'))

from ESN import ESN
from RBF import RBF
from NN import NN

import helper as hp
import barkley_helper as bh
import mitchell_helper as mh
import argparse

#set the temporary buffer for the multiprocessing module manually to the shm
#to solve "no enough space"-problems
process.current_process()._config['tempdir'] =  '/dev/shm/'

tau = {"u" : 32, "v" : 119}
N = 150
ndata = 30000
testLength = 2000
trainLength = 15000

#will be set by the *_p.py file
direction, prediction_mode, patch_radius, eff_sigma, sigma, sigma_skip, ddim = None, None, None, None, None, None
n_units, spectral_radius, leaking_rate, random_seed, noise_level, regression_parameter, sparseness = None, None, None, None, None, None, None
border_size, inner_size, center, half_inner_size, right_border_add, basis_points, width, k = None, None, None, None, None, None, None, None, None

def setup_arrays():
    global shared_input_data_base, shared_data_base, prediction_base
    global shared_input_data, shared_data, prediction

    if prediction_mode in ["NN", "RBF"]:
        shared_input_data_base = multiprocessing.Array(ctypes.c_double, ddim*ndata*2*border_size*(inner_size+(inner_size+2*border_size)))
        shared_input_data = np.ctypeslib.as_array(shared_input_data_base.get_obj())
        shared_input_data = shared_input_data.reshape(ndata, -1)
    else:
        shared_input_data_base = multiprocessing.Array(ctypes.c_double, ndata*2*border_size*(inner_size+(inner_size+2*border_size)))
        shared_input_data = np.ctypeslib.as_array(shared_input_data_base.get_obj())
        shared_input_data = shared_input_data.reshape(ndata, -1)

    shared_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_data = np.ctypeslib.as_array(shared_data_base.get_obj())
    shared_data = shared_data.reshape(ndata, N, N)

    prediction_base = multiprocessing.Array(ctypes.c_double, testLength*N*N)
    prediction = np.ctypeslib.as_array(prediction_base.get_obj())
    prediction = prediction.reshape(testLength, N, N)
setup_arrays()

def generate_data(N, trans, sample_rate, Ngrid, def_param=(shared_input_data, shared_data)):
    data = None

    if direction == "u":
        if not os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)):
            data = bh.generate_data(N, 20000, 5, Ngrid=Ngrid)
            np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
    else:
        if not os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)):
            data = mh.generate_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))

    data = data[0, :]

    input_y, input_x, output_y, output_x = hp.create_patch_indices(
        (center - (half_inner_size+border_size), center + (half_inner_size+border_size) + right_border_add),
        (center - (half_inner_size+border_size), center + (half_inner_size+border_size) + right_border_add),
        (center - (half_inner_size), center + (half_inner_size) + right_border_add),
        (center - (half_inner_size), center + (half_inner_size) + right_border_add))

    input_data = data[:, input_y, input_x].reshape(ndata, -1)
    if prediction_mode in ["NN", "RBF"]:
        shared_input_data[:] = hp.create_1d_delay_coordinates(input_data, delay_dimension=ddim, tau=tau[direction]).reshape((ndata, -1))
    else:
        shared_input_data[:] = input_data[:]

    shared_data[:] = data[:]
    prediction[:] = data[trainLength:trainLength+testLength]
    prediction[:, output_y, output_x] = 0.0

def prepare_predicter(y, x):
    if prediction_mode == "ESN":
        predicter = ESN(n_input=shared_input_data.shape[1], n_output=1, n_reservoir=n_units,
                        weight_generation="advanced", leak_rate=leaking_rate, spectral_radius=spectral_radius,
                        random_seed=random_seed, noise_level=noise_level, sparseness=sparseness,
                        regression_parameters=[regression_parameter], solver="lsqr")
    elif prediction_mode == "NN":
        predicter = NN(k=k)
    elif prediction_mode == "RBF":
        predicter = RBF(sigma=width, basisPoints=basis_points)
    else:
        raise ValueError("No valid prediction_mode choosen! (Value is now: {0})".format(prediction_mode))

    return predicter

def fit_predict_pixel(y, x, predicter, def_param=(shared_input_data, shared_data)):
    training_data_in = shared_input_data[:trainLength]
    test_data_in = shared_input_data[trainLength:trainLength+testLength]
    training_data_out = shared_data[:trainLength, y, x].reshape(-1, 1)

    predicter.fit(training_data_in, training_data_out)
    pred = predicter.predict(test_data_in)
    pred = pred.ravel()

    return pred

def get_prediction_init(q):
    get_prediction.q = q

def get_prediction(data):
    y, x = data

    predicter = prepare_predicter(y, x)
    pred = fit_predict_pixel(y, x, predicter)
    get_prediction.q.put((y, x, pred))

def process_thread_results(q, nb_results):
    global prediction

    bar = progressbar.ProgressBar(max_value=nb_results, redirect_stdout=True, poll_interval=0.0001)
    bar.update(0)

    finished_results = 0

    while True:
        if finished_results == nb_results:
            return

        new_data = q.get()
        finished_results += 1
        ind_y, ind_x, data = new_data

        prediction[:, ind_y, ind_x] = data

        bar.update(finished_results)

def mainFunction():
    generate_data(ndata, 20000, 50, Ngrid=N)

    queue = Queue() # use manager.queue() ?
    print("preparing threads...")
    pool = Pool(processes=16, initializer=get_prediction_init, initargs=[queue,])

    _, _, output_y, output_x = hp.create_patch_indices(
        (center - (half_inner_size+border_size), center + (half_inner_size+border_size) + right_border_add),
        (center - (half_inner_size+border_size), center + (half_inner_size+border_size) + right_border_add),
        (center - (half_inner_size), center + (half_inner_size) + right_border_add),
        (center - (half_inner_size), center + (half_inner_size) + right_border_add))

    jobs = []
    for i in range(len(output_y)):
        jobs.append((output_y[i], output_x[i]))

    print("fitting...")
    process_results_process = Process(target=process_thread_results, args=(queue, len(jobs)))
    process_results_process.start()
    pool.map(get_prediction, jobs)
    pool.close()

    process_results_process.join()

    print("finished fitting")

    prediction[prediction < 0.0] = 0.0
    prediction[prediction > 1.0] = 1.0

    diff = (shared_data[trainLength:trainLength+testLength]-prediction)
    mse = np.mean((diff[:, output_y, output_x])**2)
    print("test error: {0}".format(mse))

    model = "barkley" if direction == "u" else "mitchell"
    view_data = [("Orig", shared_data[trainLength:trainLength+testLength]), ("Pred", prediction), ("Diff", diff)]

    if prediction_mode == "NN":
        output_file = open("../../cache/{0}/viewdata/inner_cross_pred/{1}/{2}_viewdata_{3}_{4}_{5}_{6}_{7}.dat".format(
            model, direction, prediction_mode.lower(), trainLength, inner_size, border_size, ddim, k), "wb")
    elif prediction_mode == "RBF":
        output_file = open("../../cache/{0}/viewdata/inner_cross_pred/{1}/{2}_viewdata_{3}_{4}_{5}_{6}_{7}_{8}.dat".format(
            model, direction, prediction_mode.lower(), trainLength, inner_size, border_size, ddim, width, basis_points), "wb")
    else:
        output_file = open("../../cache/{0}/viewdata/inner_cross_pred/{1}/{2}_viewdata_{3}_{4}_{5}_{6}_{7}.dat".format(
            model, direction, prediction_mode.lower(), trainLength, inner_size, border_size, regression_parameter, n_units), "wb")
    pickle.dump(view_data, output_file)
    output_file.close()

if __name__ == '__main__':
    mainFunction()
