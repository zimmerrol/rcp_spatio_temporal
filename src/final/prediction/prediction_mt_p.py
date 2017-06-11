"""
    Performs a prediction of the u/v variable. All constants etc. must be set before by the corresponding *_p.py file.
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
import helper as hp
import barkley_helper as bh
import mitchell_helper as mh


if __name__ == '__main__':
    print("Do not call prediction_mt_p.py on its own. Set the constants and arguments according to prediction_p.py" \
            "and then call the mainFunction here.")
    exit()


#set the temporary buffer for the multiprocessing module manually to the shm
#to solve "no enough space"-problems
process.current_process()._config['tempdir'] = '/dev/shm/'

tau = {"uv" : 32, "vu" : 32, "vh" : 119, "hv" : 119}
N = 150
ndata = 30000
trainLength = 15000
testLength = 2000

useInputScaling = False

predictionLength = 100

#will be set by the *_p.py file
direction, predictionMode, patch_radius, eff_sigma, sigma, sigma_skip = None, None, None, None, None, None
n_units, spectral_radius, leaking_rate, random_seed, noise_level, regression_parameter, sparseness = None, None, None, None, None, None, None

def setup_arrays():
    global shared_input_data_base, shared_output_data_base, shared_prediction_base
    global shared_input_data, shared_output_data, shared_prediction

    shared_input_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_input_data = np.ctypeslib.as_array(shared_input_data_base.get_obj())
    shared_input_data = shared_input_data.reshape(-1, N, N)

    shared_output_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_output_data = np.ctypeslib.as_array(shared_output_data_base.get_obj())
    shared_output_data = shared_output_data.reshape(-1, N, N)

    shared_prediction_base = multiprocessing.Array(ctypes.c_double, testLength*N*N)
    shared_prediction = np.ctypeslib.as_array(shared_prediction_base.get_obj())
    shared_prediction = shared_prediction.reshape(-1, N, N)
setup_arrays()

def generate_data(N, Ngrid):
    data = None

    if direction == "u":
        if not os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)):
            data = bh.generate_uv_data(N, 50000, 5, Ngrid=Ngrid)
            np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
    else:
        if not os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)):
            data = mh.generate_vh_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))

    #fill the first ndata-predictionLength items with the real data and leave the last predictionLength items free
    #these items will never be used, and are only created to reduce the needed code
    shared_input_data[:-predictionLength] = data[0, :-predictionLength]
    shared_output_data[:-predictionLength] = data[0, predictionLength:]

def prepare_predicter(y, x, training_data_in, training_data_out):
    if y < patch_radius or y >= N-patch_radius or x < patch_radius or x >= N-patch_radius:
        #frame
        min_border_distance = np.min([y, x, N-1-y, N-1-x])
        input_dimension = int((2*min_border_distance+1)**2)
    else:
        #inner
        input_dimension = eff_sigma*eff_sigma

    input_scaling = None
    if useInputScaling:
        #approximate the input scaling using the MI
        input_scaling = hp.calculate_esn_mi_input_scaling(training_data_in, training_data_out[:, 0])

    predicter = ESN(n_input=input_dimension, n_output=1, n_reservoir=n_units,
                    weight_generation="advanced", leak_rate=leaking_rate, spectral_radius=spectral_radius,
                    random_seed=random_seed, noise_level=noise_level, sparseness=sparseness, input_scaling=input_scaling,
                    regression_parameters=[regression_parameter], solver="lsqr")

    return predicter

def get_prediction(data):
    y, x = data

    pred = None
    if y < patch_radius or y >= N-patch_radius or x < patch_radius or x >= N-patch_radius:
        #frame
        pred = fit_predict_frame_pixel(y, x)
    else:
        #inner
        pred = fit_predict_inner_pixel(y, x)
    get_prediction.q.put((y, x, pred))

def prepare_fit_data(y, x, pr, skip, def_param=(shared_input_data, shared_output_data)):
    training_data_in = shared_input_data[:trainLength][:, y-pr:y+pr+1, x-pr:x+pr+1][:, ::skip, ::skip].reshape(trainLength, -1)
    test_data_in = shared_input_data[trainLength:trainLength+testLength][:, y-pr:y+pr+1, x-pr:x+pr+1][:, ::skip, ::skip].reshape(testLength, -1)

    training_data_out = shared_output_data[:trainLength][:, y, x].reshape(-1, 1)
    test_data_out = shared_output_data[trainLength:trainLength+testLength][:, y, x].reshape(-1, 1)

    return training_data_in, test_data_in, training_data_out, test_data_out

def fit_predict_frame_pixel(y, x, def_param=(shared_input_data, shared_output_data)):
    min_border_distance = np.min([y, x, N-1-y, N-1-x])
    training_data_in, test_data_in, training_data_out, _ = prepare_fit_data(y, x, min_border_distance, 1)

    predicter = prepare_predicter(y, x, training_data_in, training_data_out)
    predicter.fit(training_data_in, training_data_out)
    pred = predicter.predict(test_data_in)
    pred = pred.ravel()

    return pred

from numpy.linalg.linalg import LinAlgError
def fit_predict_inner_pixel(y, x, def_param=(shared_input_data, shared_output_data)):
    training_data_in, test_data_in, training_data_out, _ = prepare_fit_data(y, x, patch_radius, sigma_skip)

    predicter = prepare_predicter(y, x, training_data_in, training_data_out)
    try:
        predicter.fit(training_data_in, training_data_out)

        pred = predicter.predict(test_data_in)
        pred = pred.ravel()
    except LinAlgError:
        print("(y,x) = ({0},{1}) raised a SVD error".format(y, x))

        pred = np.zeros(testLength)

    return pred

def process_thread_results(q, numberOfResults, def_param=(shared_prediction, shared_output_data)):
    bar = progressbar.ProgressBar(max_value=numberOfResults, redirect_stdout=True, poll_interval=0.0001)
    bar.update(0)

    finished_results = 0

    while True:
        if finished_results == numberOfResults:
            return

        new_data = q.get()
        finished_results += 1
        ind_y, ind_x, data = new_data

        shared_prediction[:, ind_y, ind_x] = data

        bar.update(finished_results)

def get_prediction_init(q):
    get_prediction.q = q

def mainFunction():
    if trainLength + testLength > ndata:
        print("Please adjust the trainig and testing phase length!")
        exit()

    generate_data(ndata, Ngrid=N)

    queue = Queue() # use manager.queue() ?

    pool = Pool(processes=16, initializer=get_prediction_init, initargs=[queue,])

    jobs = []
    for y in range(N):
        for x in range(N):
            jobs.append((y, x))

    process_results_process = Process(target=process_thread_results, args=(queue, len(jobs)))
    process_results_process.start()
    pool.map(get_prediction, jobs)
    pool.close()

    process_results_process.join()

    shared_prediction[shared_prediction < 0.0] = 0.0
    shared_prediction[shared_prediction > 1.0] = 1.0

    diff = (shared_output_data[trainLength:trainLength+testLength]-shared_prediction)
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    print("inner test error: {0}".format(np.mean((diff[:, patch_radius:N-patch_radius, patch_radius:N-patch_radius])**2)))

    view_data = [("Orig", shared_output_data[trainLength:trainLength+testLength]), ("Pred", shared_prediction),
                 ("Source", shared_input_data[trainLength:trainLength+testLength]), ("Diff", diff)]

    model = "barkley" if direction == "u" else "mitchell"

    output_file = open("../../cache/{0}/viewdata/predict_{1}/{2}_viewdata_{3}_{4}_{5}_{6}_{7}.dat".format(
        model, direction, predictionMode.lower(), trainLength, sigma, sigma_skip, regression_parameter, n_units), "wb")
    pickle.dump(view_data, output_file)
    output_file.close()

    hp.show_results(view_data)
    print("done")
