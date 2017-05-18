import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, os.path.join(grandparentdir, "barkley"))
sys.path.insert(0, os.path.join(grandparentdir, "mitchell"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import gaussian_filter
import progressbar
import dill as pickle

from ESN import *
from RBF import *
from NN import *

from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import multiprocessing
import ctypes
from multiprocessing import process

from helper import *
import barkley_helper as bh
import mitchell_helper as mh
import argparse

tau = {"u" : 32, "v" : 119}
N = 150
ndata = 10000
testLength = 2000
trainLength = ndata - testLength

def setup_arrays():
    global shared_input_data_base, shared_output_data_base, prediction_base
    global shared_input_data, shared_output_data, prediction

    shared_input_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_input_data = np.ctypeslib.as_array(shared_input_data_base.get_obj())
    shared_input_data = shared_input_data.reshape(ndata, N, N)

    shared_output_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_output_data = np.ctypeslib.as_array(shared_output_data_base.get_obj())
    shared_output_data = shared_output_data.reshape(ndata, N, N)

    prediction_base = multiprocessing.Array(ctypes.c_double, testLength*N*N)
    prediction = np.ctypeslib.as_array(prediction_base.get_obj())
    prediction = prediction.reshape(testLength, N, N)
setup_arrays()

def parse_arguments():
    global id, predictionMode, direction

    id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: NN, RBF, ESN")
    parser.add_argument('direction', default="u", nargs=1, type=str, help="u: unblur u, v: unblurr v")
    args = parser.parse_args()

    if args.direction[0] not in ["u", "v"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        direction = args.direction[0]

    if args.mode[0] not in ["ESN", "NN", "RBF"]:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(args.mode[0]))
    else:
        predictionMode = args.mode[0]

    print("Prediction via {0}".format(predictionMode))
parse_arguments()

def setup_constants():
    global k, ddim, sigma, sigma_skip, eff_sigma, patch_radius, n_units, regression_parameter

    print("Using parameters:")

    if (predictionMode == "ESN"):
        n_units = [50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400][id-1]
        regression_parameter = [5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,  5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,  5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,  5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5][id-1]
        sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]

        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}".format(ndata, sigma, sigma_skip, n_units, regression_parameter))
    elif (predictionMode == "NN"):
        ddim = [3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5][id-1]
        k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5][id-1]
        sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]

        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ndata, sigma, sigma_skip, ddim, k))
    elif (predictionMode == "RBF"):
        ddim = [3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5][id-1]
        width = [.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9][id-1]
        sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]
        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t width \t = {4}".format(ndata, sigma, sigma_skip, ddim, width))

    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))

    eff_sigma = int(np.ceil(sigma/sigma_skip))
    patch_radius = sigma//2
setup_constants()

def generate_data(N, trans, sample_rate, Ngrid):
    data = None

    if (direction == "u"):
        if (os.path.exists("../../cache/barkley/raw/{0}_{1}.dat.npy".format(N, Ngrid)) == False):
            data = bh.generate_data(N, 20000, 5, Ngrid=Ngrid)
            np.save("../../cache/barkley/raw/{0}_{1}.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/barkley/raw/{0}_{1}.dat.npy".format(N, Ngrid))
    else:
        if (os.path.exists("../../cache/mitchell/raw/{0}_{1}.dat.npy".format(N, Ngrid)) == False):
            data = mh.generate_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../../cache/mitchell/raw/{0}_{1}.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/mitchell/raw/{0}_{1}.dat.npy".format(N, Ngrid))

    shared_input_data[:] = data[:]

    print("blurring...")
    for t in range(ndata):
        shared_output_data[t, :, :] = gaussian_filter(shared_output_data[t], sigma=9.0)
    show_results([("Blurred", shared_input_data[:trainLength]), ("Orig", shared_output_data[:trainLength])])
    print("blurring finished")

def prepare_predicter(y, x):
    if (predictionMode == "ESN"):
        if (y < patch_radius or y >= N-patch_radius or x < patch_radius or x >= N-patch_radius):
            #frame
            min_border_distance = np.min([y, x, N-1-y, N-1-x])

            predicter = ESN(n_input = int((2*min_border_distance+1)**2), n_output = 1, n_reservoir = n_units,
                    weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
                    random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[regression_parameter], solver = "lsqr")
        else:
            #inner
            predicter = ESN(n_input = eff_sigma*eff_sigma, n_output = 1, n_reservoir = n_units,
                        weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
                        random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[regression_parameter], solver = "lsqr")

    elif (predictionMode == "NN"):
        predicter = NN(k=k)
    elif (predictionMode == "RBF"):
        predicter = RBF(sigma=5.0, basisQuota=0.05)
    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))

    return predicter

def prepare_fit_data(y, x, pr, skip, def_param=(shared_input_data, shared_output_data)):
    if (predictionMode in ["NN", "RBF"]):
        delayed_patched_input_data = create_2d_delay_coordinates(shared_input_data[:, y-pr:y+pr+1, x-pr:x+pr+1][:, ::skip, ::skip], ddim, tau=tau[direction])
        delayed_patched_input_data = delayed_patched_input_data.reshape(ndata, -1)

        delayed_patched_input_data_train = delayed_patched_input_data[:trainLength]
        delayed_patched_input_data_test = delayed_patched_input_data[trainLength:trainLength+testLength]

        training_data_in = delayed_patched_input_data_train.reshape(trainLength, -1)
        test_data_in = delayed_patched_input_data_test.reshape(testLength, -1)

        training_data_out = shared_output_data[:trainLength, y, x].reshape(-1,1)
        test_data_out = shared_output_data[trainLength:trainLength+testLength, y, x].reshape(-1,1)

    else:
        training_data_in = shared_input_data[:trainLength][:, y - pr:y + pr+1, x - pr:x + pr+1][:, ::skip, ::skip].reshape(trainLength, -1)
        test_data_in = shared_input_data[trainLength:trainLength+testLength][:, y - pr:y + pr+1, x - pr:x + pr+1][:, ::skip, ::skip].reshape(testLength, -1)

        training_data_out = shared_output_data[:trainLength][:, y, x].reshape(-1, 1)
        test_data_out = shared_output_data[trainLength:trainLength+testLength][:, y, x].reshape(-1, 1)

    return training_data_in, test_data_in, training_data_out, test_data_out

def fit_predict_pixel(y, x, running_index, predicter):
    training_data_in, test_data_in, training_data_out, test_data_out = prepare_fit_data(y, x, patch_radius, sigma_skip)

    predicter.fit(training_data_in, training_data_out)
    pred = predicter.predict(test_data_in)
    pred = pred.ravel()

    return pred

def fit_predict_frame_pixel(y, x, running_index, predicter):
    min_border_distance = np.min([y, x, N-1-y, N-1-x])
    training_data_in, test_data_in, training_data_out, test_data_out = prepare_fit_data(y, x, min_border_distance, 1)

    predicter.fit(training_data_in, training_data_out)
    pred = predicter.predict(test_data_in)
    pred = pred.ravel()

    return pred

def get_prediction_init(q):
    get_prediction.q = q

def get_prediction(data):
    y, x, running_index = data

    predicter = prepare_predicter(y, x)
    pred = None
    if (y >= patch_radius and y < N-patch_radius and x >= patch_radius and x < N-patch_radius):
        #inner point
        pred = fit_predict_pixel(y, x, running_index, predicter)

    else:
        #frame
        pred = fit_predict_frame_pixel(y, x, running_index, predicter)

    get_prediction.q.put((y, x, pred))

def process_thread_results(q, numberOfResults):
    global prediction

    bar = progressbar.ProgressBar(max_value=numberOfResults, redirect_stdout=True, poll_interval=0.0001)
    bar.update(0)

    finishedWorkers = 0
    finishedResults = 0

    while True:
        if (finishedResults == numberOfResults):
            return

        newData= q.get()
        finishedResults += 1
        ind_y, ind_x, data = newData

        prediction[:, ind_y, ind_x] = data

        bar.update(finishedResults)

def mainFunction():
    global shared_training_data, shared_test_data

    generate_data(ndata, 20000, 50, Ngrid=N)

    queue = Queue() # use manager.queue() ?
    print("preparing threads...")
    pool = Pool(processes=16, initializer=get_prediction_init, initargs=[queue,])

    modifyDataProcessList = []
    jobs = []
    index = 0

    for y in range(N):
        for x in range(N):
            jobs.append((y, x, index))

    print("fitting...")
    processProcessResultsThread = Process(target=process_thread_results, args=(queue, len(jobs)) )
    processProcessResultsThread.start()
    results = pool.map(get_prediction, jobs)
    pool.close()

    processProcessResultsThread.join()

    print("finished fitting")

    prediction[prediction < 0.0] = 0.0
    prediction[prediction > 1.0] = 1.0

    diff = (shared_output_data[trainLength:]-prediction)
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    print("inner test error: {0}".format(np.mean((diff[:, patch_radius:N-patch_radius, patch_radius:N-patch_radius])**2)))

    viewData = [("Source", shared_input_data[trainLength:]), ("Orig", shared_output_data[trainLength:]), ("Pred", prediction), ("Diff", diff)]
    #show_results(viewData)

    model = "barkley" if direction in ["uv", "vu"] else "mitchell"

    if (predictionMode == "NN"):
        f = open("../../cache/{0}/viewdata/unblur_{1}/{2}_viewdata_{3}_{4}_{5}_{6}_{7}.dat".format(model, direction, predictionMode.lower(), trainLength, sigma, sigma_skip, ddim, k), "wb")
    elif (predictionMode == "RBF"):
        f = open("../../cache/{0}/viewdata/unblur_{1}/{2}_viewdata_{3}_{4}_{5}_{6}_{7}_{8}.dat".format(model, direction, predictionMode.lower(), trainLength, sigma, sigma_skip, ddim, width, basisPoints), "wb")
    else:
        f = open("../../cache/{0}/viewdata/unblur_{1}/{2}_viewdata_{3}_{4}_{5}_{6}_{7}.dat".format(model, direction, predictionMode.lower(), trainLength, sigma, sigma_skip, regression_parameter, n_units), "wb")
    pickle.dump(viewData, f)
    f.close()

    print("done")

if __name__== '__main__':
    mainFunction()
