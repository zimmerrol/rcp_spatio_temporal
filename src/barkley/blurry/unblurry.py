import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
from barkley_helper import *
import argparse

N = 150
ndata = 10000
testLength = 2000
trainLength = ndata - testLength

def setupArrays():
    #TODO: Correct the array dimensions!
    global shared_training_data_base, shared_test_data_base, prediction_base
    global shared_training_data, shared_test_data, prediction

    shared_training_data_base = multiprocessing.Array(ctypes.c_double, 2*trainLength*N*N)
    shared_training_data = np.ctypeslib.as_array(shared_training_data_base.get_obj())
    shared_training_data = shared_training_data.reshape(2, trainLength, N, N)

    shared_test_data_base = multiprocessing.Array(ctypes.c_double, 2*testLength*N*N)
    shared_test_data = np.ctypeslib.as_array(shared_test_data_base.get_obj())
    shared_test_data = shared_test_data.reshape(2, testLength, N, N)

    prediction_base = multiprocessing.Array(ctypes.c_double, testLength*N*N)
    prediction = np.ctypeslib.as_array(prediction_base.get_obj())
    prediction = prediction.reshape(testLength, N, N)
setupArrays()

def parseArguments():
    global id, predictionMode, reverseDirection

    id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: NN, RBF, ESN")
    parser.add_argument('direction', default="vu", nargs=1, type=str, help="vu: v -> u, otherwise: u -> v")
    args = parser.parse_args()

    reverseDirection = args.direction[0] != "vu"
    if args.mode[0] not in ["ESN", "NN", "RBF"]:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(args.mode[0]))
    else:
        predictionMode = args.mode[0]

    print("Prediction via {0}: {1}".format(predictionMode, "u -> v" if reverseDirection else "v -> u"))
parseArguments()

def setupConstants():
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
        ddim = [3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5][id-1]
        k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5][id-1]
        sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]
        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ndata, sigma, sigma_skip, ddim, k))

    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))

    eff_sigma = int(np.ceil(sigma/sigma_skip))
    patch_radius = sigma//2
setupConstants()

def preparePredicter(y, x):
    if (predictionMode == "ESN"):
        if (y < patch_radius or y >= N-patch_radius or x < patch_radius or x >= N-patch_radius):
            #frame
            predicter = ESN(n_input = 1, n_output = 1, n_reservoir = n_units,
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

def fit_predict_pixel(y, x, running_index, training_data, test_data, predicter):
    training_data_in = training_data[1][:, y - patch_radius:y + patch_radius+1, x - patch_radius:x + patch_radius+1][:, ::sigma_skip, ::sigma_skip].reshape(-1, eff_sigma*eff_sigma)
    training_data_out = training_data[0][:, y, x].reshape(-1, 1)

    test_data_in = test_data[1][:, y - patch_radius:y + patch_radius+1, x - patch_radius:x + patch_radius+1][:, ::sigma_skip, ::sigma_skip].reshape(-1, eff_sigma*eff_sigma)
    test_data_out = test_data[0][:, y, x].reshape(-1, 1)

    predicter.fit(training_data_in, training_data_out)
    pred = predicter.predict(test_data_in)
    pred = pred.ravel()

    return pred

def fit_predict_frame_pixel(y, x, running_index, training_data, test_data, predicter):
    ind_y, ind_x = y, x

    min_border_distance = np.min([y, x, N-1-y, N-1-x])

    training_data_in = training_data[1][:, y - min_border_distance:y + min_border_distance+1, x - min_border_distance:x + min_border_distance+1].reshape(-1, int((2*min_border_distance+1)**2))
    training_data_out = training_data[0][:, y, x].reshape(-1, 1)

    test_data_in = test_data[1][:, y - min_border_distance:y + min_border_distance+1, x - min_border_distance:x + min_border_distance+1].reshape(-1, int((2*min_border_distance+1)**2))
    test_data_out = test_data[0][:, y, x].reshape(-1, 1)

    predicter.fit(training_data_in, training_data_out)
    pred = predicter.predict(test_data_in)
    pred = pred.ravel()

    return pred

def get_prediction_init(q):
    get_prediction.q = q

def get_prediction(data, def_param=(shared_training_data, shared_test_data)):
    y, x, running_index = data

    predicter = preparePredicter(y, x)
    pred = None
    if (y >= patch_radius and y < N-patch_radius and x >= patch_radius and x < N-patch_radius):
        #inner point
        pred = fit_predict_pixel(y, x, running_index, shared_training_data, shared_test_data, predicter)

    else:
        #frame
        pred = fit_predict_frame_pixel(y, x, running_index, shared_training_data, shared_test_data, predicter)

    get_prediction.q.put((y, x, pred))

def processThreadResults(threadname, q, numberOfWorkers, numberOfResults):
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

    if (os.path.exists("../cache/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
        print("generating data...")
        data = generate_data(ndata, 20000, 5, Ngrid=N) #20000 was 50000
        np.save("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
        print("generating finished")
    else:
        print("loading data...")
        data = np.load("../cache/raw/{0}_{1}.dat.npy".format(ndata, N))

        print("loading finished")

    training_data = data[:trainLength]
    test_data = data[trainLength:]

    shared_training_data[0, :, :, :] = training_data[:]
    shared_test_data[0, :, :, :] = test_data[:]

    print("blurring...")

    for t in range(trainLength):
        shared_training_data[1, t, :, :] = gaussian_filter(training_data[t], sigma=9.0)

    for t in range(testLength):
        shared_test_data[1, t, :, :] = gaussian_filter(test_data[t], sigma=9.0)

    show_results([("Orig", shared_training_data[0]), ("Blurred", shared_training_data[1])])

    print("blurring finished")

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
    processProcessResultsThread = Process(target=processThreadResults, args=("processProcessResultsThread", queue, 26, len(jobs)) )
    processProcessResultsThread.start()
    results = pool.map(get_prediction, jobs)
    pool.close()

    processProcessResultsThread.join()

    print("finished fitting")

    prediction[prediction < 0.0] = 0.0
    prediction[prediction > 1.0] = 1.0

    diff = (test_data-prediction)
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    print("inner test error: {0}".format(np.mean((diff[:, patch_radius:N-patch_radius, patch_radius:N-patch_radius])**2)))

    show_results([("Source", shared_test_data[1]), ("Orig", shared_test_data[0]), ("Pred", prediction), ("Diff", diff)])

if __name__== '__main__':
    mainFunction()
