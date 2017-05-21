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
ndata = 30000
testLength = 2000
trainLength = 15000

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
    global innerSize, halfInnerSize, borderSize, center, rightBorderAdd

    #there is a difference between odd and even numbers for the innerSize
    #odd size  => there is a center point and the left and the right area without this center are even spaced
    #even size => right and left half of the square are even spaced

    """
    even      odd
    aaaaaaaa  aaaaaaaaa
    a┌────┐a  a┌─────┐a
    a│ooxx│a  a│oo0xx│a
    a│ooxx│a  a│oo0xx│a
    a│ooxx│a  a│oo0xx│a
    a│ooxx│a  a│oo0xx│a
    a└────┘a  a│oo0xx│a
    aaaaaaaa  a└─────┘a
              aaaaaaaaa
    """

    print("Using parameters:")

    if (predictionMode == "ESN"):
        n_units = [50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400][id-1]
        regression_parameter = [5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,  5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,  5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,  5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5][id-1]
        innerSize = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        borderSize = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]

        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}".format(ndata, innerSize, borderSize, n_units, regression_parameter))
    elif (predictionMode == "NN"):
        ddim = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][id-1]
        k = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
             4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
             4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][id-1]
        innerSize = [4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,][id-1]
        borderSize = [1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,
                      1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,
                      1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,][id-1]

        print("\t ndata \t = {0} \n\t innerSize \t = {1}\n\t borderSize \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ndata, innerSize, borderSize, ddim, k))
    elif (predictionMode == "RBF"):
        basisPoints = 100

        ddim = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][id-1]
        width = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
             3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
             3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][id-1]
        innerSize = [4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146,146, 148,148,][id-1]
        borderSize = [1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,
                      1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,
                      1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,][id-1]
    print("\t ndata \t = {0} \n\t innerSize \t = {1}\n\t borderSize \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ndata, innerSize, borderSize, ddim, k))

    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))

    halfInnerSize = int(np.floor(innerSize / 2))
    borderSize = 1
    center = N//2
    rightBorderAdd = 1 if innerSize != 2*halfInnerSize else 0
setup_constants()

def setup_arrays():
    global shared_input_data_base, shared_data_base, prediction_base
    global shared_input_data, shared_data, prediction

    if (predictionMode in ["NN", "RBF"]):
        shared_input_data_base = multiprocessing.Array(ctypes.c_double, ddim*ndata*2*borderSize*(innerSize+(innerSize+2*borderSize)))
        shared_input_data = np.ctypeslib.as_array(shared_input_data_base.get_obj())
        shared_input_data = shared_input_data.reshape(ndata, -1)
    else:
        shared_input_data_base = multiprocessing.Array(ctypes.c_double, ndata*2*borderSize*(innerSize+(innerSize+2*borderSize)))
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

    if (direction == "u"):
        if (os.path.exists("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)) == False):
            data = bh.generate_data(N, 20000, 5, Ngrid=Ngrid)
            np.save("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
    else:
        if (os.path.exists("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)) == False):
            data = mh.generate_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))

    data = data[0,:]

    input_y, input_x, output_y, output_x = create_patch_indices(
                                                (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                                (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                                (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
                                                (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
                                            )
    input_data = data[:, input_y, input_x].reshape(ndata, -1)
    if (predictionMode in ["NN", "RBF"]):
        shared_input_data[:] = create_1d_delay_coordinates(input_data, delay_dimension=ddim, tau=tau[direction]).reshape((ndata, -1))
    else:
        shared_input_data[:] = input_data[:]

    shared_data[:] = data[:]
    prediction[:] = data[trainLength:trainLength+testLength]
    prediction[:, output_y, output_x] = 0.0

def prepare_predicter(y, x):
    if (predictionMode == "ESN"):
        predicter = ESN(n_input = shared_input_data.shape[1], n_output = 1, n_reservoir = n_units,
                    weight_generation = "advanced", leak_rate = 0.20, spectral_radius = 0.1,
                    random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[regression_parameter], solver = "lsqr")
    elif (predictionMode == "NN"):
        predicter = NN(k=k)
    elif (predictionMode == "RBF"):
        predicter = RBF(sigma=5.0, basisQuota=0.05)
    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))

    return predicter

def fit_predict_pixel(y, x, predicter, def_param=(shared_input_data, shared_data)):
    training_data_in = shared_input_data[:trainLength]
    test_data_in = shared_input_data[trainLength:trainLength+testLength]
    training_data_out = shared_data[:trainLength, y, x].reshape(-1, 1)
    test_data_out = shared_data[trainLength:trainLength+testLength, y, x].reshape(-1, 1)

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

    input_y, input_x, output_y, output_x = create_patch_indices(
                                            (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                            (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                            (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
                                            (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
                                        )
    jobs = []
    for i in range(len(output_y)):
        jobs.append((output_y[i], output_x[i]))

    print("fitting...")
    processProcessResultsThread = Process(target=process_thread_results, args=(queue, len(jobs)) )
    processProcessResultsThread.start()
    results = pool.map(get_prediction, jobs)
    pool.close()

    processProcessResultsThread.join()

    print("finished fitting")

    prediction[prediction < 0.0] = 0.0
    prediction[prediction > 1.0] = 1.0

    diff = (shared_data[trainLength:trainLength+testLength]-prediction)
    mse = np.mean((diff[:, output_y, output_x])**2)
    print("test error: {0}".format(mse))

    show_results([("Orig", shared_data[trainLength:trainLength+testLength]), ("Pred", prediction), ("Diff", diff)])

if __name__== '__main__':
    mainFunction()
