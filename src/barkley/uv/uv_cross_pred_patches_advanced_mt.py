# -*- coding: utf-8 -*-

#TODO: Use http://stackoverflow.com/questions/28821910/how-to-get-around-the-pickling-error-of-python-multiprocessing-without-being-in

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
import copy
from multiprocessing import Process, Queue, Pool
import multiprocessing
import ctypes

N = 150
ndata = 10000
sigma = 5
n_units = 100

def setupArrays():
    global shared_training_data_base, shared_test_data_base, prediction_base, last_states_base, output_weights_base, frame_output_weights_base
    global shared_training_data, shared_test_data, prediction, last_states, output_weights, frame_output_weights

    shared_training_data_base = multiprocessing.Array(ctypes.c_double, 2*8000*N*N)
    shared_training_data = np.ctypeslib.as_array(shared_training_data_base.get_obj())
    shared_training_data = shared_training_data.reshape(2, -1, N, N)

    shared_test_data_base = multiprocessing.Array(ctypes.c_double, 2*2000*N*N)
    shared_test_data = np.ctypeslib.as_array(shared_test_data_base.get_obj())
    shared_test_data = shared_test_data.reshape(2, -1, N, N)

    prediction_base = multiprocessing.Array(ctypes.c_double, 2000*N*N)
    prediction = np.ctypeslib.as_array(prediction_base.get_obj())
    prediction = prediction.reshape(-1, N, N)

    last_states_base = multiprocessing.Array(ctypes.c_double, (N-2)*(N-2)*n_units)
    last_states = np.ctypeslib.as_array(last_states_base.get_obj())
    last_states = last_states.reshape(-1, n_units, 1)

    output_weights_base = multiprocessing.Array(ctypes.c_double, sigma*sigma*(sigma*sigma+1+n_units))
    output_weights = np.ctypeslib.as_array(output_weights_base.get_obj())
    output_weights = output_weights.reshape(-1, sigma*sigma, sigma*sigma+1+n_units)

    frame_output_weights_base = multiprocessing.Array(ctypes.c_double, (N-2)*(N-2) * 2*2 * (2*2+1+n_units))
    frame_output_weights = np.ctypeslib.as_array(frame_output_weights_base.get_obj())
    frame_output_weights = frame_output_weights.reshape((N-2)*(N-2),2*2, 2*2+1+n_units)
setupArrays()

def generate_data(N, trans, sample_rate=1, Ngrid=100):
    Nx = Ngrid
    Ny = Ngrid
    deltaT = 1e-2
    epsilon = 0.08
    delta_x = 0.1
    D = 1/50
    h = D/delta_x**2
    print("h=" + str(h))
    #h = D over delta_x
    a = 0.75
    b = 0.06

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    bar = progressbar.ProgressBar(max_value=trans+N, redirect_stdout=True)

    for i in range(trans):
        sim.explicit_step(chaotic=True)
        bar.update(i)

    data = np.empty((2, N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step(chaotic=True)
        data[0, i] = sim._u
        data[1, i] = sim._v
        bar.update(i+trans)

    bar.finish()
    return data

def create_patch_indices(range_x, range_y):
    ind_x = np.tile(range(range_x[0], range_x[1]), range_y[1] - range_y[0])
    ind_y = np.repeat(range(range_y[0], range_y[1]), range_x[1] - range_x[0])

    return ind_y, ind_x

def fit_predict_pixel(y, x, running_index, last_states, output_weights, training_data, test_data, esn, generate_new):
    ind_y, ind_x = create_patch_indices((x - 2, x + 3), (y - 2, y + 3))

    training_data_in = training_data[1][:, ind_y, ind_x].reshape(-1, 5*5)
    training_data_out = training_data[0][:, y, x].reshape(-1, 1)

    test_data_in = test_data[1][:, ind_y, ind_x].reshape(-1, 5*5)
    test_data_out = test_data[0][:, y, x].reshape(-1, 1)

    if (generate_new):
        train_error = esn.fit(training_data_in, training_data_out, verbose=0)

        #last_states[running_index] = esn._x
        #output_weights[running_index] = esn._W_out
    else:
        esn._x = last_states[running_index]
        esn._W_out = output_weights[running_index]

    pred = esn.predict(test_data_in, verbose=0)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    return pred[:,0]

def fit_predict_frame_pixel(y, x, running_index, last_states, output_weights, training_data, test_data, esn, generate_new):
    ind_y, ind_x = y, x #create_patch_indices((x,x+2), (y-2, y))

    training_data_in = training_data[1][:, ind_y, ind_x].reshape(-1, 1*1)
    training_data_out = training_data[0][:, y, x].reshape(-1, 1*1)

    test_data_in = test_data[1][:, ind_y, ind_x].reshape(-1, 1*1)
    test_data_out = test_data[0][:, y, x].reshape(-1, 1*1)

    if (generate_new):
        train_error = esn.fit(training_data_in, training_data_out, verbose=0)
    else:
        esn._x = last_states[running_index]
        esn._W_out = frame_output_weights[running_index-(N-4)*(N-4)]

    pred = esn.predict(test_data_in, verbose=0)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    return pred[:, 0]

def get_prediction(data, def_param=(shared_training_data, shared_test_data, frame_output_weights, output_weights, last_states)):
    y, x, running_index = data

    pred = None
    if (y > 1 and y < N-2 and x > 1 and x < N-2):
        #inner point
        esn = ESN(n_input = sigma*sigma, n_output = 1, n_reservoir = n_units,
                    weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
                    random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-1], solver = "lsqr")


        pred = fit_predict_pixel(y, x, running_index, output_weights, last_states, shared_training_data, shared_test_data, esn, True)

    else:
        #frame
        esn = ESN(n_input = 1, n_output = 1, n_reservoir = n_units,
                weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
                random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-1], solver = "lsqr")

        pred = fit_predict_frame_pixel(y, x, running_index, frame_output_weights, last_states, shared_training_data, shared_test_data, esn, True)

    get_prediction.q.put((y, x, pred))

def processThreadResults(threadname, q, numberOfWorkers, numberOfResults, def_param=(shared_training_data, shared_test_data)):
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

def showResults(test_data, prediction, difference):
    i = 0
    pause = False
    image_mode = 0

    def update_new(data):
        nonlocal i

        if (image_mode == 0):
            mat.set_data(prediction[i])
            clb.set_clim(vmin=0, vmax=1)
            clb.draw_all()
        elif (image_mode == 1):
            mat.set_data(test_data[0, i])
            clb.set_clim(vmin=0, vmax=1)
            clb.draw_all()
        elif (image_mode == 2):
            mat.set_data(test_data[1, i])
            clb.set_clim(vmin=0, vmax=1)
            clb.draw_all()
        else:
            mat.set_data(difference[i])
            if (i < len(difference)-50 and i > 50):
                clb.set_clim(vmin=0, vmax=np.max(difference[i-50:i+50]))
            clb.draw_all()

        if (not pause):
            i = (i+1) % len(difference)
            sposition.set_val(i)
        return [mat]

    fig, ax = plt.subplots()
    mat = plt.imshow(prediction[0], origin="lower", interpolation="none")
    clb = plt.colorbar(mat)
    clb.set_clim(vmin=0, vmax=1)
    clb.draw_all()

    from matplotlib.widgets import Button
    from matplotlib.widgets import Slider
    class UICallback(object):
        def position_changed(self, value):
            nonlocal i
            value = int(value)
            i = value % len(difference)

        def playpause(self, event):
            nonlocal pause, bplaypause
            pause = not pause
            bplaypause.label.set_text("Play" if pause else "Pause")

        def switchsource(self, event):
            nonlocal image_mode, bswitchsource
            if (event.button == 1):
                image_mode = (image_mode + 1) % 4
            else:
                image_mode = (image_mode - 1) % 4

            if (image_mode == 0):
                bswitchsource.label.set_text("Pred")
            elif (image_mode == 1):
                bswitchsource.label.set_text("Orig")
            elif (image_mode == 2):
                bswitchsource.label.set_text("Orig v")
            else:
                bswitchsource.label.set_text("Diff")

    callback = UICallback()
    axplaypause = plt.axes([0.145, 0.91, 0.10, 0.05])
    axswitchsource = plt.axes([0.645, 0.91, 0.10, 0.05])
    axposition = plt.axes([0.275, 0.91, 0.30, 0.05])

    bplaypause = Button(axplaypause, "Pause")
    bplaypause.on_clicked(callback.playpause)

    bswitchsource = Button(axswitchsource, "Pred")
    bswitchsource.on_clicked(callback.switchsource)

    sposition = Slider(axposition, 'n', 0, len(test_data[0]), valinit=0, valfmt='%1.0f')
    sposition.on_changed(callback.position_changed)

    ani = animation.FuncAnimation(fig, update_new, interval=1, save_count=50)

    plt.show()

    print("done.")

def mainFunction():
    global output_weights, frame_output_weights, last_states
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
    if (os.path.exists("../cache/esn/uv/cross_pred_patches_advanced_mt{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units)) == False):
        generate_new = True

    if (generate_new):
        print("setting up...")

        last_states = np.empty(((N-2)*(N-2), n_units, 1))
        output_weights = np.empty(((N-2)*(N-2),sigma*sigma, sigma*sigma+1+n_units))
        frame_output_weights = np.empty(((N-2)*(N-2),2*2, 2*2+1+n_units))
    else:
        print("loading existing model...")

        f = open("../cache/esn/uv/cross_pred_patches_advanced_mt{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units), "rb")
        output_weights_t = pickle.load(f)
        frame_output_weights_t = pickle.load(f)
        last_states_t = pickle.load(f)
        f.close()

        output_weights[:] = output_weights_t[:]
        frame_output_weights[:] = frame_output_weights_t[:]
        last_states[:] = last_states_t[:]

    training_data = data[:, :ndata-2000]
    test_data = data[:, ndata-2000:]

    shared_training_data[:, :, :, :] = training_data[:, :, :, :]
    shared_test_data[:, :, :, :] = test_data[:, :, :, :]


    print("preparing threads...")
    def get_prediction_init(q):
        get_prediction.q = q

    queue = Queue()
    pool = Pool(16, get_prediction_init, [queue])
    processProcessResultsThread = Process(target=processThreadResults, args=("processProcessResultsThread", queue, 16, N*N) )

    print("process balance:")
    modifyDataProcessList = []
    jobs = []
    inner_index = 0
    outer_index = 0
    for y in range(N):
        for x in range(N):
            if (y > 1 and y < N-2 and x > 1 and x < N-2):
                inner_index += 1
                jobs.append((y, x, inner_index))
            else:
                outer_index += 1
                jobs.append((y, x, outer_index))

    print("fitting...")
    processProcessResultsThread.start()
    results = pool.map(get_prediction, jobs)
    pool.close()

    processProcessResultsThread.join()

    print("finished fitting")

    if (generate_new):
        print("saving model...")

        f = open("../cache/esn/uv/cross_pred_patches_advanced_mt{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units), "wb")
        pickle.dump(output_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(frame_output_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(last_states, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    diff = test_data[0]-prediction
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))

    showResults(shared_test_data, prediction, diff)

if __name__== '__main__':
    mainFunction()
