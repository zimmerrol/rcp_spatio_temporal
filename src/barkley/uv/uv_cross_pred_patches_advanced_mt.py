# -*- coding: utf-8 -*-

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

def create_patch_indices(range_x, range_y):
    ind_x = np.tile(range(range_x[0], range_x[1]), range_y[1] - range_y[0])
    ind_y = np.repeat(range(range_y[0], range_y[1]), range_x[1] - range_x[0])

    return ind_y, ind_x

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

def fit_predict_pixel(y, x, running_index, prediction, last_states, output_weights, training_data, test_data, esn):
    print("{0},{1}".format(y, x))    
    ind_y, ind_x = create_patch_indices((x - 2, x + 3), (y - 2, y + 3))
    print("{0},{1}".format(y, x))
    training_data_in = training_data[1][:, ind_y, ind_x].reshape(-1, 5*5)
    training_data_out = training_data[0][:, y, x].reshape(-1, 1)
    print("{0},{1}".format(y, x))
    test_data_in = test_data[1][:, ind_y, ind_x].reshape(-1, 5*5)
    test_data_out = test_data[0][:, y, x].reshape(-1, 1)
    print("{0},{1}".format(y, x))
    if (generate_new):
        train_error = esn.fit(training_data_in, training_data_out, verbose=1)
        print("{0},{1}".format(y, x))
        #last_states[running_index] = esn._x
        #output_weights[running_index] = esn._W_out
    else:
        esn._x = last_states[running_index]
        esn._W_out = output_weights[running_index]
    print("{0},{1}".format(y, x))
    pred = esn.predict(test_data_in, verbose=0)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0
    print("{0},{1}".format(y, x))
    return pred[:,0]

def fit_predict_frame_pixel(y, x, running_index, prediction, last_states, output_weights, training_data, test_data, progressCounter):
    #print("{0} - {1}".format(running_index, progressCounter))
    #return
    
    ind_y, ind_x = create_patch_indices((x, x + 2), (y, y + 2))
    #print(ind_x)
    #print("{0},{1}".format(x,y))
    training_data_in = training_data[1][:, ind_y, ind_x].reshape(-1, 2*2)
    training_data_out = training_data[0][:, ind_y, ind_x].reshape(-1, 2*2)

    test_data_in = test_data[1][:, ind_y, ind_x].reshape(-1, 2*2)
    test_data_out = test_data[0][:, ind_y, ind_x].reshape(-1, 2*2)

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
#bar = progressbar.ProgressBar(max_value=(N-4)*(N-4) + N//2*2 + (N-2)//2*2, redirect_stdout=True, poll_interval=0.0001)
#bar.update(0)

from threading import Thread
from queue import Queue

def processThreadResults(threadname, q, numberOfWorkers, numberOfResults):
    global resultData
    finishedWorkers = 0
    finishedResults = 0

    while True:
        if (finishedResults == numberOfWorkers):
            return

        newData= q.get()
        finishedResults += 1
        ind_y, ind_x, data = newData

        prediction[:, ind_y, ind_x] = data

        #bar.update(finishedResults)

def predict_inner(threadname, q, yStart, height, esn):
    for offset in range(height):
        y = offset + yStart
        
        for x in range(2, N-2):
            print("{0},{1}".format(y, x))
            pred = fit_predict_pixel(y, x, (y-2)*(N-4) + (x-2), prediction, last_states, output_weights, training_data, test_data, esn)
            print("{0},{1}".format(y, x))
            q.put((y, x, pred))

threadNumber = 16
if ((N-4) % threadNumber != 0):
    print("please adjust the threadNumber!")
    #exit()

mt_height = (N-4)//threadNumber

queue = Queue()
processThreadResultsThread = Thread(target=processThreadResults, args=("processThreadResultsThread", queue, mt_height, (N-4)*(N-4)) )
processThreadResultsThread.start()

print("thread balance:")
modifyDataThreadList = []
for i in range(threadNumber):
    y = mt_height*i
    if (i == threadNumber-1):
        mt_height = (N-2)-(i-1)*mt_height
    print(" -{0}".format(mt_height))
    modifyDataThread = Thread(target=predict_inner, args=("modifyDataThread-{0}".format(y), queue, y, mt_height, copy.deepcopy(esn)))
    modifyDataThreadList.append(modifyDataThread)

for thread in modifyDataThreadList:
    thread.start()

for thread in modifyDataThreadList:
    thread.join()

processThreadResultsThread.join()

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

i = 0
def update_new(data):
    global i

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
pause = False
image_mode = 0

from matplotlib.widgets import Button
from matplotlib.widgets import Slider
class UICallback(object):
    def position_changed(self, value):
        global i
        value = int(value)
        i = value % len(difference)

    def playpause(self, event):
        global pause, bplaypause
        pause = not pause
        bplaypause.label.set_text("Play" if pause else "Pause")

    def switchsource(self, event):
        global image_mode, bswitchsource
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
