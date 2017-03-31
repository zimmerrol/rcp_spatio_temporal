import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN
import progressbar
import dill as pickle

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

    data = np.empty((N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step(chaotic=True)
        data[i] = sim._u
        bar.update(i+trans)

    bar.finish()
    return data

def create_indices(N, sigma):
    n = N // sigma

    indices = np.empty((n,n, sigma*sigma, 2))

    def createrectangle(range_x, range_y):
        ind_x = np.tile(range(range_x[0], range_x[1]), range_y[1]-range_y[0])
        ind_y = np.repeat(range(range_y[0], range_y[1]), range_x[1]-range_x[0])

        index_list = [c for c in zip(ind_y, ind_x)]

        index_list = np.array(index_list)

        return index_list

    for i in range(n):
        starty = i*sigma
        for j in range(n):
            startx = j*sigma
            indices[i,j] = createrectangle((starty,starty+sigma),(startx,startx+sigma))

    return indices

def train_test_esn(input_idx, output_idx, training_data, test_data, merged_prediction, esn=None, rs=7, lr=0.8, n=200, reg=1e-2, rho=0.2):
    input_y = input_idx[:,0]
    input_x = input_idx[:,1]
    output_y = output_idx[:,0]
    output_x = output_idx[:,1]

    training_data_in =  training_data[:, input_y, input_x].reshape(-1, len(input_y))
    training_data_out =  training_data[:, output_y, output_x].reshape(-1, len(output_y))

    test_data_in =  test_data[:, input_y, input_x].reshape(-1, len(input_y))
    test_data_out =  test_data[:, output_y, output_x].reshape(-1, len(output_y))

    generate_new = True

    if (esn is None):
        print("setting up...")
        esn = ESN(n_input = len(input_y), n_output = len(output_y), n_reservoir = n, #was 30, use 1700 for best performance!
                weight_generation = "advanced", leak_rate = lr, spectral_radius = rho,
                random_seed=7, noise_level=0.0001, sparseness=.1, regression_parameters=[reg], solver = "lsqr")

    train_error = esn.fit(training_data_in, training_data_out,)
    np.random.seed(42)

    pred = esn.predict(test_data_in)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    merged_prediction[:, output_y, output_x] = pred

    diff = pred.reshape((-1, len(output_y))) - test_data_out
    mse = np.mean((diff)**2)
    #print("train/test: {0:5f}/ {1:5f}".format(train_error, mse))

    return esn


N = 162
sigma = 2
clusterSize = 3
clusterDistance = clusterSize + 1
numberOfClusters = (N//sigma-1)//clusterDistance #was sigma+1

if (os.path.exists("cache/raw/10000_{0}.dat.npy".format(N)) == False):
    print("generating data...")
    data = generate_data(10000, 50000, 5, Ngrid=N)
    np.save("cache/raw/10000_{0}.dat.npy".format(N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("cache/raw/10000_{0}.dat.npy".format(N))
    print("loading finished")

training_data = data[:4000]
test_data = data[4000:6000]

indices = create_indices(N=N, sigma=sigma)

bar = progressbar.ProgressBar(max_value=numberOfClusters**2, redirect_stdout=True, poll_interval=0.0001)
bar.update(0)

merged_prediction = test_data.copy()
esn = None

output_weights = [None]*(numberOfClusters*numberOfClusters)
last_state = [None]*(numberOfClusters*numberOfClusters)

generate_new = False

if (os.path.exists("cache/esn/cross_ped_patches{0}_{1}_{2}.dat".format(N, sigma, clusterSize)) == False):
    generate_new = True

generate_new = True

if (generate_new == False):
    print("loading existing model...")

    f = open("cache/esn/cross_ped_patches{0}_{1}_{2}.dat".format(N, sigma, clusterSize), "rb")
    output_weights = pickle.load(f)
    last_state = pickle.load(f)
    esn = pickle.load(f)
    f.close()

for reg in [6e-3]: #[6e-3, 1e-4, 1e-5, 1e-6]:
#for nn in [10, 20, 30, 40, 50, 80, 100, 150, 200, 250, 300]:
    nn = 30
    esn = None
    bar.update(0)
    for i in range(1, N//sigma-1, clusterDistance):
        for j in range(1, N//sigma-1, clusterDistance):
            input_idx = np.zeros(((2*(clusterSize+2)+2*clusterSize)*sigma**2,2), dtype=int)

            input_idx[0 : (clusterSize+2)*(sigma**2),:] = indices[i-1, j-1:j+1+clusterSize].reshape((-1,2))
            input_idx[(clusterSize+2)*sigma**2 : 2*(clusterSize+2)*sigma**2, :] = indices[i+clusterSize, j-1:j+1+clusterSize].reshape((-1,2))
            input_idx[2*(clusterSize+2)*sigma**2 : 2*(clusterSize+2)*sigma**2 + clusterSize*sigma**2, :] = indices[i:i+clusterSize, j-1].reshape((-1,2))
            input_idx[2*(clusterSize+2)*sigma**2 + clusterSize*sigma**2 : 2*(clusterSize+2)*sigma**2 + 2*clusterSize*sigma**2, :] = indices[i:i+clusterSize, j+clusterSize].reshape((-1,2))

            output_idx = indices[i:i+clusterSize,j:j+clusterSize].astype(int).reshape((-1, 2))

            if (generate_new):
                esn = train_test_esn(input_idx, output_idx, training_data, test_data, merged_prediction, esn, n=nn, reg=reg, lr=0.8)
                output_weights[(i-1)//clusterDistance*numberOfClusters + (j-1)//clusterDistance] = esn._W_out
                last_state[(i-1)//clusterDistance*numberOfClusters + (j-1)//clusterDistance] = esn._x
            else:
                esn._x = last_state[(i-1)//clusterDistance*numberOfClusters + (j-1)//clusterDistance]
                esn._W_out = output_weights[(i-1)//clusterDistance*numberOfClusters + (j-1)//clusterDistance]

                input_y = input_idx[:,0]
                input_x = input_idx[:,1]
                output_y = output_idx[:,0]
                output_x = output_idx[:,1]

                test_data_in = test_data[:, input_y, input_x].reshape(-1, len(input_y))
                test_data_out = test_data[:, output_y, output_x].reshape(-1, len(output_y))

                np.random.seed(42)
                pred = esn.predict(test_data_in)
                pred[pred>1.0] = 1.0
                pred[pred<0.0] = 0.0

                merged_prediction[:, output_y, output_x] = pred

                diff = pred.reshape((-1, len(output_y))) - test_data_out
                mse = np.mean((diff)**2)

            #merged_prediction[:, output_idx[:,0], output_idx[:,1]] = 1.0
            #merged_prediction[:, input_idx[:,0], input_idx[:,1]] = 0.0

            bar.update((i-1)//clusterDistance*numberOfClusters + (j-1)//clusterDistance)

    difference = np.abs(test_data - merged_prediction)
    totalMSE = np.sum((test_data - merged_prediction)**2)/(len(difference)*(numberOfClusters*sigma*clusterSize)**2)
    print("total MSE: {0}".format(totalMSE))

bar.finish()

if (generate_new):
    print("saving model...")

    f = open("cache/esn/cross_ped_patches{0}_{1}_{2}.dat".format(N, sigma, clusterSize), "wb")
    pickle.dump(output_weights, f)
    pickle.dump(last_state, f)
    pickle.dump(esn, f)
    f.close()

difference = np.abs(test_data - merged_prediction)

totalMSE = np.sum((test_data - merged_prediction)**2)/(len(difference)*(numberOfClusters*sigma*clusterSize)**2)

print("total MSE: {0}".format(totalMSE))

i = 0
def update_new(data):
    global i

    if (image_mode == 0):
        mat.set_data(merged_prediction[i])
        clb.set_clim(vmin=0, vmax=1)
        clb.draw_all()
    elif (image_mode == 1):
        mat.set_data(test_data[i])
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
mat = plt.imshow(merged_prediction[0], origin="lower", interpolation="none")
clb = plt.colorbar(mat)
clb.set_clim(vmin=0, vmax=1)
clb.draw_all()
pause = False
image_mode = 0
ani = animation.FuncAnimation(fig, update_new, interval=1, save_count=50)

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
            image_mode = (image_mode + 1) % 3
        else:
            image_mode = (image_mode - 1) % 3

        if (image_mode == 0):
            bswitchsource.label.set_text("Pred")
        elif (image_mode == 1):
            bswitchsource.label.set_text("Orig")
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

sposition = Slider(axposition, 'n', 0, len(test_data), valinit=0, valfmt='%1.0f')
sposition.on_changed(callback.position_changed)

plt.show()
plt.close()

print("done.")
