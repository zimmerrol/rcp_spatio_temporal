import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN

def generate_data(N, trans, sample_rate=1):
    Nx = 155
    Ny = 155
    deltaT = 1e-2
    epsilon = 0.08
    delta_x = 0.1
    D = 1/50
    h = D/delta_x**2
    print(h)
    #h = D over delta_x
    a = 0.75
    b = 0.06

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    import progressbar
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

    return data

def create_patch_indices(outer_range_x, outer_range_y, inner_range_x, inner_range_y):
    outer_ind_x = np.tile(range(outer_range_x[0], outer_range_x[1]+1), outer_range_y[1]-outer_range_y[0])
    outer_ind_y = np.repeat(range(outer_range_y[0], outer_range_y[1]+1), outer_range_x[1]-outer_range_x[0])

    inner_ind_x = np.tile(range(inner_range_x[0], inner_range_x[1]+1), inner_range_y[1] - inner_range_y[0])
    inner_ind_y = np.repeat(range(inner_range_y[0], inner_range_y[1]+1), inner_range_x[1] - inner_range_x[0])

    outer_list = [c for c in zip(outer_ind_y, outer_ind_x)]
    inner_list = [c for c in zip(inner_ind_y, inner_ind_x)]

    real_list = np.array([x for x in outer_list if x not in inner_list])
    inner_list = np.array(inner_list)

    return real_list[:,0], real_list[:,1], inner_list[:, 0], inner_list[:, 1]

def print_field(input_y, input_x, output_y, output_x):
    print_matrix = np.zeros((30,30))
    print_matrix[input_y, input_x] = 1.0
    print_matrix[output_y, output_x] = -1.0
    for y in range(30):
        string = "|"
        for x in range(30):
            if (print_matrix[y,x] == 1.0):
                string += "x"
            elif (print_matrix[y,x] == -1.0):
                string += "0"
            else:
                string += "."

        string += "|"
        print(string)

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

print("generating data...")
#data = generate_data(10000, 50000, 5)
#np.save("10000_155.dat", data)
#print("done")
#exit()
data = np.load("10000_155.dat.npy")

training_data = data[:4000]
test_data = data[4000:6000]

#input_y, input_x, output_y, output_x = create_patch_indices((12,17), (12,17), (13,16), (13,16)) # -> yields MSE=0.0115 with leak_rate = 0.8
#input_y, input_x, output_y, output_x = create_patch_indices((4,23), (4,23), (7,20), (7,20)) # -> yields MSE=0.0873 with leak_rate = 0.3

def train_test_esn(input_idx, output_idx, training_data, test_data, merged_prediction, esn=None):
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
        esn = ESN(n_input = len(input_y), n_output = len(output_y), n_reservoir = 20, # use1700 for best performance!
                weight_generation = "advanced", leak_rate = 0.8, spectral_radius = 0.2,
                random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[6e-1], solver = "lsqr")#,
                #out_activation = lambda x: 0.5*(1+np.tanh(x/2)), out_inverse_activation = lambda x:2*np.arctanh(2*x-1))

    train_error = esn.fit(training_data_in, training_data_out,)
    np.random.seed(42)

    pred = esn.predict(test_data_in)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    merged_prediction[:, output_y, output_x] = pred

    diff = pred.reshape((-1, len(output_y))) - test_data_out
    mse = np.mean((diff)**2)
    print("train/test: {0:5f}/ {1:5f}".format(train_error, mse))

    return esn

sigma = 5
N = 155
indices = create_indices(N=N, sigma=sigma)

print(indices.shape)

import progressbar
bar = progressbar.ProgressBar(max_value=((N//sigma+1)//2)**2, redirect_stdout=True, poll_interval=0.0001)
bar.update(0)

merged_prediction = test_data.copy()
esn = None

output_weights = [None]*(((N//sigma+1)//2)*((N//sigma+1)//2))
last_state = [None]*(((N//sigma+1)//2)*((N//sigma+1)//2))

print(len(output_weights))

generate_new = True

if (generate_new == False):
    import dill as pickle

    f = open("cross_ped_patches155_5.dat", "rb")
    output_weights = pickle.load(f)
    last_state = pickle.load(f)
    esn = pickle.load(f)
    f.close()

for i in range(1, N//sigma-1, 2):
    for j in range(1, N//sigma-1, 2):
        input_idx = np.zeros((8*sigma**2,2), dtype=int)

        input_idx[0 : 3*(sigma**2),:] = indices[i-1,j-1:j+2].reshape((-1,2))
        input_idx[sigma**2*3 : 2*sigma**2*3,:] = indices[i+1,j-1:j+2].reshape((-1,2))
        input_idx[2*sigma**2*3 : 2*sigma**2*3 + sigma**2,:] = indices[i,j-1]
        input_idx[2*sigma**2*3 + sigma**2 : 2*sigma**2*3 + 2*sigma**2,:] = indices[i,j+1]

        output_idx = indices[i,j].astype(int)

        if (generate_new):
            esn = train_test_esn(input_idx, output_idx, training_data, test_data, merged_prediction, esn)
            output_weights[(i-1)//2*((N//sigma+1)//2)+(j-1)//2] = esn._W
            last_state[(i-1)//2*((N//sigma+1)//2)+(j-1)//2] = esn._x
        else:
            esn._x = last_state[(i-1)//2*((N//sigma+1)//2)+(j-1)//2]
            esn._W = output_weights[(i-1)//2*((N//sigma+1)//2)+(j-1)//2]

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
            print("test: {0:5f}".format(mse))

        #merged_prediction[:, output_idx[:,0], output_idx[:,1]] = 0.0
        #merged_prediction[:, input_idx[:,0], input_idx[:,1]] = 1.0
        bar.update((i-1)//2*((N//sigma+1)//2)+(j-1)//2)
bar.finish()

if (generate_new):
    import dill as pickle

    f = open("cross_ped_patches155_5.dat", "wb")
    pickle.dump(output_weights, f)
    pickle.dump(last_state, f)
    pickle.dump(esn, f)
    f.close()

difference = np.abs(test_data - merged_prediction)

totalMSE = np.sum((test_data - merged_prediction)**2)/(len(difference)*((N//sigma+1)//2)**2*sigma**2)

print("total MSE: {0}".format(totalMSE))

i = 0
def update_new(data):
    global i
    if (not pause):
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
