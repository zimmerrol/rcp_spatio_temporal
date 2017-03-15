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
    Nx = 30
    Ny = 30
    deltaT = 1e-2
    epsilon = 0.08
    h = 1.0#0.2
    a = 0.75
    b = 0.00006

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    for i in range(trans):
        sim.explicit_step(chaotic=True)

    data = np.empty((N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step(chaotic=True)
        data[i] = sim._u

    return data

def create_patch_indices(outer_range_x, outer_range_y, inner_range_x, inner_range_y):
    outer_ind_x = np.tile(range(outer_range_x[0], outer_range_x[1]+1), outer_range_y[1])
    outer_ind_y = np.repeat(range(outer_range_y[0], outer_range_y[1]+1), outer_range_x[1])

    inner_ind_x = np.tile(range(inner_range_x[0], inner_range_x[1]+1), inner_range_y[1]-1)
    inner_ind_y = np.repeat(range(inner_range_y[0], inner_range_y[1]+1), inner_range_x[1]-1)

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



print("generating data...")
#data = generate_data(10000, 50000, 5)
#np.save("10000.dat", data)
data = np.load("10000.dat.npy")

training_data = data[:4000]
test_data = data[4000:]

#input_y, input_x, output_y, output_x = create_patch_indices((12,17), (12,17), (13,16), (13,16)) # -> yields MSE=0.0115 with leak_rate = 0.8
input_y, input_x, output_y, output_x = create_patch_indices((4,23), (4,23), (7,20), (7,20)) # -> yields MSE=0.0873 with leak_rate = 0.3

training_data_in =  training_data[:, input_y, input_x].reshape(-1, len(input_y))
training_data_out =  training_data[:, output_y, output_x].reshape(-1, len(output_y))

test_data_in =  test_data[:, input_y, input_x].reshape(-1, len(input_y))
test_data_out =  test_data[:, output_y, output_x].reshape(-1, len(output_y))


generate_new = False

print("setting up...")
if (generate_new):
    esn = ESN(n_input = len(input_y), n_output = len(output_y), n_reservoir = 1700,
            weight_generation = "advanced", leak_rate = 0.8, spectral_radius = 0.8,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[6e-1], solver = "lsqr",
            out_activation = lambda x: 0.5*(1+np.tanh(x/2)), out_inverse_activation = lambda x:2*np.arctanh(2*x-1))

    print("fitting...")

    train_error = esn.fit(training_data_in, training_data_out,)
    esn.save("esn" + str(len(input_y)) + ".dat")
    print("train error: {0}".format(train_error))

else:
    esn = ESN.load("esn" + str(len(input_y)) + ".dat")

print("predicting...")
pred = esn.predict(test_data_in)

merged_prediction = test_data.copy()
merged_prediction[:, output_y, output_x] = pred

diff = pred.reshape((-1, len(output_y))) - test_data_out
mse = np.mean((diff)**2)
print("test error: {0}".format(mse))

i = 0
def update_new(data):
    global i
    if (not pause):
        if (image_mode == 0):
            mat.set_data(merged_prediction[i])
        elif (image_mode == 1):
                mat.set_data(test_data[i])
        else:
            mat.set_data(np.abs(test_data[i]-merged_prediction[i]))

        i = (i+1) % len(diff)
        sposition.set_val(i)
    return [mat]


fig, ax = plt.subplots()
mat = plt.imshow(merged_prediction[0], origin="lower", vmin=0, vmax=1, interpolation="none")
plt.colorbar(mat)
pause = False
image_mode = 0
ani = animation.FuncAnimation(fig, update_new, interval=1, save_count=50)

from matplotlib.widgets import Button
from matplotlib.widgets import Slider
class UICallback(object):
    def position_changed(self, value):
        global i
        value = int(value)
        i = value

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
