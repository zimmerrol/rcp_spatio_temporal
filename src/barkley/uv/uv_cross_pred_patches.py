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

from helper import *

N = 150
ndata = 20000
sigma = 5
n_units = 1000

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
if (os.path.exists("../cache/esn/uv/cross_pred_patches{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units)) == False):
    generate_new = True

if (generate_new):
    print("setting up...")
    esn = ESN(n_input = sigma*sigma, n_output = sigma*sigma, n_reservoir = n_units,
            weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-1], solver = "lsqr")

    last_states = np.empty(((N//sigma)*(N//sigma), n_units, 1))
    output_weights = np.empty(((N//sigma)*(N//sigma),sigma*sigma, sigma*sigma+1+n_units))
else:
    print("loading existing model...")

    f = open("../cache/esn/uv/cross_pred_patches{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units), "rb")
    output_weights = pickle.load(f)
    last_states = pickle.load(f)
    esn = pickle.load(f)
    f.close()

training_data = data[:, :ndata-2000]
test_data = data[:, ndata-2000:]
prediction = np.ones((2000, N, N))

print("fitting...")
bar = progressbar.ProgressBar(max_value=(N//sigma)*(N//sigma), redirect_stdout=True, poll_interval=0.0001)
bar.update(0)

for y in range(0, N, sigma):
    for x in range(0, N, sigma):
        ind_y, ind_x = create_patch_indices((x, x + sigma), (y, y + sigma))

        training_data_in = training_data[1][:, y:y+sigma, x:x+sigma].reshape(-1, sigma*sigma)
        training_data_out = training_data[0][:, y:y+sigma, x:x+sigma].reshape(-1, sigma*sigma)

        test_data_in = test_data[1][:, y:y+sigma, x:x+sigma].reshape(-1, sigma*sigma)
        test_data_out = test_data[0][:, y:y+sigma, x:x+sigma].reshape(-1, sigma*sigma)

        if (generate_new):
            train_error = esn.fit(training_data_in, training_data_out, verbose=0)
            print("train error: {0}".format(train_error))

            last_states[y//sigma*(N//sigma) + x//sigma] = esn._x
            output_weights[y//sigma*(N//sigma) + x//sigma] = esn._W_out
        else:
            esn._x = last_states[y//sigma*(N//sigma) + x//sigma]
            esn._W_out = output_weights[y//sigma*(N//sigma) + x//sigma]

        pred = esn.predict(test_data_in, verbose=0)
        pred[pred>1.0] = 1.0
        pred[pred<0.0] = 0.0

        diff = test_data_out-pred
        mse = np.mean((diff)**2)
        print("test error: {0}".format(mse))

        pred2 = pred.reshape(-1, sigma, sigma)

        prediction[:, ind_y, ind_x] = pred

        bar.update(y//sigma*(N//sigma) + x//sigma+1)

bar.finish()

if (generate_new):
    print("saving model...")

    f = open("../cache/esn/uv/cross_pred_patches{0}_{1}_{2}_{3}.dat".format(N, ndata, sigma, n_units), "wb")
    pickle.dump(output_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
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
