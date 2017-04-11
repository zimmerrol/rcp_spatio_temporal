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



if (os.path.exists("../cache/raw/5000_{0}.uv.dat.npy".format(N)) == False):
    print("generating data...")
    data = generate_data(5000, 50000, 5, Ngrid=N)
    np.save("../cache/raw/5000_{0}.uv.dat.npy".format(N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("../cache/raw/20000_{0}.uv.dat.npy".format(N))
    print("loading finished")



sigma = 5

def create_patch_indices(range_x, range_y):
    ind_x = np.tile(range(range_x[0], range_x[1]), range_y[1] - range_y[0])
    ind_y = np.repeat(range(range_y[0], range_y[1]), range_x[1] - range_x[0])

    return ind_y, ind_x

print("setting up...")
n_units = 2000
esn = ESN(n_input = sigma*sigma, n_output = sigma*sigma, n_reservoir = n_units,
        weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
        random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-1], solver = "lsqr")

training_data = data[:, :4000]
test_data = data[:, 4000:5000]
prediction = np.ones((1000, N, N))

print("fitting...")
bar = progressbar.ProgressBar(max_value=(N//sigma)*(N//sigma), redirect_stdout=True, poll_interval=0.0001)
bar.update(0)

for y in range(0, N, sigma):
    for x in range(0, N, sigma):
        ind_y, ind_x = create_patch_indices((x, x + sigma), (y, y + sigma))

        aa = training_data[1][:, ind_y, ind_x]
        print(aa.shape)

        training_data_in = training_data[1][:, ind_y, ind_x].reshape(-1, sigma*sigma)
        training_data_out = training_data[0][:, ind_y, ind_x].reshape(-1, sigma*sigma)

        test_data_in = test_data[1][:, ind_y, ind_x].reshape(-1, sigma*sigma)
        test_data_out = test_data[0][:, ind_y, ind_x].reshape(-1, sigma*sigma)

        train_error = esn.fit(training_data_in, training_data_out, verbose=0)
        print("train error: {0}".format(train_error))

        pred = esn.predict(test_data_in, verbose=0)
        pred[pred>1.0] = 1.0
        pred[pred<0.0] = 0.0

        diff = test_data_out-pred
        mse = np.mean((diff)**2)
        print("test error: {0}".format(mse))

        pred2 = pred.reshape(-1, sigma, sigma)

        prediction[:, ind_y, ind_x] = pred

        bar.update(y//sigma*(N//sigma) + x//sigma)

bar.finish()

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
