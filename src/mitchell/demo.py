import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MitchellSimulation import MitchellSimulation
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

def demo_chaotic():
    #for chaotic u^3 simulation
    Nx = 150
    Ny = 150
    deltaT = 1e-3
    deltaX = 0.1
    D = 1e-2
    h = D/deltaX**2#1.0#0.2
    print(h)
    #h = D over delta_x

    return MitchellSimulation(Nx, Ny, deltaT, deltaX, D, D, 0.03, 0.60, 1.50, 1.20, 0.13)

def demo_oscillating():
    #for oscillations
    Nx = 200
    Ny = 200
    deltaT = 1e-2
    epsilon = 0.01
    h = 1.0#0.2
    a = 0.75
    b = 0.002

    return BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)

sim = demo_chaotic()
#sim.initialize_two_spirals()
sim.initialize_random(42, 0.1)

frame = 0
import time
def update_new(data):
    #time.sleep(50)
    global sim, frame
    for j in range(int(sskiprate.val)):
        sim.explicit_step(chaotic=True)
        frame += 1
    mat.set_data(sim._v if mode else sim._h)
    plt.title(frame, x = -0.15, y=-2)
    return [mat]

fig, ax = plt.subplots()

mat = ax.matshow(sim._v, vmin=0, vmax=1, interpolation=None, origin="lower")
plt.colorbar(mat)
ani = animation.FuncAnimation(fig, update_new, interval=0, save_count=50)

mode = True
class StorageCallback(object):
    def save(self, event):
        global sim

        np.save("simulation_cache_v.cache", sim._v)
        np.save("simulation_cache_h.cache", sim._h)

        print("Saved!")

    def load(self, event):
        global sim

        sim._v = np.load("simulation_cache_v.cache.npy")
        sim._h = np.load("simulation_cache_h.cache.npy")

        print("Loaded!")

    def switch_mode(self, event):
        global mode, bmode
        mode = not mode

        bmode.label.set_text("v" if mode else "h")


callback = StorageCallback()
axsave = plt.axes([0.15, 0.01, 0.1, 0.075])
axload = plt.axes([0.65, 0.01, 0.1, 0.075])
axmode = plt.axes([0.45, 0.01, 0.1, 0.075])
axskiprate = plt.axes([0.15, 0.95, 0.60, 0.03])

sskiprate = Slider(axskiprate, 'Skip rate', 1, 500, valinit=1, valfmt='%1.0f')
bnext = Button(axsave, 'Save')
bnext.on_clicked(callback.save)

bmode = Button(axmode, 'v')
bmode.on_clicked(callback.switch_mode)

bprev = Button(axload, 'Load')
bprev.on_clicked(callback.load)




plt.show()
