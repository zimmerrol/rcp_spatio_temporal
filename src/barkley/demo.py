import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

def demo_chaotic():
    #for chaotic u^3 simulation
    Nx = 30
    Ny = 30
    deltaT = 1e-2
    epsilon = 0.08
    h = 1.0#0.2
    a = 0.75
    b = 0.00006

    return BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)

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
sim.initialize_two_spirals()


def update_new(data):
    global sim
    for j in range(int(sskiprate.val)):
        sim.explicit_step(chaotic=True)
    mat.set_data(sim._u)
    return [mat]

fig, ax = plt.subplots()

mat = ax.matshow(sim._u, vmin=0, vmax=1, interpolation=None, origin="lower")
plt.colorbar(mat)
ani = animation.FuncAnimation(fig, update_new, interval=0, save_count=50)

class StorageCallback(object):
    def save(self, event):
        global sim

        np.save("simulation_cache_u.cache", sim._u)
        np.save("simulation_cache_v.cache", sim._v)

        print("Saved!")

    def load(self, event):
        global sim

        sim._u = np.load("simulation_cache_u.cache.npy")
        sim._v = np.load("simulation_cache_v.cache.npy")

        print("Loaded!")

callback = StorageCallback()
axsave = plt.axes([0.15, 0.01, 0.1, 0.075])
axload = plt.axes([0.65, 0.01, 0.1, 0.075])
axskiprate = plt.axes([0.15, 0.95, 0.60, 0.03])

sskiprate = Slider(axskiprate, 'Skip rate', 1, 500, valinit=10, valfmt='%1.0f')
bnext = Button(axsave, 'Save')
bnext.on_clicked(callback.save)
bprev = Button(axload, 'Load')
bprev.on_clicked(callback.load)

plt.show()
