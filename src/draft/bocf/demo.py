import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BOCFSimulation import BOCFSimulation
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

def demo_chaotic():
    #for chaotic u^3 simulation
    Nx = 150
    Ny = 150
    deltaT = 0.01#0.001
    deltaX = 1.0#0.1#0.25

    return BOCFSimulation(Nx, Ny, deltaT, deltaX)

sim = demo_chaotic()

frame = 0
def update_new(data):
    global sim, frame, clb
    for j in range(int(sskiprate.val)):
        sim.explicit_step()
        frame += 1
    field = sim._s*85.7-84
    mat.set_data(field)
    plt.title(frame, x = -0.15, y=-2)
    #clb.set_clim(vmin=np.min(field), vmax=np.max(field))
    #clb.draw_all()

    #print(np.max(sim._v))

    return [mat]

fig, ax = plt.subplots()

mat = ax.matshow(sim._u*85.7-84, vmin=-80, vmax=20, interpolation=None, origin="lower")
clb = plt.colorbar(mat)
clb.set_clim(vmin=-80, vmax=20)
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

sskiprate = Slider(axskiprate, 'Skip rate', 1, 500, valinit=1, valfmt='%1.0f')
bnext = Button(axsave, 'Save')
bnext.on_clicked(callback.save)
bprev = Button(axload, 'Load')
bprev.on_clicked(callback.load)

plt.show()
