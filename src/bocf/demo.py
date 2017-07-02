import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BOCFSimulation import BOCFSimulation
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from scipy.misc import imresize

def demo_chaotic():
    #for chaotic u^3 simulation

    #150x150 with D=0.1 is stable
    Nx = 500
    Ny = 500
    deltaT = 0.1#0.001
    deltaX = 0.125#1.0#0.1#0.25
    D = 0.01#0.2#0.2#1.71#1.171

    #Erzeugt sowohl mit "pb", "epi", "tnpp" als Wert für parameters eine anhaltende Dynamik, die allerdings kaum chaotisch ist.
    #Mit "thomas" als parameters zerfällt die erregung schnell
    return BOCFSimulation(Nx, Ny, D, deltaT, deltaX, parameters="virtheart")

sim = demo_chaotic()
sim.initialize_spiral_virtheart()
#sim.initialize_double_spiral()#(42, deltaX=0.1)
#sim.initialize_left_wave(2)
#sim.initialize_right_wave(2)
#sim.initialize_random(42, deltaX=0.05)
#sim.initialize_spiral()
#data = np.load("TestInit_169[4605].npy")

"""
sim._u = data[:, :, 0]
sim._v = data[:, :, 1]
sim._w = data[:, :, 2]
sim._s = data[:, :, 3]
"""

pace_times = []#[3000]
vertical_wave_times = []#[3500]
pertubation_time = 2500

save_data = np.empty((4, 20000, 150, 150))
frame = 0


for j in range(450000):
    sim.explicit_step()
    frame += 1

    if frame % 20 == 0:
        print(frame)

    if frame > 5000 and frame < 400000:
        #skip transient time
        if frame % 20 == 0:
            #save data
            save_data[0, frame//20-5000] = imresize(sim._u, (150, 150))
            save_data[1, frame//20-5000] = imresize(sim._v, (150, 150))
            save_data[2, frame//20-5000] = imresize(sim._w, (150, 150))
            save_data[3, frame//20-5000] = imresize(sim._s, (150, 150))

    if frame > 405000:
        np.save("D:\\bocf_data.npy", save_data)

def show_animation():
    def update_new(data):
        global sim, frame, clb

        for j in range(int(sskiprate.val)):
            sim.explicit_step()
            frame += 1

            if frame == pertubation_time:
                #sim._u[sim._Ny//2:, :sim._Nx//2] = 0.0
                #sim._u[:sim._Ny//2, sim._Nx//2:] = 0.0
                sim._u[:sim._Ny//2, :] = 0.0

            if frame in pace_times:
                sim.initialize_left_wave(2)
                #sim._u[sim._Ny//2:, :] = 0.0

            if frame in vertical_wave_times:
                sim._u[:2, :] = 1.0

            if (frame > 5000 and frame < 400000):
                #skip transient time
                if (frame % 20 == 0):
                    #save data
                    data[0, frame//20-5000] = imresize(sim._u, 150, 150)
                    data[1, frame//20-5000] = imresize(sim._v, 150, 150)
                    data[2, frame//20-5000] = imresize(sim._w, 150, 150)
                    data[3, frame//20-5000] = imresize(sim._s, 150, 150)

            if (frame > 400000):
                np.save("D:\\bocf_data.npy",save_data)

        field = sim._u
        mat.set_data(field)
        plt.title(frame, x = -0.15, y=-2)
        #clb.set_clim(vmin=np.min(field), vmax=np.max(field))
        #clb.draw_all()

        #print(np.max(sim._v))

        return [mat]

    fig, ax = plt.subplots()

    mat = ax.matshow(sim._u, vmin=0, vmax=2, interpolation=None, origin="lower")
    clb = plt.colorbar(mat)
    clb.set_clim(vmin=0, vmax=2)
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
