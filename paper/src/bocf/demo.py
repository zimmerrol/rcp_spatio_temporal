"""
    Live demo of the BOCF model. Can be used to generate the data of the BOCF model for the prediction tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BOCFSimulation import BOCFSimulation
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from scipy.misc import imresize
import cv2

def demo_chaotic():
    Nx = 500
    Ny = 500
    deltaT = 0.1
    deltaX = 1.0
    D = 0.2

    return BOCFSimulation(Nx, Ny, D, deltaT, deltaX, parameters="tnpp")

sim = demo_chaotic()
sim.initialize_spiral_virtheart()


#times to start a fast pacinf or vertical waves of the u variable
pace_times = []
vertical_wave_times = []

#time to pertubate the system by cutting parts of the u variable
pertubation_time = 2500


"""
    Simulate the BOCF model with parameters which cause massive chaos to create a good time series for the predictive tasks and save the results.
"""
def calculate_dataset():
    save_data = np.empty((4, 20000, 150, 150))
    frame = 0

    transient_time = 5000

    global frame, save_data
    for j in range(450000):
        sim.explicit_step()
        frame += 1

        if frame == pertubation_time:
            sim._u[:sim._Ny//2, :] = 0.0

        if frame in pace_times:
            sim.initialize_left_wave(2)


        if frame in vertical_wave_times:
            sim._u[:2, :] = 1.0

        if np.abs(np.max(sim._u)) < 1e-3:
            print("no more excitated.")
            plt.imshow(sim._u)
            plt.show()

            save_data = save_data[:, :frame//20, :]
            np.save("D:\\bocf_data_02.npy", save_data)

            exit()

        if frame % 20 == 0:
            print(frame)

        if frame > 5000 and frame < 400000:
            #skip transient time
            if frame % 20 == 0:
                #save data
                save_data[0, (frame-transient_time)//20] = cv2.resize(sim._u, (150, 150))
                save_data[1, (frame-transient_time)//20] = cv2.resize(sim._v, (150, 150))
                save_data[2, (frame-transient_time)//20] = cv2.resize(sim._w, (150, 150))
                save_data[3, (frame-transient_time)//20] = cv2.resize(sim._s, (150, 150))
                print((frame-transient_time)//20)

        if frame >= 20000*20+transient_time-1:
            #save the generated data to the HDD
            np.save("bocf_data.npy", save_data)
            real_field = np.empty((4, 500, 500))
            real_field[0] = sim._u
            real_field[1] = sim._v
            real_field[2] = sim._w
            real_field[3] = sim._s
            print("done.")
            exit()

"""
    Simulate the BOCF model with parameters which cause massive chaos to create a good time series for the predictive tasks and show the results live.
"""
def show_animation():
    frame = 0

    def update_new(data):
        global sim, frame, clb

        for j in range(int(sskiprate.val)):
            sim.explicit_step()
            frame += 1

            if frame == pertubation_time:
                sim._u[:sim._Ny//2, :] = 0.0

            if frame in pace_times:
                sim.initialize_left_wave(2)

            if frame in vertical_wave_times:
                sim._u[:2, :] = 1.0

        field = sim._u
        mat.set_data(field)
        plt.title(frame, x = -0.15, y=-2)
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

#uncomment this line to calculate a BOCF model time series for the predictive tasks
#calculate_dataset()
show_animation()
