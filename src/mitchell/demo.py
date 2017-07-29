import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MitchellSimulation import MitchellSimulation
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

from helper import *

def demo_mitchell():
    Nx = 150
    Ny = 150
    deltaT = 1e-2
    deltaX = 0.1
    D = 5e-3  #1e-1
    h = D/deltaX**2  #1.0#0.2
    print(h)
    #h = D over delta_x

    #tau constants according to https://books.google.de/books?id=aB34DAAAQBAJ&pg=PA134&lpg=PA134&dq=mitchell-schaefer+model&source=bl&ots=RVuc3hoJwW&sig=ukfFhjF_COsljaaznv5uB6Cn5V8&hl=de&sa=X&ved=0ahUKEwiozdj8ic7TAhURLVAKHfa3A5wQ6AEIOTAC#v=onepage&q=mitchell-schaefer%20model&f=false
    #here might be some interesting things about the D value: http://ac.els-cdn.com/S0025556416301225/1-s2.0-S0025556416301225-main.pdf?_tid=875d11c0-3e31-11e7-a57a-00000aab0f26&acdnat=1495376990_ecb27e6ed1dd80a86e9d62e38a6c187e
    return MitchellSimulation(Nx, Ny, deltaT, deltaX, D, D, t_in=0.3, t_out=6.0, t_close=150, t_open=120, v_gate=0.13,)

sim = demo_mitchell()
#sim.initialize_two_spirals()
sim.initialize_one_spiral()
#sim.initialize_random(42, 0.1)
"""
datafull = np.empty((200*100, 150, 150))

data = np.empty(9000*100)
for i in range(len(data)):
    data[i] = sim._v[120,20]
    if (i > 6800*100 and i < 6800*100+200*100):
        datafull[i-6800*100] = sim._v
    sim.explicit_step(chaotic=True)

    if (i % 500 == 0):
        print("{0}/{1:1f}%".format(i, 100/len(data)*i))

print("done!")

plt.plot(np.arange(len(data))*0.01,data)
plt.show()


import pickle as pickle
f = open("data.dat", "wb")
pickle.dump(data, f)
f.close()

viewResults = [("v", datafull)]
f = open("data.view.dat", "wb")
pickle.dump(viewResults, f)
f.close()
show_results(viewResults)

exit()
"""
frame = 0
import time
def update_new(data):
    #time.sleep(50)
    global sim, frame
    for j in range(int(sskiprate.val)):
        sim.explicit_step(chaotic=True)
        frame += 1
    mat.set_data(sim._v if mode else sim._h)
    plt.title("{0:.1f}".format(frame*1e-2), x = -0.15, y=-2)
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
