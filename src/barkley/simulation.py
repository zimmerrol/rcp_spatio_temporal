import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#for chaotic u^3 simulation
Nx = 200
Ny = 200
deltaT = 1e-2
epsilon = 0.08
h = 1.0#0.2
a = 0.75
b = 0.00006


"""
#for oscillations
Nx = 200
Ny = 200
deltaT = 1e-2
epsilon = 0.01
h = 1.0#0.2
a = 0.75
b = 0.002
"""


u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))


"""
#for one spiral
for i in range(Nx):
    for j in range(Ny):
        if (i >= Nx//2):
            u[i, j] = 1.0
        if (j >= Ny//2):
            v[i, j] = a/2.0
"""


#for two spirals
for i in range(Nx):
    for j in range(Ny):
        if (i >= Nx//3 and i <= Nx//3*2):
            u[i, j] = 1.0
        if (j >= Ny//2):
            v[i, j] = a/2.0

def set_boundaries(field, oldField):
    field[:,0] = oldField[:,1]
    field[:,-1] = oldField[:,-2]
    field[0,:] = oldField[1,:]
    field[-1,:] = oldField[-2,:]

def step():
    global u, v

    uOld = u.copy()
    vOld = v.copy()


    f = 1/epsilon * u * (1 - u) * (u - (v+b)/a)

    laplace = -4*u.copy()

    laplace += np.roll(u, +1, axis=0)
    laplace += np.roll(u, -1, axis=0)
    laplace += np.roll(u, +1, axis=1)
    laplace += np.roll(u, -1, axis=1)

    #print(np.max(laplace))


    u = u + deltaT * (f + h**2 * laplace)
    v = v + deltaT * (np.power(uOld, 3)-v)


    """
    for x in range(1, Nx-1):
        for y in range(1, Ny-1):
            u[x, y] = uOld[x, y] + deltaT * (1/epsilon*uOld[x, y] * (1.0 - uOld[x, y])*(uOld[x, y]-(vOld[x, y]+b)/a) +
            (uOld[x+1,y] + uOld[x-1,y] + uOld[x, y+1] + uOld[x, y-1] - 4*uOld[x, y]))

            v[x, y] = vOld[x, y] + deltaT * (uOld[x, y] - vOld[x, y])
    """

    set_boundaries(u, uOld)
    set_boundaries(v, vOld)


def update_new(data):
    global u, i
    for j in range(int(sskiprate.val)):
        step()
    mat.set_data(u)
    return [mat]

fig, ax = plt.subplots()

mat = ax.matshow(u, vmin=0, vmax=1, interpolation=None, origin="lower")
plt.colorbar(mat)
ani = animation.FuncAnimation(fig, update_new, interval=0, save_count=50)


from matplotlib.widgets import Button
from matplotlib.widgets import Slider

class StorageCallback(object):
    def save(self, event):
        global u, v

        np.save("simulation_cache_u.cache", u)
        np.save("simulation_cache_v.cache", v)

        print("Saved!")

    def load(self, event):
        global u, v

        u = np.load("simulation_cache_u.cache.npy")
        v = np.load("simulation_cache_v.cache.npy")

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
