import numpy as np

Nx = 200
Ny = 200
deltaT = 1e-4
epsilon =
h =
a =
b =

u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))


def step():
    uOld = u.copy()
    vOld = v.old()

    f = 1/epsilon * u * (1 - u) * (u - (v+b)/a)

    for x in range(1, Nx-1):
        for y in range(1, Ny-1):
            laplace = u[x-1, y] + u[x+1, y] + u[x, y-1] + u[x, y+1] - 4*u[x, y]
            u[x, y] = uOld[x, y] + deltaT * (f + h**2 + laplace)
