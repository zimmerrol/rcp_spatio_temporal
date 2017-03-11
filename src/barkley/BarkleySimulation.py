import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class BarkleySimulation:
    def __init__(self, Nx, Ny, deltaT, epsilon, h, a, b, boundary_mode = "noflux"):
        self._Nx = Nx
        self._Ny = Ny
        self._deltaT = deltaT
        self._epsilon = epsilon
        self._h = h
        self._a = a
        self._b = b

        self._boundary_mode = boundary_mode

    def initialize_one_spiral(self):
        self._u = np.zeros((self._Nx, self._Ny))
        self._v = np.zeros((self._Nx, self._Ny))

        #for one spiral
        for i in range(self._Nx):
            for j in range(self._Ny):
                if (i >= self._Nx//2):
                    self._u[i, j] = 1.0
                if (j >= self._Ny//2):
                    self._v[i, j] = self._a/2.0

    def initialize_two_spirals(self):
        self._u = np.zeros((self._Nx, self._Ny))
        self._v = np.zeros((self._Nx, self._Ny))

        #for two spirals
        for i in range(self._Nx):
            for j in range(self._Ny):
                if (i >= self._Nx//3 and i <= self._Nx//3*2):
                    self._u[i, j] = 1.0
                if (j >= self._Ny//2):
                    self._v[i, j] = self._a/2.0

    def _set_boundaries(self, oldFields):
        if (self._boundary_mode == "noflux"):
            for (field, oldField) in zip((self._u, self._v), oldFields):
                field[:,0] = oldField[:,1]
                field[:,-1] = oldField[:,-2]
                field[0,:] = oldField[1,:]
                field[-1,:] = oldField[-2,:]

    def explicit_step(self):
        uOld = self._u.copy()
        vOld = self._v.copy()

        f = 1/self._epsilon * self._u * (1 - self._u) * (self._u - (self._v+self._b)/self._a)

        laplace = -4*self._u.copy()

        laplace += np.roll(self._u, +1, axis=0)
        laplace += np.roll(self._u, -1, axis=0)
        laplace += np.roll(self._u, +1, axis=1)
        laplace += np.roll(self._u, -1, axis=1)

        self._u = self._u + self._deltaT * (f + self._h**2 * laplace)
        self._v = self._v + self._deltaT * (np.power(uOld, 3) - self._v)

        self._set_boundaries((uOld, vOld))
