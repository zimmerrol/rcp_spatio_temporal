import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
    Simulates a 2D Barkley system.
"""
class BarkleySimulation:
    """
        Initializes the system.
    """
    def __init__(self, Nx, Ny, deltaT, epsilon, h, a, b, boundary_mode = "noflux"):
        self._Nx = Nx
        self._Ny = Ny
        self._deltaT = deltaT
        self._epsilon = epsilon
        self._h = h
        self._a = a
        self._b = b

        self._boundary_mode = boundary_mode

    """
        Seperates the fields in equally spaced and sized squares with are
        homogeneously filled with random values.
    """
    def initialize_random(self, seed, delta_x):
        np.random.seed(seed)

        n = int(np.ceil(1/delta_x))

        self._u = np.random.rand(self._Nx//n, self._Ny//n)
        tmp = np.repeat(self._u, np.ones(len(self._u), dtype=int)*n, axis=0)
        self._u = np.repeat(tmp, np.ones(len(self._u), dtype=int)*n, axis=1)

        self._v = self._u.copy()
        self._v[self._v<0.4] = 0.0
        self._v[self._v>0.4] = 1.0

    """
        Initiliazes the system so, that one spiral exist.
    """
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

    """
        Initiliazes the system so, that two spirals exist.
    """
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

    """
        Sets the von Neumann boundaries.
    """
    def _set_boundaries(self, oldFields):
        if (self._boundary_mode == "noflux"):
            for (field, oldField) in zip((self._u, self._v), oldFields):
                field[:,0] = oldField[:,1]
                field[:,-1] = oldField[:,-2]
                field[0,:] = oldField[1,:]
                field[-1,:] = oldField[-2,:]

    """
        Performs a discrete and explicit time step to solve the PDE.
    """
    def explicit_step(self, chaotic=False):
        uOld = self._u.copy()
        vOld = self._v.copy()

        f = 1/self._epsilon * self._u * (1 - self._u) * (self._u - (self._v+self._b)/self._a)

        laplace = -4*self._u.copy()

        laplace += np.roll(self._u, +1, axis=0)
        laplace += np.roll(self._u, -1, axis=0)
        laplace += np.roll(self._u, +1, axis=1)
        laplace += np.roll(self._u, -1, axis=1)

        self._u = self._u + self._deltaT * (f + self._h * laplace)

        if (chaotic):
            self._v = self._v + self._deltaT * (np.power(uOld, 3) - self._v)
        else:
            self._v = self._v + self._deltaT * (np.power(uOld, 1) - self._v)

        self._set_boundaries((uOld, vOld))
