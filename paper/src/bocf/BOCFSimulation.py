"""
    Implementation of the BOCF model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.ndimage.filters as filters

class BOCFSimulation:
    def __init__(self, Nx, Ny, D, delta_t, delta_x, parameters="tnpp"):
        self._Nx = Nx
        self._Ny = Ny
        self._delta_t = delta_t
        self._delta_x = delta_x

        self.D = D

        self._u = np.zeros((Nx, Ny))
        self._v = np.ones((Nx, Ny))
        self._w = np.ones((Nx, Ny))
        self._s = np.zeros((Nx, Ny))

        #use this matrix to discretize the laplacian operator
        self._laplacian_matrix = np.array([[1, 4, 1],
                                          [4, -20, 4],
                                          [1, 4, 1]])/6.0

        """
            Initializes the parameters according to the EPI parameter set.
        """
        def paper_init_epi():
            self._u_o = 0
            self._u_u = 1.55
            self._theta_v = 0.3
            self._theta_w = 0.13
            self._theta_v_minus = 0.006
            self._theta_o = 0.006
            self._tau_v1_minus = 60
            self._tau_v2_minus = 1150
            self._tau_v_plus = 1.4506
            self._tau_w1_minus = 60
            self._tau_w2_minus = 15
            self._k_w_minus = 65
            self._u_w_minus = 0.03
            self._tau_w_plus = 200
            self._tau_fi = 0.11
            self._tau_o1 = 400
            self._tau_o2 = 6
            self._tau_so1 = 30.0181
            self._tau_so2 = 0.9957
            self._k_so = 2.0458
            self._u_so = 0.65
            self._tau_s1 = 2.7342
            self._tau_s2 = 16
            self._k_s = 2.0994
            self._u_s = 0.9087
            self._tau_si = 1.8875
            self._tau_winfinity = 0.07
            self._w_inf_star = 0.94

        """
            Initializes the parameters according to the PB parameter set.
        """
        def paper_init_pb():
            self._u_o = 0
            self._u_u = 1.45
            self._theta_v = 0.35
            self._theta_w = 0.13
            self._theta_v_minus = 0.175
            self._theta_o = 0.006
            self._tau_v1_minus = 10
            self._tau_v2_minus = 1150
            self._tau_v_plus = 1.4506
            self._tau_w1_minus = 140
            self._tau_w2_minus = 6.25
            self._k_w_minus = 65
            self._u_w_minus = 0.015
            self._tau_w_plus = 326
            self._tau_fi = 0.105
            self._tau_o1 = 400
            self._tau_o2 = 6
            self._tau_so1 = 30.0181
            self._tau_so2 = 0.9957
            self._k_so = 2.0458
            self._u_so = 0.65
            self._tau_s1 = 2.7342
            self._tau_s2 = 16
            self._k_s = 2.0994
            self._u_s = 0.9087
            self._tau_si = 1.8875
            self._tau_winfinity = 0.175
            self._w_inf_star = 0.9

        """
            Initializes the parameters according to the TNPP parameter set.
        """
        def paper_init_tnnp():
            self._u_o = 0
            self._u_u = 1.58
            self._theta_v = 0.3
            self._theta_w = 0.015
            self._theta_v_minus = 0.015
            self._theta_o = 0.006
            self._tau_v1_minus = 60
            self._tau_v2_minus = 1150
            self._tau_v_plus = 1.4506
            self._tau_w1_minus = 70
            self._tau_w2_minus = 20
            self._k_w_minus = 65
            self._u_w_minus = 0.03
            self._tau_w_plus = 280
            self._tau_fi = 0.11
            self._tau_o1 = 6
            self._tau_o2 = 6
            self._tau_so1 = 43
            self._tau_so2 = 0.2
            self._k_so = 2
            self._u_so = 0.65
            self._tau_s1 = 2.7342
            self._tau_s2 = 3
            self._k_s = 2.0994
            self._u_s = 0.9087
            self._tau_si = 2.8723
            self._tau_winfinity = 0.07
            self._w_inf_star = 0.94


        if parameters == "epi":
            paper_init_epi()
        elif parameters == "pb":
            paper_init_pb()
        elif parameters == "tnpp":
            paper_init_tnnp()
        else:
            raise ValueError("no parameter set found!")

        #initial values which are immediately discarded
        self._tau_v_minus = None
        self._tau_w_minus = None
        self._tau_so = None
        self._tau_s = None
        self._tau_o = None
        self._v_infinity = None
        self._w_infinity = None

    """
        Initializes a spiral according to the implementation of thevirtualheart.org
    """
    def initialize_spiral_virtheart(self):
        #implements http://www.thevirtualheart.org/webgl/DS_SIAM/4v_minimal_model.html

        self._u = np.zeros((self._Ny, self._Nx))
        self._v = np.zeros((self._Ny, self._Nx))
        self._w = np.zeros((self._Ny, self._Nx))
        self._s = np.zeros((self._Ny, self._Nx))

        for i in range(self._Ny):
            for j in range(self._Nx//2, self._Nx):
                t  = (i-self._Ny//2)*0.1
                t2 = (i-self._Ny//2+20)*0.05
                self._u[i, j] = 1.5*np.exp(-t*t)
                self._v[i, j] = 1.0 - 0.9*np.exp(-t2*t2)


    """
        Sets the Neumann boundaries.
    """
    def _set_boundaries(self, old_fields):
        for (field, old_field) in zip((self._u, self._v, self._w, self._s), old_fields):
            field[:, 0] = old_field[:, 1]
            field[:, -1] = old_field[:, -2]
            field[0, :] = old_field[1, :]
            field[-1, :] = old_field[-2, :]


    """
        Calculates the laplacian of the field.
    """
    def _laplace(self, field):
        return filters.convolve(field, self._laplacian_matrix)

    """
        Heaviside function.
    """
    def H(self, y):
        return (y > 0.0).astype(float)

    """
        Calculates the current of fast incoming ions.
    """
    def _current_fi(self):
        return -self._v * self.H(self._u - self._theta_v) * (self._u - self._theta_v) * (self._u_u - self._u) / self._tau_fi

    """
        Calculates the current of slow outgoing ions.
    """
    def _current_so(self):
        return (self._u - self._u_o) * (1 - self.H(self._u - self._theta_w)) / self._tau_o + self.H(self._u - self._theta_w) / self._tau_so

    """
        Calculates the current of slow incoming ions.
    """
    def _current_si(self):
        return -self.H(self._u - self._theta_w) * self._w * self._s / self._tau_si

    """
        Updates the voltage dependet constants in time.
    """
    def _update_constants(self):
        self._tau_v_minus = (1.0 - self.H(self._u - self._theta_v_minus)) * self._tau_v1_minus + self.H(self._u - self._theta_v_minus) * self._tau_v2_minus
        self._tau_w_minus = self._tau_w1_minus + (self._tau_w2_minus - self._tau_w1_minus) * (1 + np.tanh(self._k_w_minus * (self._u - self._u_w_minus))) / 2.0
        self._tau_so      = self._tau_so1      + (self._tau_so2      - self._tau_so1)      * (1 + np.tanh(self._k_so      * (self._u - self._u_so)))      / 2.0
        self._tau_s = (1.0 - self.H(self._u - self._theta_w)) * self._tau_s1 + self.H(self._u - self._theta_w) * self._tau_s2
        self._tau_o = (1.0 - self.H(self._u - self._theta_o)) * self._tau_o1 + self.H(self._u - self._theta_o) * self._tau_o2

        self._v_infinity = (self._u < self._theta_v_minus).astype(float)
        self._w_infinity = (1.0 - self.H(self._u - self._theta_o)) * (1.0 - self._u / self._tau_winfinity) + self.H(self._u - self._theta_o) * self._w_inf_star

    """
        Calculate one explicit time step.
    """
    def explicit_step(self):
        u_old = self._u.copy()
        v_old = self._v.copy()
        w_old = self._w.copy()
        s_old = self._s.copy()

        self._update_constants()

        J_fi = self._current_fi()
        J_so = self._current_so()
        J_si = self._current_si()

        dudt = self.D * self._laplace(self._u) - (J_fi + J_so + J_si)
        dvdt = (1.0 - self.H(self._u - self._theta_v)) * (self._v_infinity - self._v) / self._tau_v_minus - self.H(self._u - self._theta_v) * self._v / self._tau_v_plus
        dwdt = (1.0 - self.H(self._u - self._theta_w)) * (self._w_infinity - self._w) / self._tau_w_minus - self.H(self._u - self._theta_w) * self._w / self._tau_w_plus
        dsdt = ((1.0 + np.tanh(self._k_s * (self._u - self._u_s))) / 2.0 - self._s) / self._tau_s

        self._u += self._delta_t * dudt
        self._v += self._delta_t * dvdt
        self._w += self._delta_t * dwdt
        self._s += self._delta_t * dsdt

        self._set_boundaries((u_old, v_old, w_old, s_old))
