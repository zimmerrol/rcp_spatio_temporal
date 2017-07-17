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

        self._laplacian_matrix = np.array([[1, 4, 1],
                                          [4, -20, 4],
                                          [1, 4, 1]])/6.0


        def init_old_spiral():
            self._u = np.zeros((Nx, Ny))
            for x in range(Nx//2):
                for y in range(Ny):
                    self._u[y, x] = 0.5

            self._v = np.zeros((Nx, Ny))
            for x in range(Nx):
                for y in range(Ny//2):
                    self._v[y, x] = 0.5
            self._w = np.ones((Nx, Ny))

            self._s = np.zeros((Nx, Ny))
        #init_spiral()

        def point(start_y, start_x, width, rad):
            for y in range(start_y-width//2, start_y+width//2):
                for x in range(start_x-width//2, start_x+width//2):
                    r = (y-start_y)**2+(x-start_x)**2
                    self._u[y, x] += 1.0*np.exp(-r*0.001)
                    self._v[y, x] += 0.5 * np.exp(-r*rad*0.001)
                    #self._w[y, x] += 1.0 - 0.5 * np.exp(-r*rad)

        """
            Initlisiert die Parameter mittels des Parametersatzes von Thomas
        """
        def thomas_init():
            self._u_o = 0#13.03
            self._tau_o1 = 33.25
            self._theta_w = 800
            self._tau_so2 = 0.85
            self._tau_v1_minus = 0.45
            self._tau_s1 = 0.04
            self._tau_w1_minus = 0.45
            self._u_s = 0.04
            self._u_w_minus = 0.45
            self._w_inf_star = 0.04

            self._tau_fi = 12.5
            self._theta_v = 1250
            self._tau_so1 = 0.13
            self._theta_o = 0.45
            self._u_so = 0.04
            self._tau_v_plus = 0.45
            self._k_s = 0.04
            self._k_w_minus = 0.45
            self._tau_winfinity = 0.04

            self._u_u = 19.6
            self._tau_o2 = 29
            self._theta_v_minus = 40
            self._k_so = 0.04
            self._tau_v2_minus = 0.45
            self._tau_s2 = 0.04
            self._tau_w2_minus = 0.45
            self._tau_si = 0.04
            self._tau_w_plus = 0.45

        """
            Initlisiert die Parameter mittels des EPI-Parametersatzes
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
            Initlisiert die Parameter mittels des PB-Parametersatzes
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
            Initlisiert die Parameter mittels desTNPP-Parametersatzes
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

        def paper_init_virtheart():
            self._u_o = 0
            self._u_u = 1.58
            self._theta_v = 0.3
            self._theta_w = 0.015
            self._theta_v_minus = 0.015
            self._theta_o = 0.006
            self._tau_v1_minus = 60
            self._tau_v2_minus = 60
            self._tau_v_plus = 1.4506
            self._tau_w1_minus = 170
            self._tau_w2_minus = 120
            self._k_w_minus = 65
            self._u_w_minus = 0.03
            self._tau_w_plus = 280
            self._tau_fi = 0.2
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
            self._tau_si = 3.8723
            self._tau_winfinity = 0.07
            self._w_inf_star = 0.94


        if parameters == "thomas":
            thomas_init()
        elif parameters == "epi":
            paper_init_epi()
        elif parameters == "pb":
            paper_init_pb()
        elif parameters == "tnpp":
            paper_init_tnnp()
        elif parameters == "virtheart":
            paper_init_virtheart()
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
        Erzeugt zufällige "Flecken" als Startdynamik
    """
    def initialize_random(self, seed, deltaX):
        np.random.seed(seed)

        n = int(np.ceil(1/deltaX))

        self._u = np.random.rand(self._Nx//n, self._Ny//n)

        tmp = np.repeat(self._u, np.ones(len(self._u), dtype=int)*n, axis=0)
        self._u = np.repeat(tmp, np.ones(len(self._u), dtype=int)*n, axis=1)

        """
        self._v = self._u.copy()
        self._v[self._v<0.4] = 0.0
        self._v[self._v>0.4] = 1.0

        self._s = self._u.copy()
        self._s[self._s<0.4] = 0.0
        self._s[self._s>0.4] = 1.0

        self._w = self._u.copy()
        self._w[self._w<0.4] = 0.0
        self._w[self._w>0.4] = 1.0
        """

    """
        Erzeugt überlappende Rechtecke um mit den TNPP-Parametern eine Spirale zu erzeugen
    """
    def initialize_spiral(self):
        Nx = self._Nx
        Ny = self._Ny

        self._u = np.zeros((Nx, Ny))
        for x in range(Nx//4) :
            for y in range(Ny):
                self._u[y, x] = 0.5
        for x in range(Nx//4*3, Nx) :
            for y in range(Ny):
                self._u[y, x] = 0.5

        self._v = np.zeros((Nx, Ny))
        for x in range(Nx//2):
            for y in range(Ny//5*2):
                self._v[y, x] = 0.5
        for x in range(Nx//2, Nx):
            for y in range(Ny//5*3, Ny):
                self._v[y, x] = 0.5
        self._w = np.ones((Nx, Ny))

        self._s = np.zeros((Nx, Ny))

    """
        Erzeugt eine Startbedingung um mit den TNPP-Parametern zwei Spiralen zu erzeugen
    """
    def initialize_double_spiral(self):
        Nx = self._Nx
        Ny = self._Ny

        self._u = np.zeros((Nx, Ny))
        for x in range(Nx//7*3, Nx//7*4) :
            for y in range(Ny//2):
                self._u[y, x] = 0.5

        self._v = np.zeros((Nx, Ny))
        for x in range(Nx):
            for y in range(Ny//2):
                self._v[y, x] = 0.5
        self._w = np.ones((Nx, Ny))

        self._s = np.zeros((Nx, Ny))

    def initialize_left_wave(self, width):
        self._u[:, 1:width+1] = 1.0

    def initialize_right_wave(self, width):
        self._u[:, -width-1:-1] = 1.0


    def _set_boundaries(self, old_fields):
        for (field, old_field) in zip((self._u, self._v, self._w, self._s), old_fields):
            field[:, 0] = old_field[:, 1]
            field[:, -1] = old_field[:, -2]
            field[0, :] = old_field[1, :]
            field[-1, :] = old_field[-2, :]


    def _laplace(self, field):
        """"
        #5point stencel

        laplace = -4*field.copy()

        laplace += np.roll(field, +1, axis=0)
        laplace += np.roll(field, -1, axis=0)
        laplace += np.roll(field, +1, axis=1)
        laplace += np.roll(field, -1, axis=1)
        """

        return filters.convolve(field, self._laplacian_matrix)

        """
        laplace = -5.0*field.copy()

        laplace += np.roll(field, +1, axis=0)/3.0*4.0
        laplace += np.roll(field, -1, axis=0)/3.0*4.0
        laplace += np.roll(field, +1, axis=1)/3.0*4.0
        laplace += np.roll(field, -1, axis=1)/3.0*4.0

        laplace -= np.roll(field, +2, axis=0)/12.0
        laplace -= np.roll(field, -2, axis=0)/12.0
        laplace -= np.roll(field, +2, axis=1)/12.0
        laplace -= np.roll(field, -2, axis=1)/12.0
        """
        #return laplace/(self._delta_x**2)

    def H(self, y):
        return (y > 0.0).astype(float)

    def _current_fi(self):
        return -self._v * self.H(self._u - self._theta_v) * (self._u - self._theta_v) * (self._u_u - self._u) / self._tau_fi

    def _current_so(self):
        return (self._u - self._u_o) * (1 - self.H(self._u - self._theta_w)) / self._tau_o + self.H(self._u - self._theta_w) / self._tau_so

    def _current_si(self):
        return -self.H(self._u - self._theta_w) * self._w * self._s / self._tau_si

    def _update_constants(self):
        self._tau_v_minus = (1.0 - self.H(self._u - self._theta_v_minus)) * self._tau_v1_minus + self.H(self._u - self._theta_v_minus) * self._tau_v2_minus
        self._tau_w_minus = self._tau_w1_minus + (self._tau_w2_minus - self._tau_w1_minus) * (1 + np.tanh(self._k_w_minus * (self._u - self._u_w_minus))) / 2.0
        self._tau_so      = self._tau_so1      + (self._tau_so2      - self._tau_so1)      * (1 + np.tanh(self._k_so      * (self._u - self._u_so)))      / 2.0
        self._tau_s = (1.0 - self.H(self._u - self._theta_w)) * self._tau_s1 + self.H(self._u - self._theta_w) * self._tau_s2
        self._tau_o = (1.0 - self.H(self._u - self._theta_o)) * self._tau_o1 + self.H(self._u - self._theta_o) * self._tau_o2

        self._v_infinity = (self._u < self._theta_v_minus).astype(float)
        self._w_infinity = (1.0 - self.H(self._u - self._theta_o)) * (1.0 - self._u / self._tau_winfinity) + self.H(self._u - self._theta_o) * self._w_inf_star

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
