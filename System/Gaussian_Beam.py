##################################################
##########                              ##########
##########           GAUSSIAN           ##########
##########             BEAM             ##########
##########                              ##########
##################################################

# TODO Header for file

# Imports
import numpy as np
from scipy import special
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from itertools import combinations, chain
from multiprocessing import Pool, cpu_count
import time

np.seterr(divide='ignore', invalid='ignore')




##################################################
##########                              ##########
##########           CLASSES            ##########
##########                              ##########
##################################################


class Hermite:
    '''
    Class representing a Hermite-Gaussian mode generated by multiple Gaussian beams.
    '''

    def __init__(self, l: int = -1, m: int = -1, amplitude: float = 1.0, phase: float = 0.0, w_0: float = 0.4, wavelength: float = 600.0, n: int = 500, resolution = 128):
        '''
        Initialise class.
        '''
        self.l = l
        self.m = m
        self.amplitude = amplitude # 0 -> 1
        self.phase = phase # -π -> π
        self.w_0 = w_0 # Waist radius
        self.wavelength = wavelength
        self.n = n

        self.z_R = (np.pi * w_0**2 * n) / wavelength # Rayleigh range
        self.k = (2 * np.pi * n) / wavelength # Wave number

        self.resolution = resolution

        self.sort_items = [self.l**2 + self.m**2, self.l, self.m] # Define the sorting index for this mode (for Superposition sorting)

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return "HG(" + str(self.l) + ", " + str(self.m) + ")"

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.l) + ", " + str(self.m) + ", " + str(round(self.amplitude, 2)) + ", " + str(round(self.phase, 2)) + ")"

    def __mul__(self, val):
        '''
        Magic method for the * operator.
        '''
        x = self.copy()
        x *= val
        return x

    def __rmul__(self, val):
        '''
        Magic method for reverse multiplication.
        '''
        return self.__mul__(val)

    def __imul__(self, val):
        '''
        Magic method for the *= operator.
        '''
        self.amplitude *= val
        return self

    def copy(self):
        '''
        Method for copying the Gaussian mode.
        '''
        return Hermite(self.l, self.m, self.amplitude, self.phase, self.w_0, self.wavelength, self.n, self.resolution)

#    def E(self, r, z):
#        '''
#        Electric field at a given radial distance and axial distance.
#        '''
#        w_ratio = self.w_0 / self.w(z)
#        exp_1 = np.exp(-(r**2) / self.w(z)**2)
#        exp_2 = np.exp(-1j * (self.k * z + self.k * (r**2 / (2 * self.R(z))) - self.phi(z)))
#
#        return np.array((w_ratio * exp_1 * exp_2) * self.amplitude * np.e**(1j*self.phase))

    def E_mode(self, x, y, z):
        '''
        Electric field amplitude at x, y, z for a given mode of order l, m.
        '''
        return np.array(self.amplitude * np.exp(1j * self.phase) * self.u(x, z, self.l) * self.u(y, z, self.m) * np.exp(-1j * self.k * z))

    def I(self, x, y, z):
        '''
        Intensity at x, y, z, for a given mode of order l, m.
        '''
        return np.abs(self.E_mode(x, y, z)**2)

    def w(self, z):
        '''
        Spot size parameter is given by a hyperbolic relation.
        '''
        return self.w_0 * np.sqrt(1 + (z / self.z_R)**2)

    def R(self, z):
        '''
        Radius of curvature.
        '''
        return z * (1 + (self.z_R / z)**2)

    def phi(self, z):
        '''
        Gouy phase is a phase advance gradually aquired by a beam around the focal region.
        '''
        return np.arctan(z / self.z_R)

    def P(self, r, z):
        '''
        Power passing through a circle of radius r in the tarnsverse plane at position z.
        '''
        return 1 - np.exp((-2 * r**2) / self.w(z)**2)

    def q(self, z):
        '''
        Complex beam parameter.
        '''
        return z + self.z_R * 1j

    def u(self, x, z, J):
        '''
        Factors for the x and y dependance.
        '''
        q0 = self.q(0)

        t1 = np.sqrt(np.sqrt(2 / np.pi) / (2**J * np.math.factorial(J) * self.w_0))
        t2 = np.sqrt(q0 / self.q(z))
        t3 = (-np.conj(self.q(z)) / self.q(z))**(J / 2)
        t4 = (np.sqrt(2) * x) / self.w(z)
        t5 = np.exp(-1j * ((self.k * x**2) / (2 * self.q(z))))

        return t1 * t2 * t3 * special.eval_hermite(J, t4) * t5

    def plot(self, save: bool = False, title: bool = True):
        '''
        Plot the Gaussian mode.
        '''
        X, Y = np.meshgrid(np.arange(-1.2, 1.2, 2.4 / self.resolution), np.arange(-1.2, 1.2, 2.4 / self.resolution))

        plt.figure(self.__class__.__name__)
        plt.imshow(self.I(X, Y, 0), cmap='jet')

        if title: plt.title(str(self))
        plt.axis('off')

        if save: plt.savefig("Images/" + str(self) + ".png", bbox_inches='tight', pad_inches=0)
        else: plt.show()

    def add_phase(self, phase):
        '''
        Adds some phase value to current mode phase
        '''
        self.phase += phase # Adding extra phase

        if self.phase < -np.pi: # Ensuring phase stays within -π -> π
            self.phase = self.phase % np.pi
        elif self.phase > np.pi:
            self.phase = self.phase % -np.pi




class Superposition(list):
    '''
    Class repreenting a superposition of multiple Gaussian modes.
    '''

    def __init__(self, *modes):
        '''
        Initialise the class with the list of modes that compose the superposition.
        '''
        self.noise_variation = 0.0
        self.exposure = (0.0, 1.0)

        # Merge Laguerre modes into the Hermite basis set
        mode_list = []
        for mode in modes:
            if type(mode) == Hermite: # Process Hermite modes
                mode_list.append(mode)
            elif type(mode) == Laguerre: # Process Laguerre modes in Hermite basis
                mode_list.extend(mode)

        # Merge multiples of the same mode
        sorted_modes = sorted(mode_list, key=lambda x: (x.sort_items[0], x.sort_items[1], x.sort_items[2])) # [mode.copy() for mode in modes] # Create duplicate of Gaussian modes for random normalised ampltidues
        self.modes = []
        for mode in sorted_modes:
            existing_mode = [x for x in self.modes if (x.l == mode.l and x.m == mode.m)]
            if len(existing_mode) == 0: # No identical matches
                self.modes.append(mode)
            else: # Duplicate exists
                self.add_modes(existing_mode[0], mode) # Merge duplicate

        super().__init__(self.modes)
        self.resolution = self.modes[0].resolution

        amplitudes = [i.amplitude for i in self]
        normalised_amplitudes = amplitudes / np.linalg.norm(amplitudes) # Normalise the amplititudes

        for i in range(len(self)): self[i].amplitude = normalised_amplitudes[i] # Set the normalised amplitude variations to the modes

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return "S" + str(tuple([str(i) for i in self])).replace("'", "")

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.modes)[1:-1] + ")"

    def __mul__(self, value):
        '''
        Magic method for the * operator.
        '''
        x = Superposition(*self.modes)
        x *= value
        return x

    def __rmul__(self, value):
        '''
        Magic method for reverse multiplication.
        '''
        return self.__mul__(value)

    def __imul__(self, value):
        '''
        Magic method for the *= operator.
        '''
        for i in self: i.amplitude *= value
        return self

    def contains(self, mode: Hermite):
        '''
        Returns the mode that exists within the superposition.
        '''
        # return int(str(mode) in str(self))
        for i in self:
            if str(i) == str(mode): # Mode of correct l and m values exists in the superposition
                return i

        return Hermite(l=0, m=0, amplitude=0.0)

    def plot(self, save: bool = False, title: bool = True, constituents: bool = False):
        '''
        Plot the superposition.
        '''
        if constituents: [i.show() for i in self] # Plot the constituent Gaussian modes

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})

        ax1.imshow(self.superpose(), cmap='jet')
        ax2.imshow(self.phase_map(), cmap='jet')

        if title: fig.suptitle(str(self))
        ax1.axis('off')
        ax2.axis('off')

        if save: plt.savefig("Images/" + str(self) + ".png", bbox_inches='tight', pad_inches=0)
        else: plt.show()

    def superpose(self):
        '''
        Compute the superposition of the Gaussian modes.
        '''
        X, Y = np.meshgrid(np.arange(-1.2, 1.2, 2.4 / self.resolution), np.arange(-1.2, 1.2, 2.4 / self.resolution))

        superposition = np.abs(sum([i.E_mode(X, Y, 0) for i in self])**2)

        return superposition / np.linalg.norm(superposition) # Normalise the superposition

    def phase_map(self):
        '''
        Returns the phase map for the superposition.
        '''
        X, Y = np.meshgrid(np.arange(-1.2, 1.2, 2.4 / self.resolution), np.arange(-1.2, 1.2, 2.4 / self.resolution))

        superposition = sum([i.E_mode(X, Y, 0) for i in self])**2

        return np.arctan(np.imag(superposition / np.real(superposition)))

    def add_modes(self, mode, new_mode):
        '''
        Combine mode amplitudes and phases in the event of Hermite duplicates.
        '''
        r1 = mode.amplitude
        r2 = new_mode.amplitude
        phi1 = mode.phase
        phi2 = new_mode.phase

        mode.amplitude = np.sqrt(r1**2 + r2**2 + 2 * r1 * r2 * np.cos(phi1 - phi2))
        mode.add_phase(np.arctan((r1 * np.sin(phi1) + r2 * np.sin(phi2)) / (r1 * np.cos(phi1) + r2 * np.cos(phi2))) - mode.phase)

        return mode




class Laguerre(Superposition):
    '''
    Class representing a Laguerre mode generated as a superposition of multiple Gaussian modes.
    '''

    def __init__(self, p: int = 1, m: int = 1, amplitude: float = 1.0, phase: float = 0.0, resolution: int = 128):
        '''
        Initialise the class with the order (p, m) of the Laguerre mode.
        '''
        self.p = p
        self.m = m
        self.amplitude = amplitude
        self.phase = phase
        self.resolution = resolution
        self.sort_items = [self.p**2 + self.m**2, self.p, self.m] # Define the sorting index for this mode (for Superposition sorting)

        # From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=247715

        self.modes = []
        for q in range(p + 1):
            for s in range(int((m)/2) + 1):
                frac = ((fact(2*(q-s) + m)*fact(2*(p-q+s)))/(2**(2*p + m - 1) * fact(p) * fact(p+m) * (1 + choose(0, m))))**0.5
                y = Hermite(2*(q - s) + m, 2*(p - q + s))
                y.amplitude = choose(p, q) * choose(m, 2*s) * frac
                y.add_phase((s+p)*np.pi)
                self.modes.append(y)

        super().__init__(*self.modes)

        for mode in self.modes:
            mode *= self.amplitude
            mode.add_phase(self.phase) # Propagates total Laguerre amp and phase to each constituent mode

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.p) + ", " + str(self.m) + ")"

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.p) + ", " + str(self.m) + ", " + str(round(self.amplitude, 2)) + ", " + str(round(self.phase, 2)) + ")"

    def __mul__(self, value):
        '''
        Magic method for the * operator.
        '''
        x = self.copy()
        x *= value
        return x

    def copy(self):
        '''
        Make a copy of the object.
        '''
        return Laguerre(self.p, self.m, self.amplitude, self.phase, self.resolution)

    def add_phase(self, phase):
        '''
        Add phase to superposition, and propagate down to component modes.
        '''
        self.phase += phase # Adding extra phase

        if self.phase < -np.pi: # Ensuring phase stays within -π -> π
            self.phase = self.phase % np.pi
        elif self.phase > np.pi:
            self.phase = self.phase % -np.pi

        [mode.add_phase(phase) for mode in self.modes] # Propogate phase to constituent modes

    def E_mode(self, x, y, z):
        '''
        Electric field amplitude at x, y, z for the superposition.
        '''
        return sum([i.E_mode(x, y, z) for i in self])




##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################
def fact(x):
    res = 1
    if x != 0:
        for i in range(x):
            res *= i + 1
    return res

def choose(n, r):
    return fact(n)/(fact(r)*fact(n-r))




##################################################
##########                              ##########
##########            MAIN              ##########
##########                              ##########
##################################################


if __name__ == '__main__':
     x = Superposition(Hermite(0,1,0.2,0), Hermite(1,2,0.6,np.pi/2), Hermite(2,1,0.4,0))
     print(x)
     x.plot()



##################################################
##########                              ##########
##########           TESTING            ##########
##########                              ##########
##################################################


# def process(x):
#     print("Process started for '" + x[0] + "' for k = " + str(x[1]) + " for event " + str(x[2]) + "...")
#     fh.write_data(x[0], x[2], x[1])

# def pool_handler(x, t):
#     p = Pool(t)
#     p.map(process, x)

# if __name__ == '__main__':
#     d = input("File: "š
#     k = input("k: ")
#     t = input("Threads: ")

#     x = [(str(d), int(k), int(e)) for e in range(9999)]
#     pool_handler(x, int(t))




# TODO Animate Hermite-Gaussian modes as a GIF

# data = []
# for i in tqdm(range(1, 500)):
#     mode = Hermite(0.4, 600, i)
#     data.append(mode.E(X, Y))

# def update(data):
#     mat.set_data(data)
#     return mat

# def animate():
#     for i in data: yield i # Animate the states of the game

# mat = ax.imshow(np.real(mode.E(X, Y)))
# anim = animation.FuncAnimation(fig, update, animate, interval=100, save_count=50) # Create the animation for state evolutio using Markov Chain