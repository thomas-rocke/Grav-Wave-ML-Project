import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

class Gaussian_Mode:
    '''
    Class representing a Hermite-Gaussian mode.
    '''

    def __init__(self, w0, wavelength, n):
        '''
        Initialise class.
        '''
        self._w0 = w0 # Waist radius
        self._zR = (np.pi * w0**2 * n) / wavelength # Rayleigh range
        self._k = (2 * np.pi * n) / wavelength # Wave number

    def E(self, r, z):
        '''
        Electric field at a given radial distance and axial distance.
        '''
        w_ratio = self._w0 / self.w(z)
        exp_1 = np.exp(-(r**2) / self.w(z)**2)
        exp_2 = np.exp(-1j * (self._k * z + self._k * (r**2 / (2 * self.R(z))) - self.phi(z)))

        return w_ratio * exp_1 * exp_2
    
    def w(self, z):
        '''
        Spot size parameter is given by a hyperbolic relation.
        '''
        return self._w0 * np.sqrt(1 + (z / self._zR)**2)
    
    def R(self, z):
        '''
        Radius of curvature.
        '''
        return z * (1 + (self._zR / z)**2)
    
    def phi(self, z):
        '''
        Gouy phase is a phase advance gradually aquired by a beam around the focal region.
        '''
        return np.arctan(z / self._zR)
    
    def P(self, r, z):
        '''
        Power passing through a circle of radius r in the tarnsverse plane at position z.
        '''
        return 1 - np.exp((-2 * r**2) / self.w(z)**2)
    
    def q(self, z):
        '''
        Complex beam parameter.
        '''
        return z + self._zR * 1j
    
    def u(self, x, z, J):
        '''
        Factors for the x and y dependance.
        '''
        q0 = self.q(z) # ???
        
        t1 = np.sqrt(np.sqrt(2 / np.pi) / (2**J * np.math.factorial(J) * self._w0))
        t2 = np.sqrt(q0 / self.q(z))
        t3 = (-np.conj(self.q(z)) / self.q(z))**(J / 2)
        t4 = (np.sqrt(2) * x) / self.w(z)
        t5 = np.exp(-1j * ((self._k * x**2) / (2 * self.q(z))))

        return t1 * t2 * t3 * special.eval_hermite(J, t4) * t5
    
    def E2(self, x, y, z, l, m):
        '''
        Electric field amplitude at x, y, z for a given mode of order l, m.
        '''
        return self.u(x, z, l) * self.u(y, z, m) * np.exp(-1j * self._k * z)






# Testing

x = np.arange(-1.2, 1.2, 0.01)
y = np.arange(-1.2, 1.2, 0.01)

X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()

# data = []
# for i in tqdm(range(1, 500)):
#     mode = Gaussian_Mode(0.4, 600, i)
#     data.append(mode.E(X, Y))

# def update(data):
#     mat.set_data(data)
#     return mat

# def animate():
#     for i in data: yield i # Animate the states of the game

# mat = ax.imshow(np.real(mode.E(X, Y)))
# anim = animation.FuncAnimation(fig, update, animate, interval=100, save_count=50) # Create the animation for state evolutio using Markov Chain

mode = Gaussian_Mode(0.4, 600, 500)

plt.imshow(np.abs(mode.E2(X, Y, 0, 1, 0) + mode.E2(X, Y, 0, 2, 2) + mode.E2(X, Y, 0, 4, 1)), cmap='Greys_r')

plt.axis('off')
plt.show()

for i in range(5):
    for j in tqdm(range(5)):
        mode = Gaussian_Mode(0.4, 600, 500)

        plt.imshow(np.abs(mode.E2(X, Y, 0, i, j)), cmap='Greys_r')

        plt.axis('off')
        plt.savefig("Images/" + str(i) + str(j) + ".png", bbox_inches='tight', pad_inches=0)