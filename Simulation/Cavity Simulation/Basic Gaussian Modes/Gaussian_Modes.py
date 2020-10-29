##################################################
##########                              ##########
##########           GAUSSIAN           ##########
##########             MODES            ##########
##########                              ##########
##################################################

# TODO Header for file

# Imports
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from itertools import combinations




##################################################
##########                              ##########
##########           CLASSES            ##########
##########                              ##########
##################################################


class Gaussian_Mode:
    '''
    Class representing a Hermite-Gaussian mode generated by multiple Gaussian beams.
    '''

    def __init__(self, l: int = -1, m: int = -1, w_0: float = 0.4, wavelength: float = 600.0, n: int = 500):
        '''
        Initialise class.
        '''
        self.w_0 = w_0 # Waist radius
        self.z_R = (np.pi * w_0**2 * n) / wavelength # Rayleigh range
        self.k = (2 * np.pi * n) / wavelength # Wave number

        self.l = l
        self.m = m
    
    def __str__(self):
        '''
        Magic method for str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.l) + ", " + str(self.m) + ")"
    
    def __repr__(self):
        '''
        Magic method for repr() function.
        '''
        return str(self)

    def E(self, r, z):
        '''
        Electric field at a given radial distance and axial distance.
        '''
        w_ratio = self.w_0 / self.w(z)
        exp_1 = np.exp(-(r**2) / self.w(z)**2)
        exp_2 = np.exp(-1j * (self.k * z + self.k * (r**2 / (2 * self.R(z))) - self.phi(z)))

        return w_ratio * exp_1 * exp_2
    
    def E_mode(self, x, y, z):
        '''
        Electric field amplitude at x, y, z for a given mode of order l, m.
        '''
        return self.u(x, z, self.l) * self.u(y, z, self.m) * np.exp(-1j * self.k * z)
    
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
        q0 = self.q(0) # ???
        
        t1 = np.sqrt(np.sqrt(2 / np.pi) / (2**J * np.math.factorial(J) * self.w_0))
        t2 = np.sqrt(q0 / self.q(z))
        t3 = (-np.conj(self.q(z)) / self.q(z))**(J / 2)
        t4 = (np.sqrt(2) * x) / self.w(z)
        t5 = np.exp(-1j * ((self.k * x**2) / (2 * self.q(z))))

        return t1 * t2 * t3 * special.eval_hermite(J, t4) * t5
    
    def plot(self, title: bool = True):
        '''
        Plot the Gaussian mode.
        '''
        X, Y = np.meshgrid(np.arange(-1.2, 1.2, 0.01), np.arange(-1.2, 1.2, 0.01))
        plt.figure(self.__class__.__name__)

        plt.imshow(np.abs(np.abs(self.E_mode(X, Y, 0))), cmap='Greys_r')

        if title: plt.title(str(self))
        plt.axis('off')
    
    def show(self, title: bool = True):
        '''
        Show the plot of the Gaussian mode.
        '''
        self.plot(title)
        plt.show()
    
    def save(self, title: bool = True):
        '''
        Save the plot of the Gaussian mode.
        '''
        self.plot(title)
        plt.savefig("Images/" + str(self) + ".png", bbox_inches='tight', pad_inches=0)




class Superposition(list):
    '''
    Class repreenting a superposition of multiple Gaussian modes.
    '''

    def __init__(self, modes: list):
        '''
        Initialise the class with the list of modes that compose the superposition.
        '''
        super().__init__(modes)
        self.modes = modes

        X, Y = np.meshgrid(np.arange(-1.2, 1.2, 0.01), np.arange(-1.2, 1.2, 0.01))
        self.superposition = np.abs(sum([i.E_mode(X, Y, 0) for i in self]))
    
    def __str__(self):
        '''
        Magic method for str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.modes) + ")"
    
    def __repr__(self):
        '''
        Magic method for repr() function.
        '''
        return str(self)

    def plot(self, title: bool = True):
        '''
        Plot the superposition.
        '''
        # for i in self: i.plot() # Plot the constituent Gaussian modes
        plt.figure(self.__class__.__name__)

        plt.imshow(self.superposition, cmap='Greys_r')

        if title: plt.title(str(self))
        plt.axis('off')
    
    def show(self, title: bool = True):
        '''
        Show the plot of the Gaussian mode.
        '''
        self.plot(title)
        plt.show()
    
    def save(self, title: bool = True):
        '''
        Save the plot of the Gaussian mode.
        '''
        self.plot(title)
        plt.savefig("Images/" + str(self) + ".png", bbox_inches='tight', pad_inches=0)




class Generate_Data(list):
    '''
    Class representing many superpositions of multiple Guassian modes at a specified complexity.
    '''
    
    def __init__(self, modes: int = 1, complexity: int = 1):
        '''
        Initialise the class with the required complexity.

        'modes': Max mode number for superpositions.
        'complexity': How many modes you want to superimpose together.
        '''
        self.modes = modes
        self.complexity = complexity

        print("\n-----| Generating Data for " + str(modes) + " Modes of Complexity " + str(complexity) + " |-----\n")
        print("Generating Gaussian modes...")

        gauss_modes = [Gaussian_Mode(l=i, m=j) for i in range(modes) for j in range(modes)]

        print("Done! Found " + str(len(gauss_modes)) + " modes.\n\nGenerating superpositions...")

        combs = list(combinations(gauss_modes, complexity))
        superposition_combinations = []
        for i in tqdm(range(len(combs))): superposition_combinations.append(Superposition(combs[i]))

        super().__init__(superposition_combinations)
        
        print("Done! Found " + str(len(combs)) + " combinations.\n")
    
    def __str__(self):
        '''
        Magic method for str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.modes) + ", " + str(self.complexity) + ")"
    
    def __repr__(self):
        '''
        Magic method for repr() function.
        '''
        return str(self)

    def plot(self):
        '''
        Plot all superpositions generated.
        '''
        for i in self: i.plot()




##################################################
##########                              ##########
##########            MAIN              ##########
##########                              ##########
##################################################


def generate_modes(ls, ms):
    '''
    Generate and save modes from (0,0) to (ls,ms).
    '''
    for l in range(ls):
        for m in tqdm(range(ms)):
            X, Y = np.meshgrid(np.arange(-1.2, 1.2, 0.01), np.arange(-1.2, 1.2, 0.01))
            
            mode = Gaussian_Mode(0.4, 600, 500, l, m)

            plt.imshow(np.abs(mode.E_mode(X, Y, 0)), cmap='Greys_r')
            plt.axis('off')
            plt.savefig("Images/" + str(l) + str(m) + ".png", bbox_inches='tight', pad_inches=0)





##################################################
##########                              ##########
##########           TESTING            ##########
##########                              ##########
##################################################


x = Generate_Data(5, 3)

print(x[1000])
x[1000].show()
x[1000].save(False)




# x = np.arange(-1.2, 1.2, 0.01)
# y = np.arange(-1.2, 1.2, 0.01)

# X, Y = np.meshgrid(x, y)

# fig, ax = plt.subplots()

# Testing a superposition of 3 basic modes.

# mode = Gaussian_Mode(0.4, 600, 500)

# plt.imshow(np.abs(mode.E2(X, Y, 0, 1, 0) + mode.E2(X, Y, 0, 2, 2) + mode.E2(X, Y, 0, 4, 1)), cmap='Greys_r')
# plt.axis('off')
# plt.savefig("Images/superposition.png", bbox_inches='tight', pad_inches=0)

# Generating and saving all mode combinations from (0,0) to (5,5).

# generate_modes(5, 5)




# TODO Animate Hermite-Gaussian modes as a GIF

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