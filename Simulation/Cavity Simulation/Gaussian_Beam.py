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

    def __init__(self, l: int = -1, m: int = -1, amplitude: float = 1.0, phase: float = 0.0, w_0: float = 0.4, wavelength: float = 600.0, n: int = 500):
        '''
        Initialise class.
        '''
        self.l = l
        self.m = m
        self.amplitude = amplitude
        self.phase = phase
        self.w_0 = w_0 # Waist radius
        self.wavelength = wavelength
        self.n = n

        self.z_R = (np.pi * w_0**2 * n) / wavelength # Rayleigh range
        self.k = (2 * np.pi * n) / wavelength # Wave number

        self.resolution = 50

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.l) + ", " + str(self.m) + ")"

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.l) + ", " + str(self.m) + ", " + str(self.amplitude) + ", " + str(self.phase) + ")"

    def __mul__(self, val):
        '''
        Magic method for the * operator.
        '''
        x = self.copy()
        x *= val
        return x
    
    def __rmul__(self, val):
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
        return Hermite(self.l, self.m, self.amplitude, self.phase, self.w_0, self.wavelength, self.n)

    def plot(self, title: bool = True):
        '''
        Plot the Gaussian mode.
        '''
        X, Y = np.meshgrid(np.arange(-1.2, 1.2, 1.0 / self.resolution), np.arange(-1.2, 1.2, 1.0 / self.resolution))

        plt.figure(self.__class__.__name__)
        plt.imshow(self.I(X, Y, 0), cmap='Greys_r')

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

    def add_phase(self, phase):
        '''
        Adds some phase value to current mode phase
        '''
        self.phase += phase
        self.phase = self.phase % (2 * np.pi)




class Superposition(list):
    '''
    Class repreenting a superposition of multiple Gaussian modes.
    '''

    def __init__(self, modes: list, amplitude: float = 1.0, phase: float = 0.0):
        '''
        Initialise the class with the list of modes that compose the superposition.
        '''
        self.amplitude = amplitude
        self.phase = phase

        mode_list = []
        for mode in modes:
            if type(mode) == Hermite: # Process Hermite modes
                mode_list.append(mode)
            elif type(mode) == Laguerre: # Process Laguerre modes in Hermite basis
                mode_list.extend(mode)
        
        self.modes = mode_list # [mode.copy() for mode in modes] # Create duplicate of Gaussian modes for random normalised ampltidues
        self.resolution = modes[0].resolution
        self.amplitude = amplitude
        self.phase = phase

        super().__init__(self.modes)

        amplitudes = [i.amplitude for i in self]
        normalised_amplitudes = amplitudes / np.linalg.norm(amplitudes) # Normalise the amplititudes

        for i in range(len(self)): 
            self[i].amplitude = round(normalised_amplitudes[i], 2) # Set the normalised amplitude variations to the modes
            self[i].add_phase(phase)

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.modes)[1:-1] + ")"

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.modes) + ", " + str(self.amplitude) + ", " + str(self.phase) + ")"

    def __mul__(self, value):
        '''
        Magic method for the * operator.
        '''
        x = Superposition(self.modes)
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
        for i in self:
            i.amplitude *= value
        return self

    def plot(self, title: bool = True):
        '''
        Plot the superposition.
        '''
        plt.figure(self.__class__.__name__)
        plt.imshow(self.superpose(), cmap='Greys_r')

        if title: plt.title(str(self))
        plt.axis('off')

    def show(self, title: bool = True, constituents: bool = False):
        '''
        Show the plot of the Gaussian mode.
        '''
        if constituents: [i.show() for i in self] # Plot the constituent Gaussian modes

        self.plot(title)
        plt.show()

    def save(self, title: bool = True):
        '''
        Save the plot of the Gaussian mode.
        '''
        self.plot(title)
        plt.savefig("Images/" + str(self) + ".png", bbox_inches='tight', pad_inches=0)

    def superpose(self):
        '''
        Compute the superposition of the Gaussian modes.
        '''
        X, Y = np.meshgrid(np.arange(-1.2, 1.2, 1.0 / self.resolution), np.arange(-1.2, 1.2, 1.0 / self.resolution))

        superposition = np.abs(sum([i.E_mode(X, Y, 0) for i in self])**2)
        # superposition = np.abs(self.E_mode(X, Y, 0)**2)

        return superposition / np.linalg.norm(superposition) # Normalise the superposition
    
    def add_phase(self, phase):
        '''
        Add phase to Superposition, and propagate down to component modes.
        '''
        self.phase += phase
        self.phase = self.phase % 2 * np.pi
        [mode.add_phase(phase) for mode in self.modes]




class Laguerre(Superposition):
    '''
    Class representing a Laguerre mode generated as a superposition of multiple Gaussian modes.
    '''

    def __init__(self, p: int = 1, m: int = 1, amplitude: float = 1.0, phase: float = 0.0):
        '''
        Initialise the class with the order (p, m) of the Laguerre mode.
        '''
        self.p = p
        self.m = m
        self.amplitude = amplitude
        self.phase = phase

        # From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=247715

        self.modes = []
        for q in range(p + 1):
            for s in range(int((m)/2) + 1):
                frac = ((fact(2*(q-s) + m)*fact(2*(p-q+s)))/(2**(2*p + m - 1) * fact(p) * fact(p+m) * (1 + choose(0, m))))**0.5
                y = Hermite(2*(q - s) + m, 2*(p - q + s))
                y.amplitude = (-1)**(s+p) * choose(p, q) * choose(m, 2*s) * frac
                self.modes.append(y)

        super().__init__(self.modes)

        for mode in self.modes:
            mode *= self.amplitude
            mode.add_phase(self.phase) #Propagates total Laguerre amp and phase to each constituent mode

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.p) + ", " + str(self.m) + ")"

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.p) + ", " + str(self.m) + ", " + str(self.amplitude) + ", " + str(self.phase) + ")"

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
        return Laguerre(self.p, self.m, self.amplitude, self.phase)

    def E_mode(self, x, y, z):
        '''
        Electric field amplitude at x, y, z for the superposition.
        '''
        return sum([i.E_mode(x, y, z) for i in self])




class Generate_Data(list):
    '''
    Class representing many superpositions of multiple Guassian modes at a specified complexity.
    '''

    def __init__(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0.0, phase_variation: float = 0.0, noise_variation: float = 0.0, exposure: tuple = (0.0, 1.0), repeats: int = 1, info: bool = True):
        '''
        Initialise the class with the required complexity.

        'max_order': Max order of Guassian modes in superpositions (x > 0).
        'number_of_modes': How many modes you want to superimpose together (x > 0).
        'ampiltude_variation': How much you want to vary the amplitude of the Gaussian modes by (x > 0).
        '''
        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.phase_variation = phase_variation
        self.noise_variation = noise_variation
        self.exposure = exposure
        self.repeats = repeats

        if info: print("\n_____| Generating Data |_____\n")
        if info: print("Max order of mode: " + str(max_order) + "\nNumber of modes in superposition: " + str(number_of_modes) + "\nVariation in mode amplitude: " + str(amplitude_variation) + "\nVariation in mode phase: "
                        + str(phase_variation) + "\nVariation in saturation noise: " + str(noise_variation) + "\nVariation in saturation exposure: " + str(exposure) + "\nRepeats of combinations: " + str(repeats) + "\n")
        if info: print("Generating Gaussian modes...")

        hermite_modes = [Hermite(l=i, m=j) for i in range(max_order) for j in range(max_order)]
        #laguerre_modes = [Laguerre(p=i, m=j) for i in range(max_order) for j in range(max_order)]
        self.gauss_modes = hermite_modes# + laguerre_modes

        if info: print("Done! Found " + str(len(self.gauss_modes)) + " gaussian modes.\n\nGenerating superpositions...")

        self.combs = [list(combinations(self.gauss_modes, i)) for i in range(1, number_of_modes + 1)]
        self.combs = [i[j] for i in self.combs for j in range(len(i))]

        super().__init__()

        p = Pool(cpu_count())
        self.extend(p.map(self.generate_process, self.combs * repeats))

        if info: print("Done! Found " + str(len(self)) + " combinations.\n")

    def generate_process(self, item):
        '''
        Process for generating superposition objects across multiple threads in the CPU.
        '''
        randomised_item = [self.randomise_amp_and_phase(i) for i in item]
        return Superposition(randomised_item)

    def show(self, title: bool = True):
        '''
        Plot and show all superpositions generated.
        '''
        for i in self: i.show(title)

    def save(self, title: bool = True):
        '''
        Plot and save all superpositions generated.
        '''
        print("Saving dataset...")

        p = Pool(cpu_count())
        p.map(self.save_process, self)

        print("Done!\n")

    def save_process(self, data):
        '''
        Process for saving images of the dataset across multiple threads in the CPU.
        '''
        data.save(False)
    
    def get_inputs(self, desc: str = None):
        '''
        Get all the superpositions for the dataset.
        '''
        # return np.array([i.superpose() for i in tqdm(self, desc)])[..., np.newaxis]

        p = Pool(cpu_count())
        n = len(self) // (cpu_count() - 1)
        jobs = [self[i:i + n] for i in range(0, len(self), n)]
        threads = p.map(self.superpose_process, tqdm(jobs, desc))
        return np.array([item for item in chain(*threads)])[..., np.newaxis]

    def superpose_process(self, data):
        '''
        Process for superposing elements of the dataset across multiple threrads in the CPU.
        '''
        return [add_exposure(add_noise(item.superpose(), self.noise_variation), self.exposure) for item in data]

    def get_outputs(self):
        '''
        Get all possible Gaussian modes that could comprise a superposition.
        '''
        return np.array(self.repeats * [[int(str(j)[:-1] in str(i))  for j in self.gauss_modes] for i in self.combs])

    def get_classes(self):
        '''
        Get the num_classes result required for model creation.
        '''
        return np.array(self.gauss_modes, dtype=object) # * self.repeats
    
    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()

        x *= np.abs(round(np.random.normal(scale=self.amplitude_variation), 2) + 1)
        x.add_phase(np.abs(round(np.random.normal(scale=self.phase_variation), 2)))

        return x


    # def pool_handler(self, data, threads):
    #     '''

    #     '''
    #     p = Pool(threads)
    #     p.map(self.process, data)

    # def process(self, data):
    #     '''

    #     '''
    #     # print(self.combs.index(data), data)
    #     data[0].append(Superposition(data[1], self.amplitude_variation))

    # def __str__(self):
    #     '''
    #     Magic method for str() function.
    #     '''
    #     return self.__class__.__name__ + "(" + [self.__getitem__(i) for i in range(len(self))] + ")"

    # def __repr__(self):
    #     '''
    #     Magic method for repr() function.
    #     '''
    #     return self.__class__.__name__ + "(" + str(self.max_order) + ", " + str(self.number_of_modes) + ", " + str(self.amplitude_variation) + ")"




##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################


def randomise_amplitudes(mode_list, variance):
    amplitudes = np.zeros((len(mode_list)))
    new_modes = []
    for i in range(len(mode_list)):
        amplitudes[i] = abs(round(np.random.normal(scale=variance), 2) + 1) # Make randomised amplitude based on normal distribution
    amplitudes /= np.linalg.norm(amplitudes) # Normalise amplitudes
    for i, mode in enumerate(mode_list):
        new_modes.append(mode * amplitudes[i])
    return new_modes

def unpack_and_superpose(mode_list):
    modes = []
    for mode in mode_list:
        if type(mode) in [Superposition, Laguerre]:
            modes.extend(mode)
        elif type(mode) == Hermite:
            modes.append(mode)
    return Superposition(modes)
        
def fact(x):
    res = 1
    if x != 0:
        for i in range(x):
            res *= i + 1
    return res

def choose(n, r):
    return fact(n)/(fact(r)*fact(n-r))

def add_noise(image, noise_variance: float = 0.0):
    '''
    Adds random noise to a copy of the image according to a normal distribution of variance 'noise_variance'.
    Noise Variance defined as a %age of maximal intensity
    '''
    max_val = np.max(image)
    norm = lambda i: np.random.normal(loc=i, scale=noise_variance*max_val)
    f = np.vectorize(norm)
    image = f(image)
    return image

def add_exposure(image, exposure:tuple = (0.0, 1.0)):
    '''
    Adds in exposure limits to the image, using percentile limits defined by exposure.
    exposure[0] is the x% lower limit of detection, exposure[1] is the upper.
    Percents calculated as a function of the maximum image intensity.
    '''
    max_val = np.max(image)
    lower_bound = max_val * exposure[0]
    upper_bound = max_val * exposure[1]
    exp = np.vectorize(exposure_comparison)
    image = exp(image, upper_bound, lower_bound)
    return image
    
def exposure_comparison(val, upper_bound, lower_bound):
    if val > upper_bound:
        val = upper_bound
    elif val < lower_bound:
        val = lower_bound
    return val

##################################################
##########                              ##########
##########            MAIN              ##########
##########                              ##########
##################################################


if __name__ == '__main__':
    

     x1 = Hermite(0,1)
     x2 = Hermite(1,0)
     x = Superposition([x1,x2])
     
     im = add_exposure(add_noise(x.superpose(), 0.00), (0.2, 0.5))
     plt.imshow(im, cmap='Greys_r')
     plt.show()
     x.plot()

     for i in range(len(im[:, 0])):
         for j in range(len(im[0, :])):
             if type(im[i, j]) != np.float64:
                 print(type(im[i, j]))
    # x = Generate_Data(5, 2, 0.0)
    # x.save(False)




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
#     d = input("File: ")
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