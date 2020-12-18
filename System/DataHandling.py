import numpy as np
from scipy import special
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from itertools import combinations, chain, zip_longest, islice
from multiprocessing import Pool, cpu_count
import time
import random
import sys
import os
from Utils import meanError
from Gaussian_Beam import Hermite, Superposition, Laguerre


def saveData(superposList, fname, path=None):    
    if(path==None):
        path = os.getcwd() + r'\Simulation\Cavity Simulation\Data'
    file = path + os.sep + fname

    superposStrings = [repr(superpos) + '\n' for superpos in superposList]
    with open(file, 'w') as f:
        f.writelines(superposStrings)
        




def loadData(fname, path=None):
   import numpy as np
   if(path==None):
        path = os.getcwd() + r'\Simulation\Cavity Simulation\Data'
   file = path + os.sep + fname
    
   with open(file, 'r') as f:
       superText = np.genfromtxt(f, delimiter='\n', dtype=str)
   supers = [eval(superposition) for superposition in superText]
   return supers




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

        if info: print("_____| Generating Data |_____\n")
        if info: print("Max order of mode: " + str(max_order) + "\nNumber of modes in superposition: " + str(number_of_modes) + "\nVariation in mode amplitude: " + str(amplitude_variation) + "\nVariation in mode phase: "
                        + str(phase_variation) + "\nVariation in saturation noise: " + str(noise_variation) + "\nVariation in saturation exposure: " + str(exposure) + "\nRepeats of combinations: " + str(repeats) + "\n")
        if info: print("Generating Gaussian modes...")

        self.hermite_modes = [Hermite(l=i, m=j) for i in range(max_order) for j in range(max_order)]
        self.laguerre_modes = [Laguerre(p=i, m=j) for i in range(max_order // 2) for j in range(max_order // 2)]
        self.gauss_modes = self.hermite_modes + self.laguerre_modes

        if info: print("Done! Found " + str(len(self.hermite_modes)) + " hermite modes and " + str(len(self.laguerre_modes)) + " laguerre modes giving a total of " + str(len(self.gauss_modes)) + " gaussian modes.\n\nGenerating superpositions...")

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
        # Generate superposition with random amplitude and phase modes
        randomised_item = [self.randomise_amp_and_phase(i) for i in item]
        sup = Superposition(*randomised_item)

        # Set noise variation and exposure correctly
        sup.noise_variation = self.noise_variation
        sup.exposure = self.exposure

        return sup

    def plot(self, save: bool = False, title: bool = True):
        '''
        Plot and show / save all superpositions generated.
        '''
        if save:
            print("Saving dataset...")

            p = Pool(cpu_count())
            p.map(self.save_process, self)

            print("Done!\n")

        else:
            for i in self: i.plot(title=title)

    def save_process(self, data):
        '''
        Process for saving images of the dataset across multiple threads in the CPU.
        '''
        data.plot(save=True)

    def get_inputs(self, desc: str = None):
        '''
        Get all the superpositions for the dataset.
        '''
        # return np.array([i.superpose() for i in tqdm(self, desc)])[..., np.newaxis]

        # Below is support for multiprocessing of superpositions, however it is limited by disk speed and can cause memory overflow errors

        p = Pool(cpu_count())
        n = len(self) // (cpu_count() - 1)
        jobs = [self[i:i + n] for i in range(0, len(self), n)]
        threads = p.map(self.superpose_process, tqdm(jobs, desc))
        return np.array([item for item in chain(*threads)])[..., np.newaxis]

    def superpose_process(self, data):
        '''
        Process for superposing elements of the dataset across multiple threrads in the CPU.
        '''
        return [item.superpose() for item in data]

    def get_outputs(self):
        '''
        Get all possible Gaussian modes that could comprise a superposition.
        '''
        # return np.array(self.repeats * [[int(str(j)[:-1] in str(i)) * 0.5 for j in self.gauss_modes] for i in self.combs])

        # return np.array([[int(i.contains(j).amplitude and round(i.contains(j).phase / (2 * np.pi), 1) == p / 10) for j in self.hermite_modes for p in range(11)] for i in self]) # Phase via probability distribution

        return np.array([[i.contains(j).amplitude for j in self.hermite_modes] + [np.cos(i.contains(j).phase) for j in self.hermite_modes] for i in self]) # + [(i.contains(j).phase / (2 * np.pi)) % 1 for j in self.hermite_modes]

    def get_classes(self):
        '''
        Get the num_classes result required for model creation.
        '''
        # tmp = []
        # for i in range(11):
        #     for j in self.hermite_modes:
        #         mode = j.copy()
        #         mode.phase = (i / 10) * (2 * np.pi)
        #         tmp.append(mode)

        # return np.array(tmp, dtype=object)

        return np.array(self.hermite_modes * 2, dtype=object)

    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()

        x *= np.abs(np.random.normal(scale=self.amplitude_variation) + 1)
        x.add_phase(np.random.normal(scale=self.phase_variation))

        return x

    def get_random(self):
        '''
        Returns a random superposition from the dataset.
        '''
        return self.__getitem__(np.random.randint(len(self)))




class Dataset():
    '''
    Class to load/generate dataset for Machine Learning
    '''

    def __init__(self, max_order, image_params = [0, 0, (0, 1)], info: bool = True, stage:int = 1, batch_size:int = 10, pixels=128):
        '''
        Initialise the class with the required complexity.

        'max_order': Max order of Guassian modes in superpositions (x > 0).
        'image_params': sets image noise params [noise_variance, (exposure_minimum, max_pixel_shift, exposure_maximum)]
        '''
        self.max_order = max_order
        self.noise_variation = image_params[0]
        self.exposure = image_params[1]
        self.stage = stage
        self.info = info
        self.batch_size = batch_size
        self.pixels = pixels

        if self.info: print("\n_____| Generating Data |_____\n")
        if self.info: print("Max order of mode: " + str(self.max_order) + "\nVariation in saturation noise: " + str(self.noise_variation) + "\nVariation in saturation exposure: " + str(self.exposure)  + "\n")
        if self.info: print("Generating Gaussian modes...")
        
        self.hermite_modes = [Hermite(l=i, m=j, pixels=self.pixels) for i in range(max_order) for j in range(max_order)]
        self.laguerre_modes = [Laguerre(p=i, m=j, pixels=self.pixels) for i in range(self.max_order // 2) for j in range(self.max_order // 2)]
        self.gauss_modes = self.hermite_modes + self.laguerre_modes


    def load_data(self):
        '''
        Generates a single batch of data
        Returns [input_data, output_data]
        Where:
        input_data[0] is the 0th Image np.array
        output_data[0] is the 0th Neural Net ouput array containing mode amps and cos(phase)s
        '''
        input_data = np.zeros((self.batch_size, self.pixels, self.pixels))
        output_data = np.zeros((self.batch_size, np.size(self.hermite_modes)*2))
        for b in range(self.batch_size):
            mode_components = [randomise_amp_and_phase(mode) for mode in self.gauss_modes]
            s = Superposition(*mode_components, pixels=self.pixels)
            input_data[b, :, :] = add_exposure(add_noise(s.superpose(), self.noise_variation), self.exposure) # Generate noise image

            output_data[b, :] = np.array([s.contains(j).amplitude for j in self.hermite_modes] + [np.cos(s.contains(j).phase) for j in self.hermite_modes])
        

        input_data = np.array(input_data)[..., np.newaxis]
        output_data = np.array(output_data) # Convert to arrays of correct shape
        return input_data, output_data




##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################

#### Functions affecting Superpositions, Hermites, Laguerres

def randomise_amplitudes(mode_list, variance):
    amplitudes = np.zeros((len(mode_list)))
    new_modes = []
    for i in range(len(mode_list)):
        amplitudes[i] = abs(round(np.random.normal(scale=variance), 2) + 1) # Make randomised amplitude based on normal distribution
    amplitudes /= np.linalg.norm(amplitudes) # Normalise amplitudes
    for i, mode in enumerate(mode_list):
        new_modes.append(mode * amplitudes[i])
    return new_modes

def randomise_amp_and_phase(mode):
    '''
    Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
    Returns new mode with randomised amp and phase.
    '''
    x = mode.copy()

    x *= random.random() # Change amp by random amount
    x.add_phase(random.random()* 2 * np.pi) # Add random amount of phase

    return x



##### Functions affecting image matrices


def add_noise(image, noise_variance: float = 0.0):
    '''
    Adds random noise to a copy of the image according to a normal distribution of variance 'noise_variance'.
    Noise Variance defined as a %age of maximal intensity
    '''

    actual_variance = np.abs(np.random.normal(0, noise_variance)) 
    # Noise Variance parameter gives maximum noise level for whole dataset
    # Actual Noise is the gaussian noise variance used for a specific add_noise call

    max_val = np.max(image)
    return np.random.normal(loc=image, scale=actual_variance*max_val) # Variance then scaled as fraction of brightest intensity


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


def shift_image(image, max_pixel_shift):
    '''
    Will translate target image in both x and y by integer pixels by random numbers in the range (-max_pixel_shift, max_pixel_shift)
    '''
    copy = np.zeros_like(image)
    x_shift = random.randint(-max_pixel_shift, max_pixel_shift)
    y_shift = random.randint(-max_pixel_shift, max_pixel_shift)
    shape = np.shape(image)
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_coords = [i + x_shift, j + y_shift]
            if new_coords[0] in range(shape[0]) and new_coords[1] in range(shape[1]): # New coordinates still within image bounds
                copy[i, j] = image[new_coords[0], new_coords[1]]
            else:
                copy[i, j] = 0
    return copy



#### Misc functions


def grouper(iterable, n, fillvalue=None):
    '''
    Itertools grouper recipe
    '''
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)




if __name__ == "__main__": 
    s = Superposition(Hermite(0, 0))
    im = s.superpose()

    new_im = shift_image(im, 10)
    print(new_im)
    plt.imshow(new_im)
    plt.show()


