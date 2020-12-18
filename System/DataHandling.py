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



class Dataset():
    '''
    Class to load/generate dataset for Machine Learning
    '''

    def __init__(self, max_order, image_params = [0, 0, (0, 1)], info: bool = True, batch_size:int = 10, pixels=128):
        '''
        Initialise the class with the required complexity.

        'max_order': Max order of Guassian modes in superpositions (x > 0).
        'image_params': sets image noise params [noise_variance, max_pixel_shift, (exposure_minimum, exposure_maximum)]
        '''
        self.max_order = max_order
        self.image_params = image_params
        self.info = info
        self.batch_size = batch_size
        self.pixels = pixels

        if self.info: print("\n_____| Generating Data |_____\n")
        if self.info: print("Max order of mode: " + str(self.max_order) + "\nVariation in noise: " + str(self.image_params[0]) + "\nVariation in exposure: " + str(self.image_params[2]) + "\nMax coordinate shift: " + str(self.image_params[1])  + "\n")
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
            input_data[b, :, :] = Image_Processing(s.superpose(), self.image_params) # Generate noise image

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

def Image_Processing(image, image_params):
    '''
    Performs all image processing on target image
    'image_params': sets image noise params [noise_variance, max_pixel_shift, (exposure_minimum, exposure_maximum)]
    '''
    noise_variance = image_params[0]
    max_pixel_shift = image_params[1]
    exposure_lims = image_params[2]

    shifted_image = shift_image(image, max_pixel_shift) # Shift the image in x and y coords
    noisy_image = add_noise(shifted_image, noise_variance) # Add Gaussian Noise to the image
    exposed_image = add_exposure(noisy_image, exposure_lims) # Add exposure

    return exposed_image



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


