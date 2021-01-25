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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from Utils import meanError
from Gaussian_Beam import Hermite, Superposition, Laguerre
import keras
from ImageProcessing import ModeProcessor, camera_presets




##################################################
##########                              ##########
##########           CLASSES            ##########
##########                              ##########
##################################################


class GenerateData(list):
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
        self.info = info

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
        # self.extend(self.combs)

        p = Pool(cpu_count())
        self.extend(p.map(self.generate_process, self.combs * repeats))

        if info: print("Done! Found " + str(len(self)) + " combinations.\n")

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + f"({self.max_order}, {self.number_of_modes}, {self.amplitude_variation}, {self.phase_variation}, {self.noise_variation}, {self.exposure}, {self.repeats}, {self.info})"

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
        # if os.path.exists("Data/" + str(self) + ".txt"): # Data already exists
        #     print(desc + "... ", end='')
        #     data = np.loadtxt("Data/" + str(self) + ".txt").reshape((len(self), self[0][0].resolution, self[0][0].resolution, 1))
        #     print(f"Done! Loaded from file: '{str(self)}'.")
        #     return data

        # else: # Generate and save new data

        # Below is support for multiprocessing of superpositions, however it is limited by disk speed and can cause memory overflow errors

        if len(self) < cpu_count(): return np.array([i.superpose() for i in tqdm(self, desc)])[..., np.newaxis]

        p = Pool(cpu_count())
        jobs = np.reshape(np.array(self, dtype=object), [-1, len(self.combs)]) # Split data into repeats and process each in a new thread
        threads = p.map(self.superpose_process, tqdm(jobs, desc))
        return np.array([item for item in chain(*threads)])[..., np.newaxis] # Chain the threads together

        # np.savetxt("Data/" + str(self) + ".txt", data.reshape((len(self), -1)))
        # return data

        # p = Pool(cpu_count())
        # n = len(self) // (cpu_count() - 1)
        # jobs = [self[i:i + n] for i in range(0, len(self), n)]
        # threads = p.map(self.superpose_process, tqdm(jobs, desc))
        # return np.array([item for item in chain(*threads)])[..., np.newaxis]

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




class Dataset(keras.utils.Sequence):
    '''
    Class to load/generate dataset for Machine Learning
    '''


    def __init__(self, mode_processor:ModeProcessor, mode_mask:int = 0,  max_order: int = 3, resolution: int = 128, batch_size: int = 128, batches_per_repeat: int = 100, repeats_per_epoch: int = 100, training_stage: int = 0, info: bool = True):
        '''
        Initialise the class with the required complexity.
        '''
        self.mode_mask = mode_mask
        self.mode_processor = mode_processor
        self.max_order = max_order
        self.resolution = resolution
        self.batch_size = batch_size
        self.steps = batches_per_repeat
        self.repeats = repeats_per_epoch
        self.steps_per_epoch = self.steps * self.repeats
        self.stage = training_stage
        self.info = info

        self.epoch = 1
        self.current_step = 0
        self.current_repeat = -1
        self.seed = self.get_seed(self.stage, self.epoch)

        if self.info: print("\n_____| Dataset |_____\n")
        if self.info: print(f"Max order of mode: {self.max_order}\n")

        self.hermite_modes = [Hermite(l=i, m=j, resolution=self.resolution) for i in range(max_order) for j in range(max_order)]
        self.laguerre_modes = [Laguerre(p=i, m=j, resolution=self.resolution) for i in range(self.max_order // 2) for j in range(self.max_order // 2)]
        self.gauss_modes = self.hermite_modes + self.laguerre_modes

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        #TODO: Fix repr
        #return self.__class__.__name__ + f"({self.max_order}, {self.image_params}, {self.sup_params}, {self.resolution}, {self.batch_size}, {self.steps_per_epoch}, {self.info})"

    def __len__(self):
        '''
        Denotes the number of batches per epoch.
        For us this is the (total number of combinations * repeats) / batch size.
        '''
        return self.steps_per_epoch

    def __getitem__(self, index):
        '''
        Generates a single batch of data for use in Keras model.fit_generator
        Returns [input_data, output_data]
        Where:
        input_data[n, :, :, 0] is the Image np.array for the nth batch element
        output_data[n, :] is the nth Neural Net ouput array containing mode amps and cos(phase)s

        The index param is unused, and is included for compatability with keras model.fit_generator
        '''
        if self.current_step % self.steps == 0: # Processed a full repeat
            np.random.seed(self.seed) # Reset randomness between repeats
            self.current_repeat += 1

        input_data = np.zeros((self.batch_size, self.resolution, self.resolution))
        output_data = np.zeros((self.batch_size, np.size(self.hermite_modes) * 2))

        for b in range(self.batch_size):
            s = Superposition(*[randomise_amp_and_phase(mode) for mode in self.gauss_modes])
            if self.mode_mask:
                for mode in s[self.mode_mask:]: # Filter out modes above self.mode_mask
                    mode.amplitude = 0
                    mode.phase = 0
            input_data[b, :, :] = self.mode_processor.getImage(s.superpose()) # Generate noise image
            output_data[b, :] = np.array([s.contains(j).amplitude for j in self.hermite_modes] + [np.cos(s.contains(j).phase) for j in self.hermite_modes])

        input_data = np.array(input_data)[..., np.newaxis]
        output_data = np.array(output_data) # Convert to arrays of correct shape

        self.current_step += 1
        return input_data, output_data

    def load_batch(self):
        '''
        Generates a single batch of data for use in non-keras applications
        Returns [input_data, output_data]
        Where:
        input_data[n, :, :] is the Image np.array for the nth batch element
        output_data[n] is the nth Superposition object
        '''
        input_data = np.zeros((self.batch_size, self.resolution, self.resolution))

        p = Pool(cpu_count())
        inp, output_data = zip(*p.map(self.batch_load_process, range(self.batch_size)))
        p.close()
        p.join()
        input_data = np.array(inp) 

        return input_data, output_data

    def batch_load_process(self, n):
        s = Superposition(*[randomise_amp_and_phase(mode) for mode in self.gauss_modes])
        if self.mode_mask:
                for mode in s[self.mode_mask:]: # Filter out modes above self.mode_mask
                    mode.amplitude = 0
                    mode.phase = 0
        input_data = self.mode_processor.getImage(s.superpose()) # Generate noise image
        output_data = s

        return input_data, output_data
    
    def on_epoch_end(self):
        '''
        Defines steps taken at the end of each epoch - changing the seed for np.random and iterating the self.epoch
        '''
        self.epoch += 1
        self.seed = self.get_seed(self.stage, self.epoch)
    
    def get_seed(self, stage, epoch):
        '''
        Function which gets a unique seed per epoch, per training stage.
        The function n**2 + n + 41 produces a unique prime number for 0 <= n < 40
        p**m will always be unique for a unique combination of prime p and non-zero integer m
        '''
        base = stage**2 + stage + 41
        seed = base ** epoch
        return seed



##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################

#### Functions affecting Superpositions, Hermites, Laguerres

def superpose_effects(modes, sup_params):
    '''
    Permorms all randomisation processes on a list of modes to turn them into a superposition for ML
    'sup_params': sets all params affecting superpositions [w_0_variance]
    '''

    ## Processes before Superposition
    randomised_modes = [randomise_amp_and_phase(m) for m in modes]
    w_0_variance = sup_params[0]
    varied_w_0_modes = vary_w_0(randomised_modes, w_0_variance)
    s = Superposition(*varied_w_0_modes)
    ## Processes after superposition

    return s

def randomise_amp_and_phase(mode):
    '''
    Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
    Returns new mode with randomised amp and phase.
    '''
    x = mode.copy()

    x *= random.random() # Change amp by random amount
    x.add_phase(random.random() * 2 * np.pi) # Add random amount of phase

    return x

def vary_w_0(modes, w_0_variance):
    '''
    Varies w_0 param for all modes within a superposition
    '''
    new_w_0 = np.random.normal(modes[0].w_0, w_0_variance)
    new_modes = [mode.copy() for mode in modes]
    for m in new_modes:
        m.w_0 = new_w_0
    return new_modes


##### Functions affecting image matrices

def image_processing(image, image_params):
    '''
    Performs all image processing on target image
    'image_params': sets image noise params [noise_variance, max_pixel_shift, (exposure_minimum, exposure_maximum), image_bits]
    '''
    noise_variance = image_params[0]
    max_pixel_shift = image_params[1]
    exposure_lims = image_params[2]
    bits = image_params[3]

    shifted_image = shift_image(image, max_pixel_shift) # Shift the image in x and y coords
    noisy_image = add_noise(shifted_image, noise_variance) # Add Gaussian Noise to the image
    exposed_image = add_exposure(noisy_image, exposure_lims) # Add exposure

    if bits: # Bits > 0 therefore quantize
        quantized_image = quantize_image(exposed_image, bits)
        return quantized_image
    else:
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
    Will translate target image in both x and y by integer resolution by random numbers in the range (-max_pixel_shift, max_pixel_shift)
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

def quantize_image(image, bits):
    '''
    Quantize the image, so that only 255 evenly spaced values possible
    '''
    max_val = np.max(image)
    vals = 2**bits - 1
    bins = np.linspace(0, max_val, vals, endpoint=1)
    quantized_image = np.digitize(image, bins)
    return quantized_image


#### Misc functions

def grouper(iterable, n, fillvalue=None):
    '''
    Itertools grouper recipe.
    '''
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)




##################################################
##########                              ##########
##########             MAIN             ##########
##########                              ##########
##################################################

if __name__=='__main__':
    processor = ModeProcessor(camera_presets['ideal_camera'])
    x = Dataset(processor, mode_mask=1)
    img = x.load_batch()[0][0]
    plt.imshow(img)
    plt.show()