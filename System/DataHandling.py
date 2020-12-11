import numpy as np
from scipy import special
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from itertools import combinations, chain
from multiprocessing import Pool, cpu_count
import time
import random
import sys
import os

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

        if info: print("\n_____| Generating Data |_____\n")
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
        randomised_item = [self.randomise_amp_and_phase(i) for i in item]
        return Superposition(*randomised_item)

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

        p = Pool(cpu_count())
        n = len(self) // (cpu_count() - 1)
        jobs = [self[i:i + n] for i in range(0, len(self), n)]
        threads = p.map(self.superpose_process, tqdm(jobs, desc))
        return np.array([item for item in chain(*threads)])[..., np.newaxis]

    def superpose_process(self, data):
        '''
        Process for superposing elements of the dataset across multiple threrads in the CPU.
        '''
        return [item.superpose(self.noise_variation, self.exposure) for item in data]

    def get_outputs(self):
        '''
        Get all possible Gaussian modes that could comprise a superposition.
        '''
        # return np.array(self.repeats * [[int(str(j)[:-1] in str(i)) * 0.5 for j in self.gauss_modes] for i in self.combs])
        return np.array([[i.contains(j) for j in self.hermite_modes] for i in self]) # + [(i.contains(j).phase / (2 * np.pi)) % 1 for j in self.hermite_modes]

    def get_classes(self):
        '''
        Get the num_classes result required for model creation.
        '''
        return np.array(self.hermite_modes, dtype=object) # * self.repeats

    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()

        x *= np.abs(round(np.random.normal(scale=self.amplitude_variation), 2) + 1)
        x.add_phase(np.abs(round(np.random.normal(scale=self.phase_variation), 2)))

        return x

    def get_random(self):
        '''
        Returns a random superposition from the dataset.
        '''
        return self.__getitem__(np.random.randint(len(self)))

class Dataset():
    '''
    '''
    def __init__(self):
        pass