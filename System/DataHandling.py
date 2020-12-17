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
from Gaussian_Beam import Hermite, Superposition, Laguerre, add_noise, add_exposure


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

    def __init__(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0.0, phase_variation: float = 0.0, noise_variation: float = 0.0, exposure: tuple = (0.0, 1.0), repeats: int = 1, info: bool = True, stage:int = 1, foldername:str = "defaultdata", batch_size:int = 10):
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
        self.dir = os.getcwd() + os.sep + foldername #saves path to file
        self.files = os.listdir(self.dir)
        self.stage = stage
        self.info = info
        self.batch_size = batch_size

        self.sup_fname = "Stage_{}.txt".format(self.stage)

        self.check_data() #Check if file exists, create if empty

        sup_file = open(self.dir + os.sep + self.sup_fname, 'r') #Open file to read

        sup_lines = [str(line) for line in sup_file]

        sup_file.close()

        self.pixels = int(sup_lines[0]) # Recover pixels saved from file

        self.sup_strings = sup_lines[1:] # Save down all possible combs to memory

        self.hermite_modes = [Hermite(l=i, m=j) for i in range(self.max_order) for j in range(self.max_order)]

    

    def check_data(self):
        '''
        Checks if data files exist, and creates them if not
        '''

        file_exists = self.sup_fname in self.files # Check if data file exists
        
        if not file_exists:
            combs = self.make_data()
            self.save_data(combs) # Save data if files not present
        
    def make_data(self):
        '''
        Generates initial dataset of all potential superposition combinations
        '''
        if self.info: print("\n_____| Generating Data |_____\n")
        if self.info: print("Max order of mode: " + str(self.max_order) + "\nNumber of modes in superposition: " + str(self.number_of_modes) + "\nVariation in mode amplitude: " + str(self.amplitude_variation) + "\nVariation in mode phase: "
                        + str(self.phase_variation) + "\nVariation in saturation noise: " + str(self.noise_variation) + "\nVariation in saturation exposure: " + str(self.exposure) + "\nRepeats of combinations: " + str(self.repeats) + "\n")
        if self.info: print("Generating Gaussian modes...")

        self.laguerre_modes = [Laguerre(p=i, m=j) for i in range(self.max_order // 2) for j in range(self.max_order // 2)]
        self.gauss_modes = self.hermite_modes + self.laguerre_modes

        if self.info: print("Done! Found " + str(len(self.hermite_modes)) + " hermite modes and " + str(len(self.laguerre_modes)) + " laguerre modes giving a total of " + str(len(self.gauss_modes)) + " gaussian modes.\n\nGenerating superpositions...")

        combs = [list(combinations(self.gauss_modes, i)) for i in range(1, self.number_of_modes + 1)]
        combs = [i[j] for i in combs for j in range(len(i))]
        return combs

    def save_data(self, combs):
        '''
        Creates and saves down dataset
        '''
        
        sup_file =  open(self.dir + os.sep + self.sup_fname, 'x') # Create and open
        
        p = Pool(cpu_count())
        sup_file.write(str(combs[0][0].pixels) + '\n')
        print("Saving Data:")
        batches = grouper(combs, cpu_count())
        for batch in tqdm(batches, desc='Processing data in batches of {}'.format(cpu_count())):
            sups = [s for s in batch if s is not None] # Remove any Nonetype elements from the grouped batches
            for sup in sups:
                    sup_file.write('; '.join([repr(mode) for mode in sup]) + '\n')
        sup_file.close()

    def load_data(self):
        '''
        Load a batch of data from files.
        Returns [input_data, output_data]
        Where:
        input_data[0] is the 0th Image np.array
        output_data[0] is the 0th Neural Net ouput array containing mode amps and cos(phase)s
        '''

        batch = random.sample(self.sup_strings, self.batch_size) # Take random sample of combs
        
        p = Pool(cpu_count())

        data = zip(*p.map(self.load_process, batch))
        p.close()
        p.join() # Wait for processes to finish
        return data

    def load_process(self, sup_string):
        '''
        Process for loading superposition objects across multiple threads in the CPU.
        '''
        if sup_string is None:
            return None
        else:
            components = sup_string.split("; ")
            mode_components = [self.randomise_amp_and_phase(eval(c)) for c in components]
            s = Superposition(*mode_components)
            input_data = add_exposure(add_noise(s.superpose(), self.noise_variation), self.exposure) # Generate noise image

            output_data = np.array([s.contains(j).amplitude for j in self.hermite_modes] + [np.cos(s.contains(j).phase) for j in self.hermite_modes])
            return input_data, output_data
    
    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()

        x *= np.abs(round(np.random.normal(scale=self.amplitude_variation), 2) + 1)
        x.add_phase(np.abs(round(np.random.normal(scale=self.phase_variation), 2)))

        return x






def grouper(iterable, n, fillvalue=None):
    '''
    Itertools grouper recipe
    '''
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)




if __name__ == "__main__": 
    dir = 'System' + os.sep + 'TestData'
    data_obj = Dataset(3, 5, 0.6, batch_size=1, foldername=dir)
    times = [0]*10
    for i in range(10):
        start_time = time.time()
        data_obj.load_data()
        end_time = time.time()
        time_diff = end_time - start_time
        times[i] = time_diff
    
    mean, err = meanError(times)
    print("{}+-{}".format(mean, err))