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
        # return np.array([i.superpose() for i in tqdm(self, desc)])[..., np.newaxis]

        # Below is support for multiprocessing of superpositions, however it is limited by disk speed and can cause memory overflow errors

        # if os.path.exists("Data/" + str(self) + ".txt"): # Data already exists
        #     print(desc + "... ", end='')
        #     data = np.loadtxt("Data/" + str(self) + ".txt").reshape((len(self), self[0][0].pixels, self[0][0].pixels, 1))
        #     print(f"Done! Loaded from file: '{str(self)}'.")
        #     return data

        # else: # Generate and save new data

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




class Dataset():
    '''
    Class to load/generate dataset for Machine Learning
    '''

    def __init__(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0.0, phase_variation: float = 0.0, noise_variation: float = 0.0, exposure: tuple = (0.0, 1.0), repeats: int = 1, stage: int = 1, folder_name: str = "Data", batch_size: int = 128, info: bool = True):
        '''
        Initialise the class with the required complexity.

        'max_order': Max order of Guassian modes in superpositions (x > 0).
        'number_of_modes': How many modes you want to superimpose together (x > 0).
        '''
        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.phase_variation = phase_variation
        self.noise_variation = noise_variation
        self.exposure = exposure
        self.repeats = repeats
        self.stage = stage
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.info = info

        self.dir = os.getcwd() + os.sep + folder_name + os.sep + str(self) # Saves path to files
        os.makedirs(self.dir, exist_ok=True)
        self.files = os.listdir(self.dir)

        self.sup_fname = f"Stage_{self.stage}_sup.txt"
        self.dat_fname = f"Stage_{self.stage}_dat.txt"

        self.check_data() # Check if files exist, create them if empty

        dat_file = open(self.dir + os.sep + self.dat_fname, 'r')
        sup_file = open(self.dir + os.sep + self.sup_fname, 'r') # Open files to read

        dat_lines = [str(line) for line in dat_file]
        sup_lines = [str(line) for line in sup_file]

        dat_file.close()
        sup_file.close()

        self.pixels = int(sup_lines[0]) # Recover pixels saved from file

        mapIndexPosition = list(zip(dat_lines, sup_lines)) # Create a list of the zipped data lists (so indeces map together)
        random.shuffle(mapIndexPosition) # Shuffle list randomly
        dat_lines, sup_lines = zip(*mapIndexPosition) # Unzip the lists

        self.dat_batches = grouper(dat_lines, self.batch_size)
        self.sup_batches = grouper(sup_lines[1:], self.batch_size)

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + f"({self.max_order}, {self.number_of_modes}, {self.amplitude_variation}, {self.phase_variation}, {self.noise_variation}, {self.exposure}, {self.repeats}, {self.stage}, {self.folder_name}, {self.batch_size}, {self.info})"

    def check_data(self):
        '''
        Checks if data files exist, and creates them if not
        '''
        dat_exists, sup_exists = self.dat_fname in self.files, self.sup_fname in self.files #check if data files exist

        if not dat_exists or not sup_exists:
            combs = self.make_data()
            self.save_data(combs, dat_exists, sup_exists) #save data if files not present

    def make_data(self):
        '''
        Generates initial dataset of all potential superposition combinations
        '''
        if self.info: print("\n_____| Generating Data |_____\n")
        if self.info: print("Max order of mode: " + str(self.max_order) + "\nNumber of modes in superposition: " + str(self.number_of_modes) + "\nVariation in mode amplitude: " + str(self.amplitude_variation) + "\nVariation in mode phase: "
                        + str(self.phase_variation) + "\nVariation in saturation noise: " + str(self.noise_variation) + "\nVariation in saturation exposure: " + str(self.exposure) + "\nRepeats of combinations: " + str(self.repeats) + "\n")
        if self.info: print("Generating Gaussian modes...")

        self.hermite_modes = [Hermite(l=i, m=j) for i in range(self.max_order) for j in range(self.max_order)]
        self.laguerre_modes = [Laguerre(p=i, m=j) for i in range(self.max_order // 2) for j in range(self.max_order // 2)]
        self.gauss_modes = self.hermite_modes + self.laguerre_modes

        if self.info: print("Done! Found " + str(len(self.hermite_modes)) + " hermite modes and " + str(len(self.laguerre_modes)) + " laguerre modes giving a total of " + str(len(self.gauss_modes)) + " gaussian modes.\n\nGenerating superpositions...")

        combs = [list(combinations(self.gauss_modes, i)) for i in range(1, self.number_of_modes + 1)]
        combs = [i[j] for i in combs for j in range(len(i))]

        return combs

    def save_data(self, combs, dat_exists, sup_exists):
        '''
        Creates and saves down dataset
        '''
        dat_file = open(self.dir + os.sep + self.dat_fname, 'w' if dat_exists else 'x')
        sup_file = open(self.dir + os.sep + self.sup_fname, 'w' if sup_exists else 'x')

        p = Pool(cpu_count())
        print(combs[0][0])
        sup_file.write(str(combs[0][0].pixels) + '\n')
        print("Saving Data:")

        for j in tqdm(range(self.repeats), desc='Stepping through Repeats'):
            batches = grouper(combs, cpu_count())

            for batch in batches:
                sups, imgs = zip(*p.map(self.generate_process, batch))

                sups = [s for s in sups if s is not None]
                imgs = [i for i in imgs if i is not None] # Remove Nonetype elements from lists

                for i in range(len(imgs)):
                    if imgs[i] is not None and sups[i] is not None:
                        vec = imgs[i].flatten() #Flatten image data to vector
                        vec_string = ', '.join(str(v) for v in vec) + '\n'
                        dat_file.write(vec_string)
                        sup_file.write(repr(sups[i]) + '\n')

        dat_file.close()
        sup_file.close()

    def generate_process(self, item):
        '''
        Process for generating superposition objects across multiple threads in the CPU.
        '''
        if item is None: return None, None

        randomised_item = [self.randomise_amp_and_phase(i) for i in item]
        s = Superposition(*randomised_item)

        return s, s.superpose()

    def load_data(self, batch_number):
        '''
        Load a batch of data from files
        '''
        dat_batch = next(islice(self.dat_batches, batch_number, None), None)
        sup_batch = next(islice(self.sup_batches, batch_number, None), None) # Fetch the [batch_number] batch of file lines

        p = Pool(cpu_count())

        dats = p.map(self.load_dat_process, dat_batch)
        sups = p.map(self.load_sup_process, sup_batch)

        dats = [d for d in dats if d is not None]
        sups = [s for s in sups if s is not None] # Remove any Nonetype elements

        return sups, dats

    def load_sup_process(self, sup):
        '''
        Process for loading superposition objects across multiple threads in the CPU.
        '''
        return None if sup is None else eval(sup)

        # if sup is None:
        #     return None
        # else:
        #     return eval(sup)

    def load_dat_process(self, dat):
        '''
        Load image data across multiple threads
        '''
        if dat is None: return None

        new_dat = dat.split(', ') # Split up data line at each comma delimiter
        new_dat = np.array([float(d) for d in new_dat]) # Cast elements back to floats

        return new_dat.reshape((self.pixels, self.pixels))
    
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
    Itertools grouper recipe.
    '''
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)




if __name__ == "__main__": 
    dir = 'System' + os.sep + 'TestData'
    data_obj = Dataset(3, 5, 0.6, batch_size=10, repeats=3, foldername=dir)

    start_time = time.time()
    sups, imgs = data_obj.load_data(7)
    end_time = time.time()

    print(end_time - start_time)
    sups[1].plot()
    plt.imshow(imgs[0])
    plt.show()