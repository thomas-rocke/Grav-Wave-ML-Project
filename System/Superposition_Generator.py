import numpy as np
from scipy import special
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from itertools import combinations, chain, zip_longest, islice
from multiprocessing import Pool, cpu_count
import logging
import time
import random
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from Utils import meanError, get_cams, get_strategy
from Gaussian_Beam import Hermite, Superposition, Laguerre, lowest_order_zero_phase, highest_amp_zero_phase
import Logger
import keras
from ImageProcessing import ModeProcessor

LOG = Logger.get_logger(__name__)


class SuperpositionGenerator(keras.utils.Sequence):#, ModeProcessor):
    '''
    SuperpositionGenerator is the combination of BasicGenerator in the old generators, with new JSON and image processing techniques built in
    '''

    def __init__(self, max_order:int=3, batch_size:int=128, repeats:int=64, training_strategy_name:str="default", network_resolution:int=128, camera_resolution:int=128, starting_stage:int=1, info:bool = False):
        '''
        Init the class
        '''

        self.batch_size = batch_size
        self.repeats = repeats
        self.max_order = max_order

        self.training_strategy_name = training_strategy_name
        self.strategy = get_strategy(training_strategy_name)

        self.max_stage = str(list(self.strategy.keys())[-1])
        self.camera_resolution = camera_resolution
        self.info = info
        self.starting_stage = starting_stage
        self.network_resolution = network_resolution
        self.mode_processor = ModeProcessor(target_resolution=(network_resolution, network_resolution))

        self.stage = starting_stage - 1 # Set so that first call of new_stage changes stage to the starting_stage

        if self.info: LOG.info("Data Generator initialised!")
        if self.info: LOG.info(f"Max order of mode: {self.max_order}")

        self.hermite_modes = [Hermite(l=i, m=j, resolution=self.camera_resolution) for i in range(max_order) for j in range(max_order)]
        self.laguerre_modes = [Laguerre(p=i, m=j, resolution=self.camera_resolution) for i in range(self.max_order // 2) for j in range(self.max_order // 2)]
        self.gauss_modes = self.hermite_modes + self.laguerre_modes

        # Setting parameter defaults, in case they are not defined by training strategy
        self.mode_processor.change_camera(get_cams("ideal_camera"))
        self.amplitude_variation = 0
        self.phase_variation = 0
        self.number_of_modes = 1
        self._reset_combs()
    
    def new_stage(self):
        self.stage += 1
        if str(self.stage) in self.strategy: # Stage defined in training strategy
            LOG.info("Changing to stage {}".format(self.stage))
            self.change_stage(**self.strategy[str(self.stage)]) # Change params between stages
            return True # Success in changing to new stage
        else:
            LOG.info("Stage {} not found, aborting training".format(self.stage))
            return False
    
    def change_stage(self, **kwargs):
        if 'camera' in kwargs.keys():
            cam = kwargs['camera']
            cam_props = get_cams(cam)
            if self.info: LOG.info("Changing to use camera '{}' with properties {}".format(cam, cam_props))
            self.mode_processor.change_camera(cam_props)
        else:
            if self.info: LOG.info("No New Camera defined this stage, camera will not be changed")

        
        if 'number_of_modes' in kwargs.keys():
            new_num = kwargs['number_of_modes']
            if self.info: LOG.info("Changing number_of_modes to {}".format(new_num))
            self.number_of_modes = new_num
            self._reset_combs()
        else:
            if self.info: LOG.info("No new number_of_modes defined this stage, number_of_modes will not be changed")
        
        if 'amplitude_variation' in kwargs.keys():
            new_var = kwargs['amplitude_variation']
            if self.info: LOG.info("Changing amplitude_variation to {}".format(new_var))
            self.amplitude_variation = new_var
        else:
            if self.info: LOG.info("No new amplitude_variation defined this stage, amplitude_variation will not be changed")
        
        if 'phase_variation' in kwargs.keys():
            new_pha = kwargs['phase_variation']
            if self.info: LOG.info("Changing phase_variation to {}".format(new_pha))
            self.phase_variation = new_pha
        else:
            if self.info: LOG.info("No new phase_variation defined this stage, phase_variation will not be changed")

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + f"({self.max_order}, {self.batch_size}, {self.repeats}, {self.training_strategy_name}, {self.network_resolution}, {self.camera_resolution}, {self.starting_stage}, {self.info}"

    def copy(self):
        '''
        Copy this data generator.
        '''
        return eval(repr(self))

    def __len__(self):
        '''
        Denotes the number of batches per epoch.
        For us this is the (total number of combinations * repeats) / batch size.
        '''
        return int(len(self.combs) / self.batch_size)

    def _reset_combs(self):
        if self.info: LOG.info("Resetting list of combinations")
        self.combs = [list(combinations(self.gauss_modes, i + 1)) for i in range(self.number_of_modes)]
        self.combs = [i[j] for i in self.combs for j in range(len(i))] * self.repeats
        #np.random.shuffle(self.combs)
    
    def generate_superposition(self, comb):
        '''
        Generates the superposition with randomised amplitudes, phases.
        '''
        return Superposition(*[self.randomise_amp_and_phase(i) for i in comb])

    def get_inputs(self, *sups):
        '''
        Get inputs from list of superpositions.
        '''
        inputs = [0]*len(sups)
        for i, sup in enumerate(sups):
            inputs[i] = self.mode_processor.getImage(sup.superpose())

        return inputs

    def get_classes(self):
        '''
        Get the num_classes result required for model creation.
        '''
        LOG.debug("Getting classes.")

        return np.array(self.hermite_modes * 2, dtype=object)

    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()

        x *= np.abs(np.random.normal(loc=1, scale=self.amplitude_variation))
        x.add_phase(np.random.normal(scale=self.phase_variation))

        return x

    def get_random(self):
        '''
        Returns a random superposition from the dataset.
        '''
        LOG.debug("Getting a random superposition from generator.")

        comb = self.combs[np.random.randint(len(self.combs))]
        sup = self.generate_superposition(comb)

        return sup

    def __getitem__(self, index):
        '''
        Generates and returns one batch of data.
        '''
        # LOG.debug(f"Getting item {index}.")

        combs = [self.combs[i] for i in range(index * self.batch_size, (index + 1) * self.batch_size)] # Take combs in order
        # combs = [self.combs[np.random.randint(len(self.combs))] for i in range(self.batch_size)] # Take random combs from self.combs
        sups = [self.generate_superposition(comb) for comb in combs]

        X = np.array(self.get_inputs(*sups))[..., np.newaxis]
        Y = np.array([[i.contains(j).amplitude for j in self.hermite_modes] + [(i.contains(j).phase + np.pi) / (2 * np.pi) for j in self.hermite_modes] for i in sups])

        return X, Y
    
    def on_epoch_end(self):
        #np.random.shuffle(self.combs)
        pass



if __name__ == "__main__":
    gen = SuperpositionGenerator()
    gen.new_stage()
    val = gen[0]

    print(val[0].shape)
    print(val[1].shape)
    print(len(gen))
