import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np
import os
import random
import sys
sys.path.insert(1, '../Simulation/Cavity Simulation') # Move to directory containing simulation files
from Gaussian_Beam import *
from ML_Identification import *

def readImage(fname, dir):
    #Opens the .png image file with name fname in directory dir and reads the data
    #Data in the form of an np array of shape (x pixels, y pixels, 3)
    #slice [x, y, col] gives the luminosity value at coords (x, y) of colour col
    #col = 0 is red, 1 is green, 2 is blue
    #Function returns dict with key = filename and value = image data
    img_data = im.imread(dir + os.sep + fname).astype(float)
    d = {fname:img_data}
    return d

def readCSV(fname, dir, headers=None, sep=',', skiprows=0):
    #Opens CSV file and reads contents
    #Places data into a list of dicts, assuming that each column is a different input, and each row is a different input
    #If headers=None, assumes that the first row of the file contains the headers, else the function will use the supplied headers
    #The sep param is included as an override in the case of a file separated by other characters
    #Returns list of dicts of length rows, with each dict having columns keys.

    file_path = dir + os.sep + fname
    raw_data = np.genfromtxt(file_path, delimiter=sep, skiprows=skiprows)

    if headers==None:
        headers = raw_data[0, :]
        dict_list = [0]*(len(raw_data)-1) #Creates list of correct length
        for i in range(len(raw_data) - 1):
            d = dict()
            for j in range(len(headers)):
                d[headers[j]] = raw_data[i+1, j]
            dict_list[i] = d


def get_model_error(model:ML, data_object:Generate_Data, test_number:int=10, sup:Superposition=None):
    '''
    Tests the accuracy of the model from data contained within data_object
    
    Assumes a gaussian resultant error through the Central Limit Theorem
    
    Tests on test_percent of the data given in the data_object
    '''
    if sup == None:
        test_data = random.sample(data_object.combs, test_number) # Select superpositions randomly from combs
    else:
        test_data = np.array([sup.copy() for i in range(test_number)]) # Make several copies of the target superposition

    test_data = np.array([data_object.randomise_amp_and_phase(i) for i in test_data]) # Randomise all amps and phases
    
    model_predictions = np.array([model.predict(data.superpose()) for data in test_data]) # Predict superpositions through models
    
    
    test_amps = np.array([[mode.amplitude for mode in sup] for sup in test_data])
    model_amps = np.array([[mode.amplitude for mode in sup] for sup in model_predictions])
    
    amp_err = (np.sum([(model_amps[i] - test_amps[i])**2 for i in range(len(test_amps))])/(len(test_amps) - 1))**0.5 # Predicts amplitude error assuming error is constant throughout multivariate space
    
    test_phases = np.array([[mode.phase for mode in sup] for sup in test_data])
    model_phases = np.array([[mode.phase for mode in sup] for sup in model_predictions])
    
    phase_err = (np.sum([(model_phases[i] - test_phases[i])**2 for i in range(len(test_phases))])/(len(test_phases) - 1))**0.5 # Predicts phase error assuming error is constant throughout multivariate space
    
    test_imgs = np.array([sup.superpose() for sup in test_data])
    model_imgs = np.array([sup.superpose() for sup in model_predictions])
    
    img_err = (np.sum([(model_imgs[i] - test_imgs[i])**2 for i in range(len(test_imgs))])/(len(test_imgs) - 1))**0.5 # Predicts img error assuming error is constant throughout multivariate space
    

    return amp_err, phase_err, img_err

max_order = 3
number_of_modes = 5
amplitude_variation = 0.2
phase_variation = 0.0
noise_variation = 0.0
exposure = (0.0, 1.0)
repeats = 50

model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats)
sys.path.insert(1, '../System') # Move to directory containing simulation files
model.load()

data = Generate_Data(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)

errs = get_model_error(model, data, 0.5)

print(errs[0])
print(errs[1])
plt.imshow(errs[2])
plt.show()