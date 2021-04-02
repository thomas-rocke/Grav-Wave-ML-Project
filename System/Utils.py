import scipy.fftpack as fft
import numpy as np
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import json
import os
import glob
import Logger
import time

LOG = Logger.get_logger(__name__)

def getImageFourier (img_data):
    #Takes the fourier transform of each colour of the image indepentantly
    img_fourier = np.zeros_like(img_data)
    img_fourier = np.abs(fft.fft2(img_data))
    img_fourier = fft.fftshift(img_fourier)
    return img_fourier

def meanError (data):
    """
    Works out the mean and error of the input array data. 
    Returns duple of mean, err
    """
    try:
        mean = np.sum(data)/len(data)
        err_sum = np.sum([(i - mean)**2 for i in data])
        err = np.sqrt(err_sum/(len(data)*(len(data) - 1)))
    except ZeroDivisionError:
        print("Input data not valid:" + data)
        return 0, 0
    return mean, err

def get_cams(camera_name:str = "ideal_camera"):
    '''
    Open Cameras.txt and pick out camera to use
    '''
    fname = "Cameras.txt" # Name of file containing camera properties
    cwd = os.getcwd()
    filepath = glob.glob("".join([cwd, os.sep, "**", os.sep, fname]), recursive = True)[0] # Find file in all subdirectories of cwd
    cams = json.loads(open(filepath).read()) # Read file and convert to dict

    if camera_name in cams.keys():
        msg = "Loading camera '{}'".format(camera_name)
        LOG.info(msg)
        return cams[camera_name]
    else:
        msg = "Camera '{}' not found, defaulting to 'ideal_camera'".format(camera_name)
        LOG.error(msg)
        print(msg)
        return cams['ideal_camera']


def get_strategy(training_strategy_name:str = "default"):
    '''
    Open Training_Strategies.txt and pick out strategy to use
    '''
    fname = "Training_Strategies.txt" # Name of file containing strategy properties
    cwd = os.getcwd()
    filepath = glob.glob("".join([cwd, os.sep, "**", os.sep, fname]), recursive = True)[0] # Find file in all subdirectories of cwd
    strats = json.loads(open(filepath).read()) # Read file and convert to dict

    if training_strategy_name in strats.keys():
        msg = "Loading strategy '{}'".format(training_strategy_name)
        LOG.info(msg)
        return strats[training_strategy_name]
    else:
        msg = "Strategy '{}' not found, defaulting to 'default'".format(training_strategy_name)
        LOG.error(msg)
        print(msg)
        return strats["default"]
    
if __name__ == '__main__':
    n = np.arange(100000)
    t = time.time()
    res = [x for x in n if x > 10]
    print(res[:20])
    print(time.time() - t)

    t = time.time()
    res = n[n > 10]
    print(res[:20])
    print(time.time() - t)