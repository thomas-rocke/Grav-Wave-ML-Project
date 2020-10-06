import matplotlib.image as im
import scipy.fftpack as fft
import numpy as np
import os
import matplotlib.pyplot as plt

def getImageData(fname, dir):
    #Opens the .png image file with name fname in directory dir and reads the data
    #Returns a np array of shape (x pixels, y pixels, 3)
    #slice [x, y, col] gives the luminosity value at coords (x, y) of colour col
    #col = 0 is red, 1 is green, 2 is blue
    img_data = im.imread(dir + os.sep + fname).astype(float)
    return img_data

def getImageFourier (fname, dir):
    #Takes the fourier transform of each colour of the image indepentantly
    img_data = getImageData(fname, dir)
    img_fourier = np.zeros_like(img_data)
    for i in range(3):
        img_fourier[:, :, i] = np.abs(fft.fft2(img_data[:, :, i]))
    return img_fourier






