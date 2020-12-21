import scipy.fftpack as fft
import numpy as np
from skimage.measure import regionprops
from skimage.filters import threshold_otsu

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

def find_cm(image):
    threshold_value = threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    center_of_mass = properties[0].centroid
    return center_of_mass[1], center_of_mass[0]



