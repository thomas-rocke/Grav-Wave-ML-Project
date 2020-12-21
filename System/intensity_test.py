import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from DataHandling import Dataset, quantize_image, add_noise
from Utils import *
import random
from Gaussian_Beam import Superposition, Hermite
import time
import os

if __name__ == "__main__":
    x = Superposition(Hermite(0, 0), Hermite(2, 1))
    img = add_noise(x.superpose(), 0.01)

    img_8 = quantize_image(img, 8)
    img_16 = quantize_image(img, 16)

    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(img)
    ax[0].set_title('No Quantization')

    ax[1].imshow(img_8)
    ax[1].set_title('8 Bit Quantization')

    ax[2].imshow(img_16)
    ax[2].set_title('16 Bit Quantization')

    plt.show()