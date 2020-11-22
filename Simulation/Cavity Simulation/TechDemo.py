# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:55:01 2020

@author: Tom
"""

import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Beam import *

def add_noise(image, noise_variance=0.001):
        '''
        Adds random noise to a copy of the image according to a normal distribution of variance noise_variance
        '''
        img_copy = image.copy()
        max_val = np.max(img_copy)
        norm = lambda i: np.min([max_val, np.random.normal(loc=i, scale=noise_variance)])
        f = np.vectorize(norm)
        img_copy = f(img_copy)
        return img_copy


x = Laguerre(0, 0)
img = x.superpose()
print( np.max(img))
noise = add_noise(img, 0.001)

plt.imshow(noise, cmap='Greys_r')
plt.show()

plt.imshow(noise - img, cmap='Greys_r')
plt.show()

print(np.max(noise-img))