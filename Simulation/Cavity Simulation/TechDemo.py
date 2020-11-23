# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:55:01 2020

@author: Tom
"""

import numpy as np
from Gaussian_Beam import Hermite, Laguerre, Superposition, combinations
from ML_Matrix import *


#%%
x = Hermite(0, 3)
x.show()
#%%
y = Laguerre(2, 2)
y.show()
#%%
sup = Superposition([Hermite(0, 0)*0.66, Hermite(3, 2)*0.75, Laguerre(4, 4)*0.43], max_order=5)
print(sup.mode_matrix)
sup.show()

m = Model(3, 2)
m.train()