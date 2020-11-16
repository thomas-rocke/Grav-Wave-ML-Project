import os
from Gaussian_Beam import Superposition, Hermite, Laguerre
import Gaussian_Beam as gauss
import numpy as np

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




x = Superposition([Laguerre(5, 1), Hermite(0, 0)])
print(x.mode_matrix)