<<<<<<< HEAD
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




y = Laguerre(2, 1)
y.show()
x = Hermite(0, 0)
#y = Hermite(0, 2)
sup = Superposition([x, y])
sup.show()
y.show()
=======
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




y = Laguerre(2, 1)
#y.show()
x = Hermite(0, 0)
#y = Hermite(0, 2)
sup = Superposition([x, y])
#sup.show()
#y.show()
print(sup.mode_matrix)
>>>>>>> 9cbb5b13a9e3eae7fa8457938f5b61bbf34eafd9
