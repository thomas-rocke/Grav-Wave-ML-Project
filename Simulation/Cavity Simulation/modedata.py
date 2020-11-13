import os
from Gaussian_Beam import Superposition
from Gaussian_Beam import Gaussian_Mode
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

def fact(x):
    res = 1
    if x != 0:
        for i in range(x):
            res *= i + 1
    return res

def choose(n, r):
    return fact(n)/(fact(r)*fact(n-r))


def makeLaguerre(p, m):
    x=[]
    for q in range(p + 1):
        for s in range(int((m)/2) + 1):
            frac = ((fact(2*(q-s) + m)*fact(2*(p-q+s)))/(2**(2*p + m - 1) * fact(p) * fact(p+m) * (1 + choose(0, m))))**0.5
            y = Gaussian_Mode(2*(q - s) + m, 2*(p - q + s))
            y.amplitude = (-1)**(s+p) * choose(p, q) * choose(m, 2*s) * frac
            x.append(y)

    return Superposition(x)


y = makeLaguerre(3, 4)
y.show()