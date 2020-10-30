import numpy as np
import random as rand
import matplotlib.pyplot as plt
import os
import csv

def generateData(n):
    x = np.array([rand.uniform(0, 2*np.pi) for i in range(n)]) #generate imput test data on a random uniform pdf
    sin = np.sin(x)
    return x, sin

def saveData(x, sin, fname, path=None):
    data = np.zeros((len(x), 2))
    data[:, 0] = x
    data[:, 1] = sin

    if(path==None):
        path = os.getcwd() + r'\Simulation\Sin Replication Test\Data'

    file = path + os.sep + fname

    with open(file, 'w') as f:
        writer = csv.writer(f, )
        for i, j in data[:, :]:
            writer.writerow([i, j])

def readData(fname, path=None):
    if(path==None):
        path = os.getcwd() + r'\Simulation\Sin Replication Test\Data'
    file = path + os.sep + fname
    with open(file, 'r') as f:
        data = np.genfromtxt(f, dtype=float, delimiter=',')
    return data[:, 0], data[:, 1]

x, sin = readData('testdata.csv')
plt.scatter(x, sin)
plt.show()