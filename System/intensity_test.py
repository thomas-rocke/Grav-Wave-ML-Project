import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from DataHandling import Generate_Data, Dataset
from Utils import *
import random
from Gaussian_Beam import Superposition, Hermite
import time
import os

if __name__ == "__main__":
    x = Generate_Data(5, 3)