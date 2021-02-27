import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from DataHandling import Dataset, BasicGenerator
from ML_Identification import ML
from Utils import *
import random
from Gaussian_Beam import Superposition, Hermite
import time
import os

if __name__ == "__main__":
    words = {"1":[0, 1, 0, 0, 1, 0, 0, 0],
             "2":[0, 1, 1, 0, 0, 1, 0, 1],
             "3":[0, 1, 1, 0, 1, 1, 0, 0],
             "4":[0, 1, 1, 0, 1, 1, 0, 0],
             "5":[0, 1, 1, 0, 1, 1, 0, 0],
             "6":[0, 0, 1, 0, 0, 0, 0, 0],
             "7":[0, 1, 0, 1, 0, 1, 1, 1],
             "8":[0, 1, 1, 0, 1, 1, 0, 0],
             "9":[0, 1, 1, 1, 0, 0, 1, 0],
             "10":[0, 1, 1, 0, 1, 1, 0, 0],
             "11":[0, 1, 1, 0, 0, 1, 0, 0],
             "12":[0, 0, 1, 0, 0, 0, 0, 1]}

    letters = ["H", "e", "l", "l", "o", "(Space)", "W", "o", "r", "l", "d", "."]

    fig, ax = plt.subplots(nrows=2, sharex=True)
    labels = list(words.keys())
    subsections=len(words[labels[0]])
    width = 0.35
    x = np.arange(len(labels))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(letters)
    ax[0].set_ylabel("Input signal")
    ax[1].set_ylabel("Predicted signal from beam image")
    ax[0].set_xlabel("Input Character (inputted as ASCII)")
    ax[0].set_title("Using a model to interpret communicate signals")
    offsets = np.linspace(-width/2, width/2, subsections)

    model = ML(BasicGenerator(3, 3, 0.5, 0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
    model.load()
    modes = model.data_generator.hermite_modes[:8]
    for i in x:
        true_vals = words[labels[i]]
        for j in range(8): modes[j].amplitude = true_vals[j]
        s = Superposition(*modes)
        vals = [m.amplitude for m in s]
        vals /= np.max(vals)
        ax[0].bar(i + offsets, vals, width/subsections)

        img = s.superpose()
        prediction = model.predict(img)
        pred_amps = [0]*8
        for j, mode in enumerate(modes):
            for p in prediction:
                if str(mode) == str(p):
                    pred_amps[j] = p.amplitude
        pred_amps /= np.max(pred_amps)
        ax[1].bar(i + offsets, pred_amps, width/subsections)
    plt.show()