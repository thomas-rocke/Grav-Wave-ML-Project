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

def str_to_bin(str):
    return format(ord(str), '008b')

if __name__ == "__main__":
    string = 'Hello World'

    fig, ax = plt.subplots(nrows=2, sharex=False)
    labels = [s for s in string]
    pred_labels = [0]*len(labels)
    subsections=8
    width = 0.35
    x = np.arange(len(labels))
    ax[0].set_xticks(x)
    ax[1].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("Input signal")
    ax[1].set_ylabel("Predicted signal from beam image")
    ax[0].set_xlabel("Input Character (inputted as ASCII)")
    ax[1].set_xlabel("Predicted Character")
    ax[0].set_title("Using a model to interpret communicate signals")
    offsets = np.linspace(-width/2, width/2, subsections)

    model = ML(BasicGenerator(3, 3, 0.5, 0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
    model.load()
    modes = model.data_generator.hermite_modes[:subsections]
    for i in x:
        true_vals = [int(j) for j in str(str_to_bin(string[i]))]
        for j in range(subsections): modes[j].amplitude = true_vals[j]
        s = Superposition(*modes)
        vals = [m.amplitude for m in s]
        vals /= np.max(vals)
        ax[0].bar(i + offsets, vals, width/subsections)

        img = s.superpose()
        prediction = model.predict(img)
        pred_amps = [0]*subsections
        for j, mode in enumerate(modes):
            for p in prediction:
                if str(mode) == str(p):
                    pred_amps[j] = p.amplitude
        pred_amps /= np.max(pred_amps)
        ax[1].bar(i + offsets, pred_amps, width/subsections)

        trigger = [int(j >= 0.5) for j in pred_amps]
        pred_letter_bin = int(''.join([str(t) for t in trigger]), 2)
        pred_letter = chr(pred_letter_bin)
        pred_labels[i] = pred_letter
    ax[1].set_xticklabels(pred_labels)
    plt.show()