import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from DataHandling import Generate_Data

def radial_intensity(mode):
    Rs = np.arange(0, mode.pixels)
    Thetas = np.linspace(0, 2*np.pi, 100)

    I = []
    for r in Rs:
        I.append(np.sum([mode.superpose()[int(r*np.cos(theta)), int(r*np.sin(theta))]for theta in Thetas]))
    return np.array(I)

if __name__ == "__main__":
    x = Generate_Data(1, 2)
    p = Pool(2)
    dat = p.map(radial_intensity, x)
    r = np.arange(0, x[0].pixels)
    for i, d in enumerate(dat):
        plt.plot(r, d, label=str(x[i]))

    plt.legend()
    plt.show()

