import matplotlib.image as im
import numpy as np
import os

def readImage(fname, dir):
    #Opens the .png image file with name fname in directory dir and reads the data
    #Data in the form of an np array of shape (x pixels, y pixels, 3)
    #slice [x, y, col] gives the luminosity value at coords (x, y) of colour col
    #col = 0 is red, 1 is green, 2 is blue
    #Function returns dict with key = filename and value = image data
    img_data = im.imread(dir + os.sep + fname).astype(float)
    d = {fname:img_data}
    return d

def readCSV(fname, dir, headers=None, sep=',', skiprows=0):
    #Opens CSV file and reads contents
    #Places data into a list of dicts, assuming that each column is a different input, and each row is a different input
    #If headers=None, assumes that the first row of the file contains the headers, else the function will use the supplied headers
    #The sep param is included as an override in the case of a file separated by other characters
    #Returns list of dicts of length rows, with each dict having columns keys.

    file_path = dir + os.sep + fname
    raw_data = np.genfromtxt(file_path, delimiter=sep, skiprows=skiprows)

    if headers==None:
        headers = raw_data[0, :]
        dict_list = [0]*(len(raw_data)-1) #Creates list of correct length
        for i in range(len(raw_data) - 1):
            d = dict()
            for j in range(len(headers)):
                d[headers[j]] = raw_data[i+1, j]
            dict_list[i] = d