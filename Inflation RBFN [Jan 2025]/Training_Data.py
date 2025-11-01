# MARK: Opening the CSV and cleaning the Data

from numpy import *


def readData(fileName, signMask=None): # This returns a tuple with keys, then values
    file = open(fileName, "r")
    lines = file.readlines()
    keys = []
    values = []

    for line in lines:
        if "\n" in line:
            line = line[:-1]
        if "\ufeff" in line:
            line = line[1:]
            
        data = line.split(",")
        keys.append([float(string) for string in data[:-1]])
        values.append(float(data[-1]))

    if signMask == None:
        signMask = array([1] * len(keys[0]))

    procKeys = []
    for key in keys:
        procKeys.append([a * b for a, b in zip(signMask, key)])

    return (procKeys, values)
            
