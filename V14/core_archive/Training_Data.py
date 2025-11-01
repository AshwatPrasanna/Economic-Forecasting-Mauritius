# MARK: Opening the CSV and cleaning the Data

from numpy import *


def readData(fileName, signMask=None): # This returns a tuple with keys, then values
    file = open(fileName, "r")
    lines = file.readlines()[1:]
    points = []

    for line in lines:
        if "\n" in line:
            line = line[:-1]
        if "\ufeff" in line:
            line = line[1:]
            
        data = [i for i in line.split(",") if i != '']
        points.append([float(x) for x in data])

    return points
            
