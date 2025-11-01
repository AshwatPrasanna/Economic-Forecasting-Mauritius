from __future__ import division
import random
import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

from Training_Data import *


data = readData("Data.csv")
keys = data[0]
values = data[1]



flist = [0.804, 0.290, 1.423, 1.103, 0.848, -0.763, -0.094, 0.290, 1.006, -0.328, -0.106, 0.751, 0.356, 3.720, 2.054, 1.027, 1.101, 0.891]
glist = flist[:]
glist[4] *= 1.1

FinalModel = tf.keras.models.load_model('Model.keras')

def predict(X):
    return np.array(FinalModel.predict(np.array(X)).flat)

strings = ""

for index, parameter in enumerate(flist):
    modLow = flist[:]
    modHigh = flist[:]
    frac = 0.30
    modLow[index] = parameter
    modHigh[index] = parameter + (frac * abs(parameter))

    lowPred = predict([modLow])
    highPred = predict([modHigh])

    if float(np.array(lowPred.flat)[0]) < float(np.array(highPred.flat)[0]):
        strings += ("I\n")
    else:
        strings += ("D\n")

print(strings)
print(predict([flist]))


