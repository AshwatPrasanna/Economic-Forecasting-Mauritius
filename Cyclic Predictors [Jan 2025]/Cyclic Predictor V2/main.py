from __future__ import division
import random
import math
import numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd


from RFBNN_core import *
from PSO_RFBNN import *
from Training_Data import *
from Py_SARIMAX import *
from train import *

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


print("Training will commence if no model is found.")

if MainModel == None or GrowthModel == None:
    print("No model found, so training now")
    trainCyclicModels()
    replaceFullData()
    print("Fixed Missing Data")
    trainCyclicModels()
    print("Please store the values of each array in the training file")
    print("FutureModelsEx:", [x.export() for x in FutureModels])
    print("MainModelEx:", MainModel.export())
    print("GrowthModelEx:", GrowthModel.export())
    print("ARIMAModels:", ARIMAModels)
    numpy.savetxt("Data.csv", numpy.array(data), delimiter=",")
    numpy.savetxt("DataFull.csv", numpy.array([[]]), delimiter=",")
    
    # All 4 key ones, and data
else:
    print("Model found, skipping training")

originalData = data[:]


def nowCastPrepare(newRow): # Missing entries are to be denoted by -20
    data = originalData[:]
    fullData = [newRow]
    replaceFullData()

# Call nowCastPrepare before running any nowCast - note that doing this will remove previous entries
# When calling forecase, use data[:], instead of directly using data

