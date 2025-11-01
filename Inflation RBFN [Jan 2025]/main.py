from __future__ import division
import random
import math
import numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


from RFBNN_core import *
from PSO_RFBNN import *
from Training_Data import *

import matplotlib.pyplot as plt


data = readData("Data.csv")
keys = data[0]
values = data[1]


def randX0(particles, length):
    x0list = []
    for i in range(0, particles):
      x0 = []
      for _ in range(0, length):
        x0.append(random.random())
      x0list.append(x0)
    return x0list

def fitnessFunction(position):
    xtrain, xtest, ytrain, ytest = train_test_split(keys, values, train_size=0.90, random_state=40)
    model = fitness_model(position, xtrain, ytrain)
    return calculate_fit(model, xtest, ytest)


def train():
    model = PSOModel(randX0(numParticles, dims), bounds, fitnessFunction, maxiter=numIter)
    convergence = model.begin()

    FinalModel = fitness_model(model.posbestgroup, keys, values)

    return [FinalModel, convergence]



res = train()
FinalModel = res[0]
convergence = res[1]

print("MAE:", MAE(FinalModel, keys, values))
print("RMSE:", RMSE(FinalModel, keys, values))
print("MAPE:", MAPE(FinalModel, keys, values))
print("R2", R2(FinalModel, keys, values))



l = []
for arr in list(FinalModel.centres):
    l2 = (list(arr))
    l3 = []
    for ele in l2:
        l3.append(round(ele, 3))
    l.append(l3)


##
print("Centers:", l)
print("Weights:", list(FinalModel.weights))
print("Betas:", FinalModel.inputBetas)
print("Scaling:", FinalModel.scaling)

##
##for index, parameter in enumerate(flist):
##    modLow = flist[:]
##    modHigh = flist[:]
##    frac = 0.10
##    modLow[index] = parameter
##    modHigh[index] = parameter + (frac * abs(parameter))
##
##    lowPred = FinalModel.predict(modLow)
##    highPred = FinalModel.predict(modHigh)
##
##    if float(numpy.array(lowPred.flat)[0]) < float(numpy.array(highPred.flat)[0]):
##        print("I: Parameter", index, "increases with GDP for", str(100*frac) + "% variation.")
##    else:
##        print("D: Parameter", index, "decreases with GDP for", str(100*frac) + "% variation.")

##
##testlist = [0.696, 0.360, 0.178, 0.595, 0.745, 0.673, 0.266, 0.368, 0.528, 0.750, 0.395, 0.255, 0.340, 0.552, 0.652, 0.382, 0.458, 0.545]
##testlist2 = [0.804, 0.290, 1.423, 0.818, 0.846, -0.763, -0.094, 64.089, 1.006, -0.328, -0.121, 0.751, 0.356, 3.717, 2.054, 1.027, 1.100, 0.890]
##print("Test List:", FinalModel.predict(xtest[0]), ytest[0])
##print("Also Test/Train List:", FinalModel.predict(testlist), 0.517)
##print("Out of range:", FinalModel.predict(testlist2), 0.75)


########### PLOTTING ###################

PlotY = convergence
PlotX = range(1, len(PlotY) + 1)

plt.plot(PlotX, PlotY)
plt.xlabel("Iteration Number")
plt.ylabel("Compound error (MAE + MAPE + RMSE)")
plt.title("RFBN-PSO Convergence Curve")
plt.show()

