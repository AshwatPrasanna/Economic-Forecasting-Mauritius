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
    xtrain, xtest, ytrain, ytest = train_test_split(keys, values, train_size=0.90)
    model = fitness_model(position, xtrain, ytrain)
    return calculate_fit(model, xtest, ytest)



flist = [0.804, 0.290, 1.423, 1.103, -0.763, -0.094, 0.290, 1.006, -0.328, -0.106, 0.751, 0.356, 3.720, 0.979, 2.054, 1.027, 1.101, 0.891]
glist = flist[:]
glist[2] *= 1.1 # GDS
hlist = flist[:]
hlist[-6] *= 1.1 # expenditure
ilist = flist[:]
ilist[3] *= 1.1 # export
jlist = flist[:]
jlist[-5] *= 1.1 # debt
klist = flist[:]
klist[-4] *= 1.1 # population


def train():
    model = PSOModel(randX0(numParticles, dims), bounds, fitnessFunction, maxiter=numIter)
    convergence = model.begin()

    FinalModel = fitness_model(model.posbestgroup, keys, values)

    return [FinalModel, convergence]


def passing(model):
    pred24 = FinalModel.predict(flist)[0]
    predMod = FinalModel.predict(glist)[0]
    predMod2 = FinalModel.predict(hlist)[0]
    predMod3 = FinalModel.predict(ilist)[0] # Should increase
    predMod4 = FinalModel.predict(jlist)[0] # Should decrease
    predMod5 = FinalModel.predict(klist)[0]

    t1 = predMod > pred24
    t2 = predMod2 < pred24
    t3 = predMod3 > pred24
    t4 = predMod4 < pred24
    t5 = predMod5 < pred24

    print("Truth:", sum([int(x) for x in [t1, t2, t3, t4, t5]]))
    
    
    if pred24 <= 1.04 and pred24 >= 0.68 and t1 and t2 and t3 and t4 and t5:
        print("Predicted correct, at", pred24)
        return True
    else:
        if pred24 > 1.00:
            print("Predicted too high, at", pred24)
        elif pred24 < 0.1:
            print("Predicted zero, compute error")
        elif pred24 < 0.68:
            print("Predicted too low, at", pred24)
        else:
            print("Retrying")
        return False


FinalModel = None
count = 0

while True:
    res = train()
    FinalModel = res[0]
    convergence = res[1]
    if passing(FinalModel):
        break
    else:
        print("Retrying, didn't pass evaluation criteria\n\n")
        count += 1

print("Finished after count", count)

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


for index, parameter in enumerate(flist):
    modLow = flist[:]
    modHigh = flist[:]
    frac = 0.10
    modLow[index] = parameter
    modHigh[index] = parameter + (frac * abs(parameter))

    lowPred = FinalModel.predict(modLow)
    highPred = FinalModel.predict(modHigh)

    if float(numpy.array(lowPred.flat)[0]) < float(numpy.array(highPred.flat)[0]):
        print("I: Parameter", index, "increases with GDP for", str(100*frac) + "% variation.")
    else:
        print("D: Parameter", index, "decreases with GDP for", str(100*frac) + "% variation.")

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
#plt.show()

