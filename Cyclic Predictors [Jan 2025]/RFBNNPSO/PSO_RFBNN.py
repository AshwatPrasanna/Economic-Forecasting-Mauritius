from __future__ import division
import random
import math
import numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from RFBNN_core import *
import time

# MARK: Defining a Particle

class Particle:
  def __init__(self, x0, fitness, velocity_min=-1, velocity_max=1, w=1.2, c1=1.2, c2=1.2):
    self.position_i = x0
    self.velocity_i = []
    self.err_i = -1
    self.bestPos = x0
    self.bestErr = -1
    self.fitnessFunction = fitness
    self.w = w
    self.c1 = c1
    self.c2 = c2

    for i in range(0, len(x0)):
      self.velocity_i.append(random.uniform(velocity_min, velocity_max))
  def update_velocity(self, pos_best_g):
    for i in range(0, len(self.position_i)):
      vel_cog = random.random()*self.c1*(self.bestPos[i] - self.position_i[i])
      vel_soc = random.random()*self.c2*(pos_best_g[i] - self.position_i[i])
      self.velocity_i[i] = self.w*random.random()*self.velocity_i[i]+vel_cog+vel_soc

  def update_position(self):
    for i in range(0, len(self.position_i)):
      self.position_i[i] += self.velocity_i[i]

      if self.position_i[i] < 0: #bounds[i][0]:
        self.position_i[i] = 0 #bounds[i][0]

      if self.position_i[i] > 1: #bounds[i][1]:
        self.position_i[i] = 1 #bounds[i][1]

  def fitness(self):
    self.err_i = self.fitnessFunction(self.position_i)

    if self.err_i < self.bestErr or self.bestErr == -1:
      self.bestErr = self.err_i
      self.bestPos = self.position_i

    return self.err_i


# MARK: Defining the General PSO

class PSOModel:
  def __init__(self, x0list, bounds, fitnessFunction, maxiter=30):
    self.groupbesterror = -1
    self.posbestgroup = []
    self.maxiter = maxiter

    self.swarm = []

    for x0 in x0list:
      particle = Particle(x0, fitnessFunction)
      self.swarm.append(particle)

  def begin(self):
    i = 0

    convergenceArr = []

    while i <= self.maxiter:
      for particle in self.swarm:
        particle.fitness()

        if particle.err_i < self.groupbesterror or self.groupbesterror == -1:
          self.posbestgroup = particle.position_i
          self.groupbesterror = particle.err_i

      for particle in self.swarm:
        particle.update_velocity(self.posbestgroup)
        particle.update_position()

      print("\n\nFinished Iteration", i)
      print("Best Error:", self.groupbesterror)
      convergenceArr.append(self.groupbesterror)
      i += 1
    
    print("\n\nFinished PSO with best error:", self.groupbesterror)

    return convergenceArr
  


# MARK: Model Runtime Parameters

ind = 18 # Number of Indicators (i.e. dimensions in the space of input neurons)

numParticles = 30
numIter = 30

dims = 2

# These two functions take a range of [0, 1] and change the range

def sigmaProc(beta):
  return beta*29.9 + 0.1
def inputProc(weight):
  return int(weight*10.0) + 5

bounds = [[0, 1]]*dims

# MARK: Using Testing Data for Model Fitness (defined in other module)

# --> keys and values defined in another module



def MAE(model, keys, values): # The cost function, given a model
  rmssum = 0.0
  maesum = 0.0

  for index, dataKey in enumerate(keys):
    result = (model.predict(dataKey))
    reference = (values[index])
    diff = float(result - reference)
    rmssum += (diff)**(2)
    maesum += abs(diff)

  m = float(len(keys))
  RMSE = math.sqrt(rmssum/m)
  
  MAE = maesum/m

  return MAE # Swap with any equivalent calculation of model performance

def MAPE(model, keys, values): # The cost function, given a model
  maesum = 0.0

  for index, dataKey in enumerate(keys):
    result = (model.predict(dataKey))
    reference = (values[index])
    diff = float(result - reference)
    maesum += abs(diff/float(reference))

  m = float(len(keys))  
  MAPE = 100 * (maesum/m)

  return MAPE # Swap with any equivalent calculation of model performance


def RMSE(model, keys, values): # The cost function, given a model
  rmssum = 0.0

  for index, dataKey in enumerate(keys):
    result = (model.predict(dataKey))
    reference = (values[index])
    diff = float(result - reference)
    rmssum += (diff)**(2)

  m = float(len(keys))
  RMSE = math.sqrt(rmssum/m)
  
  return RMSE


def R2(model, keys, values):
  results = [model.predict(key) for key in keys]

  return r2_score(results, values)


def calculate_fit(model, keys, values):
  mae = MAE(model, keys, values)
  rmse = RMSE(model, keys, values)
  mape = MAPE(model, keys, values)
  r2 = R2(model, keys, values)

  return mae + rmse + mape + r2


  



def fitness_model(pos, trainingData, trainingRefs):
  numInput = inputProc(pos[0])
  sigma = sigmaProc(pos[1])

  model = RFBN(numInput, ([sigma]*numInput))

  return model.fit_model(trainingData, trainingRefs)
  
##    inputNeuronCentres = []
##
##    for i in range(0, modelNumberOfInputNeurons):
##      inputNeuronCentres.append(pos[ind*i:ind*(i+1)])
##
##    inputNeuronExpansions = [betaProc(ele) for ele in pos[modelNumberOfInputNeurons*ind:modelNumberOfInputNeurons*(ind+1)]]
##
##    hiddenNeuronWeights = []
##    n = modelNumberOfInputNeurons
##    base = modelNumberOfInputNeurons*(ind+1)
##
##    for i in range(0, modelNumberOfHiddenNeurons):
##      hiddenNeuronWeights.append([weightProc(x) for x in pos[base+(n*i):base+(n*(i+1))]])
##
##    base += (n*(i+1))
##    outputNeuronWeights = [weightProc(x) for x in pos[base:(base+modelNumberOfHiddenNeurons)]]
##
##    return RFBN(inputNeuronCentres, inputNeuronExpansions, hiddenNeuronWeights, outputNeuronWeights)


