from __future__ import division
import random
import math
from numpy import *
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from ANNV2 import *
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
  def __init__(self, x0list, bounds, fitnessFunction, checkNaN, maxiter=30):
    self.groupbesterror = -1
    self.posbestgroup = []
    self.maxiter = maxiter
    self.checkNaN = checkNaN

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

      if i == 4:
        bestParticle = self.swarm[0]

        for particle in self.swarm:
          if particle.bestErr != -1 and particle.bestErr < bestParticle.bestErr:
            bestParticle = particle

        if self.checkNaN(bestParticle.bestPos):
          raise ValueError
      
      i += 1
    
    print("\n\nFinished PSO with best error:", self.groupbesterror)

    return convergenceArr
  


# MARK: Model Runtime Parameters

ind = 18 # Number of Indicators (i.e. dimensions in the space of input neurons)

numParticles = 10
numIter = 25
nEpochs = 100

dims = 3


# Parameters that can vary are number of neurons per layer, number of layers, learning rate



def nProc(n):
  return int(n*10.0 + 10)
def layerProc(n):
  return int(n*3.0) + 2

def learningRateProc(l):
  return float(l/3.0)

bounds = [[0, 1]]*dims

# MARK: Using Testing Data for Model Fitness (defined in other module)

# --> keys and values defined in another module


##
##def MAE(model, keys, values): # The cost function, given a model
##  maesum = 0.0
##
##  for index, dataKey in enumerate(keys):
##    result = results[index]
##    reference = (values[index])
##    diff = float(result - reference)
##    maesum += abs(diff)
##
##  m = float(len(keys))
##  
##  MAE = maesum/m
##
##  return MAE 


def MAPE(model, res, values): # The cost function, given a model
  maesum = 0.0

  for index, result in enumerate(res):
    result = res[index]
    reference = (values[index])
    diff = float(array(result).flat[0] - array(reference).flat[0])
    maesum += abs(diff/float(reference))

  m = float(len(res))  
  MAPE = 100 * (maesum/m)

  return MAPE

##
##
##def RMSE(model, keys, values): # The cost function, given a model
##  
##  return Loss_Function(values, model.predict(keys))
##
##

def R2(results, values):
  return r2_score(results, values)


def calculate_fit(model, datapoints, values):
  results = (model.predict(array(datapoints)))
  mape = MAPE(model, results, values)

  return mape



def fitness_model(pos, trainingData, trainingRefs, final=False):
  n = nProc(pos[0])
  layers = layerProc(pos[1])
  learning = learningRateProc(pos[2])

  model = Network(ind, n, layers)
  
  if final:
    model.train(trainingData, trainingRefs, learning, 20000)
  else:
    model.train(trainingData, trainingRefs, learning, nEpochs)

  return model
