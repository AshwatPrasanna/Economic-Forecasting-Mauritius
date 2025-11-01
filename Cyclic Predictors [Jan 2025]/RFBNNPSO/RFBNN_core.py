from __future__ import division
import random
import math
import numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans


# MARK: Defining a standard RBFNN
def gaussianRBF(x, center, beta):
  return numpy.exp((-abs(cdist([x], [center], metric='sqeuclidean')[0][0]))/float(2*(beta**2)))

def euclideanDistance(p1, p2):
  return cdist([p1], [p2], metric='euclidean')[0][0]

class InputLayer:
  def __init__(self, centers, betas):
    neurons = [] # Each neuron is a tuple consisting of the center and the beta value (expansion constant)
    for index, center in enumerate(centers):
      neurons.append((center, betas[index]))
    self.neurons = neurons

  def unweightedActivation(self, x):
    activations = []
    for neuron in self.neurons:
      activations.append((gaussianRBF(x, neuron[0])**neuron[1]))
    return activations

class HiddenNeuron:
  def __init__(self, weights):
    self.weights = weights

  def rawActivation(self, inputlayerunweightedactivation):
    weightedActivations = [a*b for a, b in zip(self.weights, inputlayerunweightedactivation)]
    return sum(weightedActivations)

class HiddenLayer:
  def __init__(self, weightsArray):
    self.neurons = [HiddenNeuron(weights) for weights in weightsArray]

  def rawActivations(self, inputlayer):
    return [neuron.rawActivation(inputlayer) for neuron in self.neurons]

class OutputLayer:
  def __init__(self, weights):
    self.weights = weights

  def compute(self, unweightedHiddenLayerOutputLayers):
    outputs = [a*b for a,b in zip(unweightedHiddenLayerOutputLayers, self.weights)]
    return sum(outputs)

class BasicRFBN:
  def __init__(self, inputCenters, inputBetas, hiddenWeightsArray, outputWeights):
    self.inputCenters = inputCenters
    self.inputBetas = inputBetas
    self.hiddenWeightsArray = hiddenWeightsArray
    self.outputWeights = outputWeights
  def compute(self, x):
    inputLayer = InputLayer(self.inputCenters, self. inputBetas)
    hiddenLayer = HiddenLayer(self.hiddenWeightsArray)
    output = OutputLayer(self.outputWeights)
    return output.compute(hiddenLayer.rawActivations(inputLayer.unweightedActivation(x)))




class RFBN:
  def __init__(self, k, inputBetas):
    self.k = k
    self.inputBetas = inputBetas
  
  def hiddenLayerActivationMatrix(self, X): # X is an array of points, not a single one
    matrixlist = []
    for point in X:
      row = [gaussianRBF(point, centre, self.inputBetas[index]) for index, centre in enumerate(self.centres)]
      matrixlist.append(row)

    # Now, matrixlist is a 2-dimensional list
    return numpy.array(matrixlist) # This is the matrix of activations

  def fit_model(self, X, Y): # Here, X is a list of points (which are lists of coords), and Y is a list of ref values (a vector)
    kmeans = KMeans(n_clusters=self.k, random_state=2)
    arrX = numpy.array(X)
    kmeans.fit(arrX)
    self.centres = kmeans.cluster_centers_

    M = self.hiddenLayerActivationMatrix(X)     
    
    #Turning Y into a horizontal matrix:
    ymat = numpy.rot90(numpy.array([Y]), k=-1)
    w = numpy.linalg.pinv(M) @ ymat # Here, w is a matrix with 1 column
    
    self.weights = list(numpy.rot90(w))[0]

    return self
    
  def predict(self, X):
    M = self.hiddenLayerActivationMatrix([X])
    return M @ self.weights
    
  


##model = RFBN(2, [2, 2, 2])
##
##model.fit_model([[1, 1, 1], [1, 2, 1], [1, 1, 1], [1, 1, 1], [17, 21, 12], [18, 21, 13]], [1, 1, 1, 1, 3, 3])
##
##print(model.predict([1, 2, 1]))


