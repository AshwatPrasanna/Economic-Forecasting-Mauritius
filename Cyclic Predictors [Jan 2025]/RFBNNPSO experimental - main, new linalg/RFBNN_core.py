from __future__ import division
import random
import math
import numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

normNorm = 20.0
  

# MARK: Defining a standard RBFNN
def gaussianRBF(x, center, beta):
  return numpy.exp((-abs(cdist([x], [center], metric='sqeuclidean')[0][0]))/float(2*(beta**2)))

def dist(x1, x2):
  return float(cdist([x1], [x2], metric='euclidean')[0][0])

def euclideanDistance(p1, p2):
  return cdist([p1], [p2], metric='euclidean')[0][0]


class RFBN:
  def __init__(self, k, inputBetas, scaling, includeFit=False):
    self.k = k
    self.inputBetas = inputBetas
    self.scaling = scaling
    self.includeFit = includeFit
  
  def hiddenLayerActivationMatrix(self, X): # X is an array of points, not a single one
    matrixlist = []
    for point in X:
      row = [gaussianRBF(point, centre, self.inputBetas[index]) for index, centre in enumerate(self.centres)] + [1]
      matrixlist.append(row)

    # Now, matrixlist is a 2-dimensional list
    return numpy.array(matrixlist) # This is the matrix of activations

  def fit_model(self, X, Y): # Here, includeFit is true if the PSO-based number of neurons is to be taken into account    
    kmeans = KMeans(n_clusters=self.k)
    arrX = numpy.array(X)
    kmeans.fit(arrX)
    
    self.centres = []

    for centre in kmeans.cluster_centers_:
      normVec = numpy.linalg.norm(numpy.array(centre))
      self.centres.append(self.scaling * (normNorm/normVec) * centre)

    # GET cmax
    cmax = 0.0

    for x1 in self.centres:
      for x2 in self.centres:
        d = dist(x1, x2)
        if d > cmax:
          cmax = d
    

    # Get sigmas
      

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


