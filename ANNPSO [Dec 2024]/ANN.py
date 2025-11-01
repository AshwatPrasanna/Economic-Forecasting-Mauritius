from numpy import *


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def compute_activations():
        pass

    def back_prop():
        pass


class NeuralNetwork:

    # The weights are expressed in a 3D array - an array of arrays of arrays of floats, with the nth array corresponding to the nth layer, in which each array corresponds to a neuron
    # The activation computed for a layer is expressed

    def _activation_function(self, x):
        return 1 / (1 + exp(-min(x, 500)))

    def _activation_derivative(self, X):
        return [[x_e * (1.0 - x_e) for x_e in x] for x in X]

    def __init__(self, nHiddenLayers, layerSize, inputDim):
        random.seed(1) 
        self.hidden = nHiddenLayers
        self.layerSize = layerSize
        self.inputDim = inputDim
        self.hiddenWeights = random.random((nHiddenLayers, layerSize, layerSize)) # This is a 3D array

##        for _ in range(0, nHiddenLayers):
##            l = []
##            for _ in range(0, layerSize):
##                l2 = []
##                for _ in range(0, layerSize):
##                    l2.append(random.uniform(1.0, 3.0) - 1.0)
##                l.append(l2)
##            self.hiddenWeights.append(l)
##
##        self.hiddenWeights = array(self.hiddenWeights)

        self.firstWeights = []

        for _ in range(0, layerSize):
            l = []
            for _ in range (0, inputDim):
                l.append(random.uniform(1.0, 3.0) - 1.0)
                
            self.firstWeights.append(l)

        self.outputWeights = []

        for _ in range(0, layerSize):
            self.outputWeights.append(random.uniform(1.0, 3.0) - 1.0)


    def getActivationMatrix(self, x):
        layerActivations = []

        firstLayerActivation = [self._activation_function(y) for y in array(self.firstWeights) @ rot90(array([x]), k=-1)]

        layerActivations.append(list(rot90(firstLayerActivation))[0])

        for i in range(0, self.hidden):
            prevLayer = array([layerActivations[-1]])
            weights = array(self.hiddenWeights[i])


            computed = list(rot90(weights @ rot90(array(prevLayer), k=-1)))[0]

            print([self._activation_function(y) for y in computed])

            layerActivations.append([self._activation_function(y) for y in computed])

        return layerActivations

    

    def core_process(self, x): # Here x is an array of length inputDim

        layerActivations = self.getActivationMatrix(x)

        print(layerActivations, "\n\n\n")
        
        sumV = 0.0
        for i, val in enumerate(layerActivations[-1]):
            sumV += val * self.outputWeights[i]

        sumV = self._activation_function(sumV)

        return sumV


    def train(self, inputs, outputs, numIter):
        for iteration in range(0, numIter):
            results = []
            for inp in inputs:
                results.append([self.core_process(inp)])

            resultsArr = rot90(array(results))
            inputsArr = array(inputs)
            outputArr = array([outputs])

            error = outputArr - resultsArr # desired and actual
            procerr = rot90(error * array(self._activation_derivative(resultsArr)), k=-1)
##            print("RES", resultsArr)
##            print("OUT", outputArr)
##            print("RAWERR", error)
##            print("DER", array(self._activation_derivative(resultsArr)))
##            print("PROCERR", procerr)
##            print("INPUTARR", inputsArr)


            factor = dot(inputsArr, procerr)
##            print("FAC", factor)
            self.hiddenWeights += factor
##            print("WEIGHTS", self.hiddenWeights)

#######################

neural_network = NeuralNetwork(1, 3, 3)

inputs = array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]) 
outputs = array([1.0, 0.0, 1.0])

#print(neural_network.hiddenWeights)
  
neural_network.train(inputs, outputs, 1000)

print(neural_network.core_process([1.0, 0.0, 0.0]))
