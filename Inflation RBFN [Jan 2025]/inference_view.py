from django.http import JsonResponse
import json
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import numpy

import math

# Extension for Double (Python float already supports operations directly)
def squared(x):
    return x ** 2

# Squared Euclidean distance
def sqeuclidean(arr1, arr2):
    return sum((a - b) ** 2 for a, b in zip(arr1, arr2))

# Gaussian RBF
def gaussian_rbf(point1, point2, beta):
    return math.exp(-sqeuclidean(point1, point2) / (2 * (beta ** 2)))



class RFBN:
  def __init__(self, centres, inputBetas, weights):
    self.inputBetas = inputBetas
    self.centres = centres
    self.weights = weights
  
  def hiddenLayerActivationMatrix(self, X): # X is an array of points, not a single one
    matrixlist = []
    for point in X:
      row = [gaussian_rbf(point, centre, self.inputBetas[index]) for index, centre in enumerate(self.centres)]
      matrixlist.append(row)

    # Now, matrixlist is a 2-dimensional list
    return numpy.array(matrixlist) # This is the matrix of activations
    
  def predict(self, X):
    M = self.hiddenLayerActivationMatrix([X])
    return M @ self.weights


    

# Constants
centres = [[-0.322, -0.127, -1.248, -0.809, -0.513, -1.076, 0.869, -0.127, 0.146, 1.522, -0.486, 0.315, -0.503, -1.337, -1.486, -0.948, -0.598, -0.884], [1.366, 1.978, -0.95, 0.473, 0.774, 0.534, 0.443, 1.978, 1.778, -1.551, 1.005, 2.067, 0.807, 0.539, 0.696, -0.336, 0.604, 1.017], [-2.008, -1.796, -2.096, -2.078, -2.101, -2.239, -1.379, -1.796, -2.069, 0.37, -2.25, -1.066, -2.127, -2.173, -2.193, -1.931, -2.098, -2.152], [1.848, 0.671, -0.964, -0.006, 0.455, 1.581, 1.944, 0.671, 2.079, -1.048, 0.108, -2.011, -1.727, 1.495, 1.543, -0.048, 0.4, 0.905], [2.25, 2.25, 2.25, 2.25, 2.25, 2.25, -0.105, 2.25, 1.278, 0.591, 2.25, 1.866, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25], [0.161, 0.496, -1.292, -0.554, -0.162, -0.687, 0.507, 0.496, 0.382, -0.735, -0.145, 0.495, -0.427, -0.85, -1.125, -0.837, -0.087, -0.452], [-0.723, -0.386, -1.392, -0.699, -1.276, -2.158, 1.428, -0.386, -0.562, -0.37, -1.92, -0.183, -1.27, -1.584, -1.623, -1.473, -1.16, -1.434], [0.723, 1.454, -1.004, 0.132, 0.479, -0.265, 1.137, 1.454, 1.282, -1.168, 0.64, 0.913, -0.107, -0.526, -0.44, -0.603, 0.266, 0.248], [-1.286, -0.995, -1.421, -1.76, -1.72, -2.208, -0.121, -0.995, -0.793, -0.127, -1.882, -0.494, -1.718, -1.873, -1.868, -1.868, -1.554, -1.787], [2.088, 1.104, 0.534, 1.785, 2.235, 2.019, 0.798, 1.104, 1.584, 2.25, 1.185, 0.765, 1.038, 1.656, 1.956, 1.146, 1.374, 1.635]]
weights = [-0.6147088121051709, 3.279055046725915, 1.6453172985934887, 1.1074323951115355, 1.0774835052935536, 6.60533120682634, 2.613113076020892, -8.282732943742534, -7.311695251599946, -0.06843029048049853]
betas = [7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347, 7.714546183262347]
scaling = 3.0

denorm_list_mins = [1990.33, 455366.67, -874000000.0, -9900333333.33, -9940666666.67, -128177000000.0, 19650.0, 455366.67, 4266.67, -1.22, -4933833333.33, -23491.33, -5614833333.33, -15534783333.33, 95548.0, 2282420447.49, 2939534754.37, -27477601550.27]
denorm_list_maxes = [2027.67, 612433.33, 134982000000.0, 398942333333.33, 412848666666.67, 962311000000.0, 56850.0, 612433.33, 35733.33, 12.52, 42536833333.33, 1602679.33, 99079833333.33, 233222683333.33, 365692.0, 31129939936.07, 137944637892.23, 495624800221.47]
denorm_ref_min = -28546333333.33
denorm_ref_max = 761792333333.33
model = RFBN(centres, betas, weights)

@csrf_exempt
@require_POST
def get_inference(request):
    try:
        data = json.loads(request.body)
        inputs = data.get('inputs')
        normalized_inputs = [
            2 * ((val - min_val) / (max_val - min_val)) - 1
            for val, min_val, max_val in zip(inputs, denorm_list_mins, denorm_list_maxes)
        ]
        predicted_value = model.predict(normalized_inputs)

        denormalized_value = (
            predicted_value * 0.5 * (denorm_ref_max - denorm_ref_min)
            + ((denorm_ref_max + denorm_ref_min) / 2)
        )
        denormalized_value = max(0, denormalized_value)
        return JsonResponse({'status': 'success', 'prediction': denormalized_value}, status=201)
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
