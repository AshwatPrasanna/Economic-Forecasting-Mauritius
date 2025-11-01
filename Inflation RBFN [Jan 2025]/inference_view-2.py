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
centres = [[10.594, 16.23, -7.786, 4.276, 3.53, 5.062, 16.23, 14.331, -13.352, 6.59, 15.824, 5.646, 3.342, 2.927, 4.459, -3.149, 4.677, 7.949], [-8.847, -7.214, -9.765, -10.591, -13.601, -5.47, -7.214, -6.296, -3.873, -4.571, -3.01, -11.399, -12.041, -11.05, -11.784, -11.729, -10.334, -11.417], [5.015, 8.765, -16.968, 8.109, -9.562, 11.203, 8.765, 8.953, 5.812, 3.375, 10.078, -3.609, -11.015, -11.906, -14.296, -11.109, 0.0, -4.453], [10.3, 10.3, 10.3, 10.3, 10.3, -0.481, 10.3, 5.85, 2.705, 10.3, 8.542, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3], [-3.292, -1.298, -12.758, -12.39, -10.999, 8.883, -1.298, 1.492, 15.559, -4.477, 3.22, -5.142, -13.668, -11.746, -15.191, -9.691, -6.113, -9.037], [13.572, -0.296, -6.459, 3.652, 13.466, 11.525, -0.296, 13.424, -5.066, 1.266, -15.83, -12.58, 9.034, 11.525, 11.989, 0.971, 4.77, 6.944], [-5.997, -4.251, -8.995, 3.871, -16.156, 6.556, -4.251, -3.558, -3.088, -14.388, -1.947, -9.957, -12.038, -13.291, -12.419, -10.964, -8.727, -11.121], [-9.797, -8.762, -10.225, -9.918, -10.924, -6.724, -8.762, -10.093, 1.807, -10.847, -5.199, -10.375, -10.598, -9.761, -10.697, -9.42, -10.232, -10.492], [-7.892, -6.01, -8.728, -10.69, -13.964, 0.461, -6.01, -4.647, -0.0, -13.104, -3.136, -10.657, -11.707, -11.488, -11.74, -11.759, -9.626, -11.194], [12.566, 6.644, 3.214, 13.541, 12.151, 4.803, 6.644, 9.533, 13.541, 7.24, 4.604, 6.247, 9.912, 12.386, 11.772, 6.897, 8.197, 9.876], [-4.688, -1.468, -11.522, -8.849, -15.706, 14.435, -1.468, -4.731, -2.388, -12.88, -0.767, -8.806, -11.347, -9.288, -11.544, -10.777, -8.368, -10.054], [0.0, 2.768, -16.986, -13.977, -8.441, 1.937, 2.768, 2.214, -15.879, -8.718, 3.494, -7.3, -11.832, -10.102, -15.049, -10.413, -1.695, -6.885], [6.484, 14.15, -13.657, 6.438, -5.535, 12.047, 14.15, 12.178, -12.326, 9.368, 8.847, -3.237, -8.596, -7.089, -8.791, -8.503, 1.712, -0.633], [11.911, 9.322, -6.815, 3.498, 8.412, 15.167, 9.322, 15.167, -9.282, 0.748, -11.951, -11.223, 11.587, 9.201, 9.322, -1.577, 0.849, 5.5]]
weights = [123.52323978280339, -436.03749163069995, 261.6950621608009, 232.04528046661295, -142.93211389481615, 332.7131455224207, -277.56852203262895, 263.5930056675024, 449.52564320209245, -404.92949702735257, 90.23303739394932, -227.0713489333839, -0.7190303218692975, -264.0771510296672]
betas = [74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9, 74.9]
scaling = 2.0


denorm_list_mins = [1990.33,455366.67,-874000000.00,-9900333333.33,-128177000000.00,19650.00,455366.67,4266.67,-1.22,-4933833333.33,-23491.33,-5614833333.33,-15534783333.33,-60877500000,95548.00,2282420447.49,2939534754.37,-27477601550.27]
denorm_list_maxes = [2027.67,612433.33,134982000000.00,398942333333.33,962311000000.00,56850.00,612433.33,35733.33,12.52,42536833333.33,1602679.33,99079833333.33,233222683333.33,5.98983E+11,365692.00,31129939936.07,137944637892.23,495624800221.47]


denorm_ref_min = -28546333333.33
denorm_ref_max = 761792333333.33
model = RFBN(centres, betas, weights)


#############################

in2024 = [2024, 556700.00, 163683000000.00, 361260000000.0, 903777000.00, 36500.00, 556700.00, 35820.00, 3.40, 15927000000.00, 1400000.00, 65379000000.00, 571130000000.00, 592000000000.0, 508079.00, 31526000000.00, 144700000000.00, 466929000000.00]
normalized_inputs = [
    2 * ((val - min_val) / (max_val - min_val)) - 1
    for val, min_val, max_val in zip(in2024, denorm_list_mins, denorm_list_maxes)
]

predicted_value = model.predict(normalized_inputs)

denormalized_value = (
    predicted_value * 0.5 * (denorm_ref_max - denorm_ref_min)
    + ((denorm_ref_max + denorm_ref_min) / 2)
)
denormalized_value = max(0, float(list(numpy.array(denormalized_value).flat)[0]))

print(denormalized_value)

####################### TESTING ACCURACY


from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from Training_Data import *
from RFBNN_core import *
from PSO_RFBNN import *


data = readData("Data.csv")
keys = data[0]
values = data[1]


print("MAE:", MAE(model, keys, values))
print("RMSE:", RMSE(model, keys, values))
print("MAPE:", MAPE(model, keys, values))
print("R2", R2(model, keys, values))


########################


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
        denormalized_value = max(0, float(list(numpy.array(denormalized_value).flat)[0]))
        return JsonResponse({'status': 'success', 'prediction': denormalized_value}, status=201)
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
