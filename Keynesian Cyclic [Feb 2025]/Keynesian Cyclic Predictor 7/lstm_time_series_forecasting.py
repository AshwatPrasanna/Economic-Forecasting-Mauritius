import tensorflow as tf
import os
import pandas as pd
import numpy as np

df = pd.read_csv("DataNorm.csv")

df.index = pd.to_datetime(df['Year'], format='%Y')

temp = df['P2']
temp.plot()
#
## [[[1], [2], [3], [4], [5]]] [6]
## [[[2], [3], [4], [5], [6]]] [7]
## [[[3], [4], [5], [6], [7]]] [8]
#

def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

#WINDOW_SIZE = 5
X1, y1 = df_to_X_y(temp, 5)
#X1.shape, y1.shape
#
X_train1, y_train1 = X1[:18], y1[:18]
X_val1, y_val1 = X1[18:21], y1[18:21]
X_test1, y_test1 = X1[21:], y1[21:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape
#
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
#
model1 = Sequential()
model1.add(GRU(64))
model1.add(Flatten())
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))
#
model1.summary()
#

checkpoint_dir = os.path.dirname("Data.csv")

cp1 = ModelCheckpoint('modelT.keras', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])
#
model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, callbacks=[cp1])
#
train_predictions = list(model1.predict(X_train1).flatten())
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})


for i in range(0, 30):
  train_predictions.append(float(model1.predict(np.array([[[a] for a in train_predictions[-5:]]])).flatten()[0]))
  

#
## Split
#
#from sklearn.metrics import mean_squared_error as mse
#
def plot_predictions1(model, X, y, start=0, end=20):
  predictions = model.predict(X).flatten()
  df = pd.DataFrame(data={'Predictions': predictions, 'Actuals':y})
  plt.plot(df['Predictions'][start:end])
  plt.plot(df['Actuals'][start:end])
  return df, mse(predictions, y)

#plot_predictions1(model1, X_test1, y_test1)
#
#model2 = Sequential()
#model2.add(InputLayer((5, 1)))
#model2.add(Conv1D(64, kernel_size=2, activation='relu'))
#model2.add(Flatten())
#model2.add(Dense(8, 'relu'))
#model2.add(Dense(1, 'linear'))
#model2.summary()
#
#cp2 = ModelCheckpoint('model2/', save_best_only=True)
#model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#
#model2.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp2])
#
#plot_predictions1(model2, X_test1, y_test1)
#
#model3 = Sequential()
#model3.add(InputLayer((5, 1)))
#model3.add(GRU(64))
#model3.add(Dense(8, 'relu'))
#model3.add(Dense(1, 'linear'))
#model3.summary()
#
#cp3 = ModelCheckpoint('model3/', save_best_only=True)
#model3.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#
#model3.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp3])
#

def df_to_X_y2(df, window_size=6):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][0]
    y.append(label)
  return np.array(X), np.array(y)
#


X2, y2 = df_to_X_y2(temp_df)

print(X2)
#X2.shape, y2.shape
#
X2_train, y2_train = X2[:18], y2[:18]
X2_val, y2_val = X2[18:21], y2[18:21]
X2_test, y2_test = X2[21:], y2[21:]
#X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape
#
#temp_training_mean = np.mean(X2_train[:, :, 0])
#temp_training_std = np.std(X2_train[:, :, 0])
#
#def preprocess(X:
#  X[:, :, 0] = (X[:, :, 0] - temp_training_mean) / temp_training_std
#  return X
#
#preprocess(X2_train)
#preprocess(X2_val)
#preprocess(X2_test)
#
model4 = Sequential()
model4.add(InputLayer((6, 5)))
model4.add(LSTM(64))
model4.add(Dense(8, 'relu'))
model4.add(Dense(1, 'linear'))
#
model4.summary()
#
cp4 = ModelCheckpoint('model4.keras', save_best_only=True)
model4.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])
#
model4.fit(X2_train, y2_train, validation_data=(X2_val, y2_val), epochs=10, callbacks=[cp4])
#
plot_predictions1(model4, X2_test, y2_test)
#
p_temp_df = pd.concat([df['p (mbar)'], temp_df], axis=1)

#
#def df_to_X_y3(df, window_size=7):
#  df_as_np = df.to_numpy()
#  X = []
#  y = []
#  for i in range(len(df_as_np)-window_size):
#    row = [r for r in df_as_np[i:i+window_size]]
#    X.append(row)
#    label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
#    y.append(label)
#  return np.array(X), np.array(y)
#
#X3, y3 = df_to_X_y3(p_temp_df)
#X3.shape, y3.shape
#
#X3_train, y3_train = X3[:60000], y3[:60000]
#X3_val, y3_val = X3[60000:65000], y3[60000:65000]
#X3_test, y3_test = X3[65000:], y3[65000:]
#X3_train.shape, y3_train.shape, X3_val.shape, y3_val.shape, X3_test.shape, y3_test.shape
#
#p_training_mean3 = np.mean(X3_train[:, :, 0])
#p_training_std3 = np.std(X3_train[:, :, 0])
#
#temp_training_mean3 = np.mean(X3_train[:, :, 1])
#temp_training_std3 = np.std(X3_train[:, :, 1])
#
#def preprocess3(X):
#  X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
#  X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3
#  return X
#
#def preprocess_output(y):
#  y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
#  y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
#  return y
#
#preprocess3(X3_train)
#preprocess3(X3_val)
#preprocess3(X3_test)
#
#preprocess_output(y3_train)
#preprocess_output(y3_val)
#preprocess_output(y3_test)
#
#model5 = Sequential()
#model5.add(InputLayer((7, 6)))
#model5.add(LSTM(64))
#model5.add(Dense(8, 'relu'))
#model5.add(Dense(2, 'linear'))
#
#model5.summary()
#
#cp5 = ModelCheckpoint('model5/', save_best_only=True)
#model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#
#model5.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp5])
#
#def plot_predictions2(model, X, y, start=0, end=100):
#  predictions = model.predict(X)
#  p_preds, temp_preds = predictions[:, 0], predictions[:, 1]
#  p_actuals, temp_actuals = y[:, 0], y[:, 1]
#  df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
#                          'Temperature Actuals':temp_actuals,
#                          'Pressure Predictions': p_preds,
#                          'Pressure Actuals': p_actuals
#                          })
#  plt.plot(df['Temperature Predictions'][start:end])
#  plt.plot(df['Temperature Actuals'][start:end])
#  plt.plot(df['Pressure Predictions'][start:end])
#  plt.plot(df['Pressure Actuals'][start:end])
#  return df
#
#plot_predictions2(model5, X3_test, y3_test)
#
#def postprocess_temp(arr):
#  arr = (arr*temp_training_std3) + temp_training_mean3
#  return arr
#
#def postprocess_p(arr):
#  arr = (arr*p_training_std3) + p_training_mean3
#  return arr
#
#def get_predictions_postprocessed(model, X, y):
#  predictions = model.predict(X)
#  p_preds, temp_preds = postprocess_p(predictions[:, 0]), postprocess_temp(predictions[:, 1])
#  p_actuals, temp_actuals = postprocess_p(y[:, 0]), postprocess_temp(y[:, 1])
#  df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
#                          'Temperature Actuals':temp_actuals,
#                          'Pressure Predictions': p_preds,
#                          'Pressure Actuals': p_actuals
#                          })
#  return df
#
#post_processed_df = get_predictions_postprocessed(model5, X3_test, y3_test)
#post_processed_df
#
#start, end = 0, 100
#plt.plot(post_processed_df['Temperature Predictions'][start:end])
#plt.plot(post_processed_df['Temperature Actuals'][start:end])
#
#plt.plot(post_processed_df['Pressure Predictions'][start:end])
#plt.plot(post_processed_df['Pressure Actuals'][start:end])
#
#model6 = Sequential()
#model6.add(InputLayer((7, 6)))
#model6.add(LSTM(32, return_sequences=True))
#model6.add(LSTM(16))
#model6.add(Dense(8, 'relu'))
#model6.add(Dense(2, 'linear'))
#
#model6.summary()
#
#cp6 = ModelCheckpoint('model6/', save_best_only=True)
#model6.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#
#model6.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp6])
#
#model7 = Sequential()
#model7.add(InputLayer((7, 6)))
#model7.add(Conv1D(64, kernel_size=2, activation='relu'))
#model7.add(Flatten())
#model7.add(Dense(8, 'relu'))
#model7.add(Dense(2, 'linear'))
#model7.summary()
#
#cp7 = ModelCheckpoint('model6/', save_best_only=True)
#model7.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#
#model7.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp7])
