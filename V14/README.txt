Description of how this works:
- The model is trained for each of the 33 variables independently using an ARIMA. The order of this ARIMA is automatically determined and used for subsequent time series prediction
- An ANN is trained on the data, using the 32 variables to predict the GDP of the model, supplementing the GRU/ARIMA predictions using a weighted average.

Steps to replace the dataset:
1. Replace the Data file, maintaining the same format (ensure the respective headers are labelled P1, P2, P3, ..., P32)
2. Ensure there is no formatting of these numbers, no exponential expressions, and no comma/period separators
3. Delete the following files: idealOrders.txt, GRUModel1.keras, ..., GRUModel33.keras, GDPModel.keras
4. Run the ARIMA.py file, all missing files will be automatically regenerated
5. Repeat step 4 approximately 15-20 times (each time will begin with a different random set of initial parameters) and choose the one with best performance for the testing year (in this case, 2024). [This can be addressed by fixing the hyper-hyper-parameters]
6. The model has been successfully trained