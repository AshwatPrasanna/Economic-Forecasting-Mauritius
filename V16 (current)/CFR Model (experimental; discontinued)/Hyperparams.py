
val_cutoff_index = 20 # Everything including and beyond this is considered validation data
input_chunk_len = 3
output_chunk_len = 2
TCNEpochs = 50
dropout = 0.25
dilation_base = 2
kernel_size = 2
num_filters = 3
forecast_time_step = 10 # Must be larger than the length of both training and validation data; consider MODIFYing this later to be immediate (time_step = 1)
ann_weightage = 0.2

# NOTE TO SELF: Remember to check the training-data-only prediction later
