import torch
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import MinMaxScaler

num_nodes = 10

train_rate = 0.5
seq_len = 20
output_dim = pre_len = 10
batch_size = 32
lr = 0.001
training_epoch = 1001
validation_rate = 0.1
l2_coeff = 0.0001


#for ILI Region data
adj_pearson = pd.read_csv('input_data/Region_adj_geo1.csv', header=None)
adj = np.mat(adj_pearson)

#region features(data)
data = pd.read_csv('input_data/ILI_region_feature_1997-2020.csv')

scaler = MinMaxScaler()

time_len =data.shape[0]