import numpy as np
from sklearn import preprocessing

# Sample data
input_data = np.array([[5.1,-2.9,3.3],
                       [-1.2,7.8,-6.1],
                       [3.9,0.4,2.1],
                       [7.3,-9.9,-4.5]])

# Binarization
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("Binarized data :\n",data_binarized)

# Mean removal
print("Before : ")
print("Mean = ",input_data.mean(axis=0))
print("Std deviation = ",input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("After : ")
print("Mean = ",data_scaled.mean(axis=0))
print("Std deviation = ",data_scaled.std(axis=0))

# Min Max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaler_minmax = data_scaler_minmax.fit_transform(input_data)
print("Min Max Scaled data :\n",data_scaler_minmax)

# Normalization
data_normalized_L1 = preprocessing.normalize(input_data,norm='l1')
data_normalized_L2 = preprocessing.normalize(input_data,norm='l2')
print("L1 normalized data :\n",data_normalized_L1)
print("L2 normalized data :\n",data_normalized_L2)
