# -*- coding: utf-8 -*-

import numpy as np
import pickle 


loaded_model = pickle.load(open('/Users/GraceTee/trained_cancer_model.sav', 'rb'))


input_data = [21.56, 22.39, 142.0, 1479.0, 0.111,
 0.1159, 0.2439, 0.1389, 0.1726, 0.05623,
 1.176, 1.256, 7.673, 158.7, 0.0103,
 0.02891, 0.05198, 0.02454, 0.01114, 0.004239,
 25.45, 26.4, 166.1, 2027.0, 0.141,
 0.2113, 0.4107, 0.2216, 0.206, 0.07115]
 
    
data_numpy_array = np.asarray(input_data)
data_reshaped = data_numpy_array.reshape(1, -1)
prediction = loaded_model.predict(data_reshaped)

if (prediction[0] == 'B'):
    print("The tumor is benign")
else:
    print("The tumor is malignant")
