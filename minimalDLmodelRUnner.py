from __future__ import print_function
import h5py
import numpy as np
from keras.models import model_from_json
from numpy import array

json_file = open('models/bidirectionalClassLstmLongswordModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/bidirectionalClassLstmLongswordModelWeights.h5")

x_test = np.array([[ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27]])
prediction = loaded_model.predict(x_test)
predictionArgMax = np.argmax(prediction, axis=1)
print('Class: ', predictionArgMax[0])
print('Accuracy: ', prediction[0, predictionArgMax[0]])
