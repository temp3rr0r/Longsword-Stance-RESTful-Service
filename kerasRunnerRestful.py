
'''Train a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''
from __future__ import print_function
from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
from numpy import array
import csv

# import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


r = np.genfromtxt("longsword.csv", delimiter=',')



# TODO: add columns. If positive: 0, 32500. If negative 32500, 0. Essentially slit it into to classes for possives and negatives
r2 = np.copy(r)
r[r < 0] = 0
r2[r2 > 0] = 0
r2 *= -1
r = np.insert(r, 1, values=r2[:,1], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 2, values=r2[:,2], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 3, values=r2[:,3], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 4, values=r2[:,4], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 5, values=r2[:,5], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 6, values=r2[:,6], axis=1) # inser2t values befor2e column 3

#np.random.shuffle(r)
proportion15Percent = int(0.15 * r.shape[0])
x_validate = r[0:proportion15Percent,1:5]
y_validate = r[0:proportion15Percent,0]
x_test= r[proportion15Percent + 1:2*proportion15Percent,1:5]
y_test = r[proportion15Percent + 1:2*proportion15Percent,0]
x_train = r[2*proportion15Percent + 1:len(r),1:5]
y_train = r[2*proportion15Percent + 1:len(r),0]


print(len(x_test), 'test sequences (15%)')

print('Pad sequences (samples x time)')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)


##########

from keras.models import model_from_json
# load json and create model
json_file = open('models/bidirectionalClassLstmLongswordModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('models/bidirectionalClassLstmLongswordModelWeights.h5')
print("Loaded model from disk")

# Prediction
print('Prediction')
import timeit
prediction = loaded_model.predict(x_test)
print('Prediction Arg max')
predictionArgMax = np.argmax(prediction, axis=1)
print('Expected')
print ('Classification Accuracy: ', ((predictionArgMax == y_test).sum() * 100.0) / (1.0 * len(y_test)), '%')

app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def put(self):
        dataRow = request.form['data']
	dataRow = r[1:2,1:5]
	#dataRow = np.array([  0,  868,   0,  249])	
	print('DataRow: ', dataRow)
	start_time = timeit.default_timer()
	prediction = loaded_model.predict(dataRow)
	elapsed = timeit.default_timer() - start_time
	print('Prediction: ', prediction)
	predictionArgMax = np.argmax(prediction, axis=1)
	print('Prediction arg max: ', predictionArgMax)	
        return { 'predictedClass': predictionArgMax[0], 'accuracy': str(prediction[0, predictionArgMax[0]]), 'elapsed': elapsed * 1000 }

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
