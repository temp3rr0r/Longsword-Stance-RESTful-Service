from __future__ import print_function
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, abort, reqparse
import numpy as np
from keras.models import model_from_json
import timeit
import json
import socket
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Flask
app = Flask(__name__)
api = Api(app)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # Only basic logging

class KerasModel: # Init and load the LSTM model
    def __init__(self):
	#self.loaded_model = model_from_json('{"class_name": "Sequential", "config": [{"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": [null, 12], "dtype": "float32", "input_dim": 35537, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 12}}, {"class_name": "Bidirectional", "config": {"name": "bidirectional_1", "trainable": true, "layer": {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "implementation": 0, "units": 64, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, "merge_mode": "concat"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "rate": 0.5}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "keras_version": "2.0.8", "backend": "tensorflow"}')
	self.loaded_model = model_from_json('{"class_name": "Sequential", "config": [{"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": [null, 34], "dtype": "float32", "input_dim": 65536, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 34}}, {"class_name": "Bidirectional", "config": {"name": "bidirectional_1", "trainable": true, "layer": {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "implementation": 0, "units": 64, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, "merge_mode": "concat"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "rate": 0.5}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "keras_version": "2.0.8", "backend": "tensorflow"}')

	#self.loaded_model.load_weights("models/bidirectionalClassLstmLongswordModelWeights.h5")
	self.loaded_model.load_weights("models/bidirectionalRetrainingLstmLongswordModelWeights.h5")
 
    def predict(self, data):
	return self.loaded_model.predict(data)

kerasModel = KerasModel()

# Setup socket connection
TCP_IP = '127.0.0.1'
TCP_PORT = 5001
BUFFER_SIZE = 1024
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

verbose = True

yExpected = []
yPredicted = []
confusionMatrix = []
classAccuracy = []

class Predict(Resource):
    def post(self):
	start_time = timeit.default_timer() # TODO: temp
	json_data = request.get_json(force=True)
	
	# Input data
	x_test = np.array([json_data['row']])
	y_expected = json_data['classification']
	r = x_test
	r2 = np.copy(r)
	r[r < 0] = 0
	r2[r2 > 0] = 0
	r2 *= -1
	r = np.insert(r, 0, values=r2[:,0], axis=1)        
	r = np.insert(r, 1, values=r2[:,1], axis=1)
	r = np.insert(r, 2, values=r2[:,2], axis=1)
	r = np.insert(r, 3, values=r2[:,3], axis=1)
	r = np.insert(r, 4, values=r2[:,4], axis=1)
	r = np.insert(r, 5, values=r2[:,5], axis=1)
	x_test = r	

	# Predict	
	prediction  = kerasModel.predict(x_test)
	predictionArgMax = np.argmax(prediction, axis=1)        
      	json_data['predictedClass']= predictionArgMax[0]
	json_data['confidence'] = float(prediction[0, predictionArgMax[0]])
	elapsed = timeit.default_timer() - start_time # TODO: temp
	response =  { 'predictedClass': predictionArgMax[0], 'confidence': float(prediction[0, predictionArgMax[0]]), 'elapsedMilliseconds': elapsed * 1000 }
	if verbose == True:
		print ("x_test: ", x_test)
		print (response)	
		print(json_data)	
	
	# TODO: calc confusion matrix
	yExpected.append(y_expected)
	yPredicted.append(predictionArgMax[0])
	if (len(yPredicted) > 0):
		confusionMatrix = confusion_matrix(yExpected, yPredicted)
		rowSums = np.sum(confusionMatrix, axis = 1)
		diagonal = confusionMatrix.diagonal(0)
		
		classAccuracyOut = [0.0] * 7
		f1ScoresOut = [0.0] * 7
		
		f1Scores = f1_score(yExpected, yPredicted, average = None)
		for i in range(len(f1Scores)):
			f1ScoresOut[i] = f1ScoresOut[i] + f1Scores[i]
		for i in range(len(rowSums)):
			if rowSums[i] != 0:
				classAccuracy = float(float(diagonal[i]) / float(rowSums[i]))
				classAccuracyOut[i] = classAccuracyOut[i] + classAccuracy

		if verbose == True:
			print("yExpected: " + str(yExpected))
			print("yPredicted: " + str(yPredicted))
			print("f1Scores: " + str(f1Scores))
			print("classAccuracy: " + str(classAccuracyOut))
			print("rowSums: " + str(rowSums))
			print("diagonal: " + str(diagonal))

		json_data['classAccuracy'] = classAccuracyOut
		json_data['f1Scores'] = f1ScoresOut
		json_data['accuracyScore'] = accuracy_score(yExpected, yPredicted)

	s.send(json.dumps(json_data)) # Send socket response	
	return response # Send response to lambda # TODO: disable response for speed?

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
     app.run(host='0.0.0.0')

s.close() # Close socket connection
