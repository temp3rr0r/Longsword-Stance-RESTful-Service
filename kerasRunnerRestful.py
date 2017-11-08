from __future__ import print_function
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, abort, reqparse
import numpy as np
from keras.models import model_from_json
import timeit
import json

# Flask
app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def post(self):

	json_data = request.get_json(force=True)

	# Load model
	json_file = open('models/bidirectionalClassLstmLongswordModel.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("models/bidirectionalClassLstmLongswordModelWeights.h5")

	# Input data
	x_test = np.array([json_data['row']])

	# Predict
	start_time = timeit.default_timer()
	prediction = loaded_model.predict(x_test)
	elapsed = timeit.default_timer() - start_time
	predictionArgMax = np.argmax(prediction, axis=1)
        return { 'predictedClass': predictionArgMax[0], 'confidence': float(prediction[0, predictionArgMax[0]]), 'elapsedMilliseconds': elapsed * 1000 }

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
     app.run(host='0.0.0.0')
