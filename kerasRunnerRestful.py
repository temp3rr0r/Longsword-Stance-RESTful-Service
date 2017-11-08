from __future__ import print_function
from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
from keras.models import model_from_json
import timeit
import json

# Flask
app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def put(self):
        dataRow = request.form['data']

	# Load model
	json_file = open('models/bidirectionalClassLstmLongswordModel.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("models/bidirectionalClassLstmLongswordModelWeights.h5")

	# Input data
	x_test = np.array([[ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27]])
	#x_test2 = json.loads(dataRow)

	#print ('xtest_2', x_test2)

	#x_test3 = x_test2["row"]

	#x_test = np.array([x_test3])
	x_test = np.array([json.loads(dataRow)["row"]])


	# Predict
	start_time = timeit.default_timer()
	prediction = loaded_model.predict(x_test)
	elapsed = timeit.default_timer() - start_time
	predictionArgMax = np.argmax(prediction, axis=1)	

        return { 'predictedClass': predictionArgMax[0], 'confidence': str(prediction[0, predictionArgMax[0]]), 'elapsedMilliseconds': elapsed * 1000 }

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
#    app.run(debug=True)
     app.run()
