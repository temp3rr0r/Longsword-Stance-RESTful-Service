from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def put(self):
        dataRow = request.form['data']
        return { 'predictedClass': 2, 'accuracy': 0.5, 'data': dataRow }

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
