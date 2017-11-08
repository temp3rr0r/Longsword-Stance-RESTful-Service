from __future__ import print_function
import json
import requests
print('Loading function')

#def handler(event, context):

    #print("Received event: " + json.dumps(event, indent=2))

    #curl http://52.208.37.243:5000/predict -d 'data={ "row": [ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27] }' -X PUT
url = 'http://52.208.37.243:5000/predict'
headers = {'content-type': 'application/json'}
#data = 'data={ "row": [ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27]}'
#data= '{ "row": [ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27]}'


    # TODO: live edit data with lambda (increase rows, remove negative nums etc...)
#response = requests.post(url, data={'row': [ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27]})
response = requests.post(url, headers= headers, data= {"row":[ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27] })


#response = requests.post(url, json={"user": user,"pass": password})
print(response, response.text)
    #print('ok1')

    #for record in event['Records']:
    #    print(record['eventID'])
        #print(record['eventName']) print("DynamoDB Record: " + json.dumps(record['dynamodb'], indent=2))
#return response
    #return 'Successfully processed {} records.'.format(len(event['Records']))

