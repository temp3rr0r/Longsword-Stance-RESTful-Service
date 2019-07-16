# Longsword Stance RESTful Service #

Working demo: https://www.youtube.com/watch?v=v7hvOyPQ0EM

Longsword Stance RESTful Service: Invokes prediction results with real-time multivariate time series data. Using Flask and python, the pre-trained bidirectional LSTM deep learning model is loaded to memory. RESTful post request containing real-time rows of IMU data can be used to classify the longsword movement stance. Information on the classification confidence and execution time in milliseconds is also provided.

## Technologies
- Deep Learning
- Bidirectional Long-Short Term Memory (LSTM)

## Hardware
- Nvidia Jetson TX2
- Amazon EC2 t2.small

## SDKs & Libraries
- keras
- Flask
- numpy
- h5py
- csv, json
- socket
