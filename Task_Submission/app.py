# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:32:24 2021

@author: Pradeep Jajjara
"""


import uvicorn
from fastapi import FastAPI
import numpy as np
from encode_muril import encode
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("random_forest.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def Hello(name: str):
    return {'Hello': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted sentiment with the confidence
@app.post('/predict')
def predict_sentiment(data: str):
    encoded_str = encode([data])
    final = np.array(encoded_str)
    prediction = classifier.predict(final)
    if prediction == 1:
        pred = 'positive sentiment'
    else:
        pred = 'negative sentiment'
    return {
        'prediction': pred 
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload