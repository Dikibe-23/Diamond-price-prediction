# Import the required packages
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# load the trained model
trained_model = 'rf_model.pkl'
model = pickle.load(open(trained_model, 'rb'))

# Create a class 'diamondpred' that defines data types expected from the user.

class dprice_pred(BaseModel):
    Carat: float
    Cut: int
    Color: int
    Clarity: int
    Depth: float
    Table: float
    X: float
    Y: float
    Z: float
    

@app.get('/')
@app.get('/home')

def read_root():
    return {'message': 'Diamond price prediction App'}
    
# Define the function, which will make the prediction using the input data provided by the user.

@app.post("/predict")
def predict_price(diamond_details:dprice_pred):
    data = diamond_details.dict()
    
    carat = data['Carat']
    cut = data['Cut']
    color = data['Color']
    clarity = data['Clarity']
    depth = data['Depth']
    table = data['Table']
    x = data['X']
    y = data['Y']
    z = data['Z']
    
    # Make prediction
    prediction = model.predict([[carat, cut, color, clarity, depth, table, x, y, z]])
    
    return {"prediction":prediction[0]}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5000)
