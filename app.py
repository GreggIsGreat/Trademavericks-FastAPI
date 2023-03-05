import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from price_info import NAS100,US30,GER30

app = FastAPI()
pkl_in = open("nas100.pkl", 'rb')
nas100model=pickle.load(pkl_in)

us30pkl = open("us30.pkl", 'rb')
us30model=pickle.load(us30pkl)

ger30pkl = open("ger40.pkl", 'rb')
ger30model=pickle.load(ger30pkl)

@app.get('/')
def index():
    return {'message': 'Hello Thabang'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Weclome back Sir,{name}'}

# NAS100 model api
@app.post('/nas100')
def pred_price(data:NAS100):
    data = data.dict()
    open=data['open']
    volume=data['volume']
    low=data['low']
    high=data['high']
    inpt_val = (open,volume,low,high)
    input_dt = np.asarray(inpt_val)
    reshape_vals = input_dt.reshape(1, -1)
    anticipated = nas100model.predict(reshape_vals)
    if (inpt_val[0] < anticipated):
        anticipated = f"Buy!! & TP:{anticipated.round(2)}"
    else:
        anticipated = f"Sell!! & TP:{anticipated.round(2)}"
    return {"prediction": anticipated}
    
@app.get('/prediction_nas100')
def get_cat(open: float, volume: float, low: float, high: float):
    anticipated = nas100model.predict([[open, volume, low, high]])
    return {'prediction': anticipated}

# US30 model api
@app.post('/us30')
def pred_price(data:US30):
    data = data.dict()
    open=data['open']
    volume=data['volume']
    low=data['low']
    high=data['high']
    inpt_val = (open,volume,low,high)
    input_dt = np.asarray(inpt_val)
    reshape_vals = input_dt.reshape(1, -1)
    anticipated = us30model.predict(reshape_vals)
    if (inpt_val[0] < anticipated):
        anticipated = f"Buy!! & TP:{anticipated.round(2)}"
    else:
        anticipated = f"Sell!! & TP:{anticipated.round(2)}"
    return {"prediction": anticipated}
    
@app.get('/prediction_us30')
def get_cat(open: float, volume: float, low: float, high: float):
    anticipated = us30model.predict([[open, volume, low, high]])
    return {'prediction': anticipated}

# Ger30 model api
@app.post('/ger30')
def pred_price(data:GER30):
    data = data.dict()
    open=data['open']
    volume=data['volume']
    low=data['low']
    high=data['high']
    inpt_val = (open,volume,low,high)
    input_dt = np.asarray(inpt_val)
    reshape_vals = input_dt.reshape(1, -1)
    anticipated = ger30model.predict(reshape_vals)
    if (inpt_val[0] < anticipated):
        anticipated = f"Buy!! & TP:{anticipated.round(2)}"
    else:
        anticipated = f"Sell!! & TP:{anticipated.round(2)}"
    return {"prediction": anticipated}
    
@app.get('/prediction_ger30')
def get_cat(open: float, volume: float, low: float, high: float):
    anticipated = ger30model.predict([[open, volume, low, high]])
    return {'prediction': anticipated}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7000)
