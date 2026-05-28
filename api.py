from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load('model.pkl')#load the model we saved in anomaly_model.py to use it in our API

app = FastAPI()# create a FastAPI 


class SensorReading(BaseModel):# define the structure of the input data for our API
    temperature: float
    humidity: float
    vibration: float
    pressure: float

# endpoint that accepts sensor readings and returns prediction
@app.post("/predict")
def predict(reading: SensorReading):
    data = [[reading.temperature, 
             reading.humidity, 
             reading.vibration, 
             reading.pressure]]
    
    result = model.predict(data)
    status = "Normal" if result[0] == 1 else "Anomaly"
    
    return {"status": status, "reading": reading} # return prediction with original reading