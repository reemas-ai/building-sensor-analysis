import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest


df = pd.read_csv('sensor_data.csv')
features = ['temperature', 'humidity', 'vibration', 'pressure']
x = df[features]

model = IsolationForest(contamination=0.004, random_state=42)
model.fit(x)


app = FastAPI()


class SensorReading(BaseModel):
    temperature: float
    humidity: float
    vibration: float
    pressure: float


@app.post("/predict")
def predict(reading: SensorReading):
    data = [[reading.temperature, 
             reading.humidity, 
             reading.vibration, 
             reading.pressure]]
    
    result = model.predict(data)
    status = "Normal" if result[0] == 1 else "Anomaly"
    
    return {"status": status, "reading": reading}