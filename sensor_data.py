import pandas as pd 
import numpy as np
from datetime import datetime, timedelta 
def generate_sensor_data(days=30):
    start_time= datetime(2024,1,1)
    num_readings= days * 24
    timestamps = [start_time + timedelta(hours=i) 
                  for i in range(num_readings)]
    np.random.seed(42)
    data = {
        'timestamp': timestamps,
        'temperature': np.random.normal(22, 3, num_readings),
        'humidity': np.random.normal(55, 10, num_readings),
        'vibration': np.random.normal(0.02, 0.005, num_readings),
        'pressure': np.random.normal(101, 2, num_readings)
    }
    
    return pd.DataFrame(data)
df = generate_sensor_data()
df.to_csv('sensor_data.csv', index=False)

print(f" Done {len(df)} ")
print(df.head())

