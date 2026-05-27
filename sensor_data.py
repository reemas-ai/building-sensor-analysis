import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# seed(42) ensures same random data every run — important for reproducibility
np.random.seed(42)

def generate_realistic_sensor_data(days=30):# Generate synthetic sensor data — one reading per hour for 'days' days 
    start_time = datetime(2026, 5, 16)
    num_readings = days * 24
    timestamps = [start_time + timedelta(hours=i) 
                  for i in range(num_readings)]
    
    
    hours = np.array([t.hour for t in timestamps])
    # Diurnal pattern: temperature follows a sine wave — peaks at noon, lowest at dawn
    temp_pattern = 22 + 6 * np.sin((hours - 6) * np.pi / 12)
    temperature = temp_pattern + np.random.normal(0, 1, num_readings)
    
    # Humidity inversely correlated with temperature — realistic building behavior
    humidity = 70 - 0.8 * (temperature - 22) + np.random.normal(0, 2, num_readings)
    
   
    vibration = np.random.normal(0.02, 0.002, num_readings)
    pressure = np.random.normal(101, 1, num_readings)
    
    data = {
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'vibration': vibration,
        'pressure': pressure,
        'is_anomaly': 0
          }
    
    df = pd.DataFrame(data)# make a data frame 
    
    # Inject 3 known anomalies for model testing — we know exactly where they are
    anomaly_indices = [150, 300, 450] 
    for idx in anomaly_indices:
        df.loc[idx, 'temperature'] = 45
        df.loc[idx, 'humidity'] = 90
        df.loc[idx, 'vibration'] = 0.1
        df.loc[idx, 'is_anomaly'] = 1
    
    return df

df = generate_realistic_sensor_data()
df.to_csv('sensor_data.csv', index=False)
print(f"Done:{len(df)}")
print(f"Anomalies: {df['is_anomaly'].sum()}")
print(df.head())