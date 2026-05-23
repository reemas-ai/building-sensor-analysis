import pandas as pd 
import numpy as np

df=pd.read_csv('sensor_data.csv')
df.loc[620:, 'temperature'] = df.loc[620:, 'temperature'] + 12
threshold = df['temperature'].iloc[:168].std() * 0.5
baseline=df['temperature'].iloc[:168].mean()
current =df['temperature'].iloc[168:].mean()

print(f"threshold  :{threshold}")
print(f"Baseline : {baseline}")
print(f'Current : {current}')

if(abs(current-baseline)>=threshold):
    print('Data Drift Detected')
else:
    print('Data Normal')