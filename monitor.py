# Monitor data drift — detects if sensor behavior changes over time
import pandas as pd 

df=pd.read_csv('sensor_data.csv')# Load latest sensor readings

#To check if the data has changed from the previous seven days
threshold = df['temperature'].iloc[:168].std() * 0.5 #The amount of change allowed
baseline=df['temperature'].iloc[:168].mean()
current =df['temperature'].iloc[168:].mean()

print(f"threshold  :{threshold}")
print(f"Baseline : {baseline}")
print(f'Current : {current}')

# If the current mean temperature deviates from the baseline by more than the threshold, we flag it as data drift
if(abs(current-baseline)>=threshold):
    print('Data Drift Detected')
else:
    print('Data Normal')