import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('sensor_data.csv') # Load the dataset

features = ['temperature', 'humidity', 'vibration', 'pressure']
x = df[features] # we need numeric data only 


model = IsolationForest(contamination=0.004, random_state=42)# contamination=0.004 means we expect ~3 anomalies in 720 readings
                                                             # random_state=42 ensures reproducible results
model.fit(x)

# start prediction
df['predicted'] = model.predict(x)
df['predicted'] = df['predicted'].map({1: 'Normal', -1: 'Anomaly'})# we know 1 mean normal and -1 mean anomaly


print("Predicted Anomalies:") #print the prediction  
print(df['predicted'].value_counts())

print("Real_anomalies:")
real_anomalies = df[df['is_anomaly'] == 1]# extract real anomalies from our label to see if it predict correct 
print(real_anomalies[['timestamp', 'temperature', 'humidity', 'predicted']])
print("\nClassification Report:")
print(classification_report(df['is_anomaly'], 
                            df['predicted'].map({'Normal': 0, 'Anomaly': 1})))



df['timestamp'] = pd.to_datetime(df['timestamp'])# make it in time and date format 

# take the results to make a plot for our analysis
normal = df[df['predicted'] == 'Normal']
anomaly = df[df['predicted'] == 'Anomaly']
real = df[df['is_anomaly'] == 1]


plt.figure(figsize=(14, 5))
plt.scatter(normal['timestamp'], normal['temperature'], 
            color='blue', alpha=0.3, s=10, label='Normal')
plt.scatter(anomaly['timestamp'], anomaly['temperature'], 
            color='orange', alpha=0.7, s=200, label='Predicted Anomaly')
plt.scatter(real['timestamp'], real['temperature'], 
            color='red', s=100, marker='X', label='Real Anomaly', zorder=5)

plt.title('Building Sensor - Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('anomaly_detection_plot.png')
plt.show()
