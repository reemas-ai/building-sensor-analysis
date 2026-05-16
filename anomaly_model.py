import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

df = pd.read_csv('sensor_data.csv')

features = ['temperature', 'humidity', 'vibration', 'pressure']
x = df[features]


model = IsolationForest(contamination=0.004, random_state=42)
model.fit(x)


df['predicted'] = model.predict(x)
df['predicted'] = df['predicted'].map({1: 'Normal', -1: 'Anomaly'})


print("Predicted Anomalies:")
print(df['predicted'].value_counts())

print("Real_anomalies:")
real_anomalies = df[df['is_anomaly'] == 1]
print(real_anomalies[['timestamp', 'temperature', 'humidity', 'predicted']])
print("\nClassification Report:")
print(classification_report(df['is_anomaly'], 
                            df['predicted'].map({'Normal': 0, 'Anomaly': 1})))