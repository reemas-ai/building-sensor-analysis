import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df=pd.read_csv('sensor_data.csv')# Load the dataset

# We will use the IQR method to detect anomalies in the temperature readings

Q1=df['temperature'].quantile(0.25)
Q3=df['temperature'].quantile(0.75)
IQR=Q3-Q1

# IQR method: flags values beyond 1.5x the interquartile range as anomalies

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f'Q1--> {Q1:0.3f}')
print(f'Q3--> {Q3:0.3f}')
print(f'IQR--> {IQR:0.3f}')
print(f"Lower Bound--> {lower:.3f}")
print(f"Upper Bound--> {upper:.3f}")

# Identify anomalies based on the calculated bounds
anomalies = df[(df['temperature'] < lower) | 
               (df['temperature'] > upper)]
print(f"anomalies--> {len(anomalies)}")
print(anomalies[['timestamp', 'temperature']])

# Visualize the anomalies

plt.figure(figsize=(14,6))
plt.scatter(df['timestamp'],df['temperature'],color='blue',alpha=0.5,label='Normal')
plt.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', s=100,label='anomalies',zorder=5)
plt.axhline(y=upper, color='orange', linestyle='--', label='Upper Bound')
plt.axhline(y=lower, color='green', linestyle='--', label='Lower Bound')
plt.title('Detection of temperature anomalies')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('anomaly_plot.png')
plt.show()