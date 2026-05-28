import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('sensor_data.csv')# Load the dataset

all_re=df[['temperature', 'humidity', 'vibration', 'pressure']].corr()# Calculate the correlation matrix for the sensor features to understand how they relate to each other
print(all_re)

# Visualize the correlation matrix using a heatmap to easily identify strong correlations between features

plt.figure(figsize=(10, 8))# Plot size: wide enough to show all sensor labels clearly
sns.heatmap(all_re, 
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            center=0)
plt.title('Correlation Matrix - Building Sensors')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
