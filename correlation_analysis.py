import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('sensor_data.csv')

all_re=df[['temperature', 'humidity', 'vibration', 'pressure']].corr()
print(all_re)

plt.figure(figsize=(10, 8))
sns.heatmap(all_re, 
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            center=0)
plt.title('Correlation Matrix - Building Sensors')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
