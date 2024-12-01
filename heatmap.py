# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("type1.csv")

# Select sensor columns and status column
sensor_columns = [col for col in data.columns if 'sensor' in col]
status_column = 'status'

# Compute the correlation matrix for sensors and status
#corr_matrix = data[sensor_columns + [status_column]].corr()

# Compute the correlation matrix for sensor columns only
sensor_corr_matrix = data[sensor_columns].corr()

# Extract only the correlations between sensors and status
#corr_with_status = corr_matrix[[status_column]].drop(status_column)

#corr_matrix = data[row + ['status']].corr()

plt.figure(figsize = (10,10))
sns.heatmap(sensor_corr_matrix, 
            fmt=".2f",
            vmin=-1, 
            vmax=1)

#plt.title("Correlation of Sensors with Status")
plt.ylabel("Sensors")
plt.xlabel("Sensors")
plt.show()
