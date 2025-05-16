import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Read JSON data from file
with open('data.json', 'r') as f:
    data = json.load(f)

# Extract dates and values
dates = [datetime.strptime(item['Date'], "%Y-%m-%d") for item in data]
# DWSI = [item['DWSI'] for item in data]
# GNDVI = [item['GNDVI'] for item in data]
NDVI = [item['NDVI'] for item in data]
# RSVI = [item['RSVI'] for item in data]
# NPCI = [item['NPCI'] for item in data]


# Create a plot
plt.figure(figsize=(15, 8))

# Plot the data
# plt.plot(dates, DWSI, label='DWSI', marker='o')
# plt.plot(dates, GNDVI, label='GNDVI', marker='o')
plt.plot(dates, NDVI, label='NDVI', marker='o')
# plt.plot(dates, RSVI, label='RSVI', marker='o')
# plt.plot(dates, NPCI, label='NPCI', marker='o')

# Format the x-axis to show each date under the point
plt.xticks(ticks=dates, labels=[d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Growth Indicators Over Time')
plt.legend()
plt.yticks([i * 0.1 for i in range(-1, 18)])

# Show the plot
plt.tight_layout()
plt.show()