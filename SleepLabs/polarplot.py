import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('./Data/Pilot/Movement_Orientation_2.csv')

fig = plt.figure(figsize=(7,7))

ax = plt.subplot(111, projection='polar')

# Extract hour and minute information from Timestamp column
hours = pd.to_datetime(data['Timestamp']).dt.hour
minutes = pd.to_datetime(data['Timestamp']).dt.minute

# Calculate angle for each hour-minute value
angles = ( (hours * 60 + minutes)/720*np.pi )# - (np.pi/2)

# Plot bars with correct angle values
ax.bar(angles, data['Movement'], width=0.02, alpha=0.3, color='red', label='Orientation')

# Make the labels go clockwise
ax.set_theta_direction(-1)

# Place zero at top
ax.set_theta_offset(np.pi/2)

# Set the circumference ticks
ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))

# Set the label names
ticks = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM',
         '8 AM', '9 AM', '10 AM', '11 AM', '12 PM', '1 PM', '2 PM', '3 PM',
         '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM']
ax.set_xticklabels(ticks)

# Suppress the radial labels
plt.setp(ax.get_yticklabels(), visible=False)

# Bars to the wall
plt.ylim(0, 2)

plt.legend(bbox_to_anchor=(1, 0), fancybox=True, shadow=True)
plt.savefig('./orientation.png')