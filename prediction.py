import pandas as pd
import numpy as np
from keras.layers import Input, Dense 
from keras.models import Model 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy import stats
from anomaly_detection import CurveConstructor

# Basic prediction script for the CurveConstructor class.

# Define a set of event periods where the data is defined to be anomalous (either going over or under the typical range),
# And provide a period where the event is expected within the future to recieve a value to add or subtract from normal predictions.

data = pd.read_csv('yourdata.csv')
data = pd.DataFrame(data[["Date","Volume"]])
data.columns = ['ds','y']

# Where the time series shows a major anomalous spike
events_max = [
    ['2022-10-10 01:00:00','2022-10-30 06:00:00'],
    ['2023-08-20 12:00:00','2023-10-09 13:00:00']
]

# As above for dips
events_min = [
    ['2023-01-24 01:00:00','2023-03-25 01:00:00']
]

# Timeframes to predict (where product launches are suspected to influence values)
deltas = [
    ['2024-01-01 00:00:00','2024-01-01 12:00:00'],
    ['2024-02-01 00:00:00','2024-02-01 21:00:00']
]

c = CurveConstructor(events_max,events_min)

# Fit data
c.fit(data)

# Plot each type of threshold
c.plot_thresholds(thresh_type='max')
c.plot_thresholds(thresh_type='min')

# For each item, produce a DF of deltas to append to the normal series predictions
for d in deltas:
    out = c.predict(d,thresh_type='max')