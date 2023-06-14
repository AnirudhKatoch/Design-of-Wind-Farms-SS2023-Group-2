import pandas as pd
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from floris.tools import WindRose, FlorisInterface


data = pd.read_csv("nc_files/mesotimeseries_Bayern_DE_100m/mesotimeseries_Bayern_DE_100m.csv")
print("The wind rose dataframe looks as follows: \n\n {} \n".format(data))

#Convert csv columns into arrays
ws = data['ws'].tolist()
wd = data['wd'].tolist()

#Adapt the wind speed to the hub-height
alpha = 0.20
Hub_Height = 120
ws = np.multiply(ws,(Hub_Height/100)**alpha)

#Plot the wind rose base on wind speed and wind direction
wr = WindRose()
data=wr.make_wind_rose_from_user_data(wd, ws)
print("The wind rose dataframe looks as follows: \n\n {} \n".format(data))

wr.plot_wind_rose()
plt.show()

# data.to_csv("output_data.csv", index=False)
# print("CSV file created successfully.")
