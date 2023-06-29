# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import pandas as pd
import numpy_financial as npf
import matplotlib.pyplot as plt

from scipy.interpolate import NearestNDInterpolator

from floris.tools import FlorisInterface, WindRose


print("===================== This is a new run of the program ==============================")
#Define the turbine/site parameters hardcoded  
#============================ Turbine ========================================
#=============================================================================
alpha = 0.17
Hub_Height = 120
rated_power = 3250


#Import wind speed and direction and plots the Windrose for the given site
#============================ Bayern-DE 100m =================================
#=============================================================================
data = pd.read_csv("inputs/final_proyect/mesotimeseries_Kastrup-DK10m.csv")
#print("The wind rose dataframe looks as follows: \n\n {} \n".format(data))



#=============================================================================
#Convert csv columns into arrays 
ws = data['ws'].tolist()
wd = data['wd'].tolist()

#Correct the wind speed to the hub-height using the power law

ws = np.multiply(ws,(Hub_Height/10)**alpha)

#Plot the wind rose base on wind speed and wind direction
wind_rose = WindRose()
data=wind_rose.make_wind_rose_from_user_data(wd, ws)
#print("The wind rose dataframe looks as follows: \n\n {} \n".format(data))

# Show the wind rose
wind_rose.plot_wind_rose()

#=============================================================================

df_wr = data
# Derive the wind directions and speeds we need to evaluate in FLORIS
wd_array = np.array(df_wr["wd"].unique(), dtype=float)
ws_array = np.array(df_wr["ws"].unique(), dtype=float)

# Format the frequency array into the conventional FLORIS v3 format, which is
# an np.array with shape (n_wind_directions, n_wind_speeds). To avoid having
# to manually derive how the variables are sorted and how to reshape the
# one-dimensional frequency array, we use a nearest neighbor interpolant. This
# ensures the frequency values are mapped appropriately to the new 2D array.
wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
freq_interp = NearestNDInterpolator(df_wr[["wd", "ws"]], df_wr["freq_val"])
freq = freq_interp(wd_grid, ws_grid)

# Normalize the frequency array to sum to exactly 1.0
freq = freq / np.sum(freq)

# Load the FLORIS object
fi = FlorisInterface("inputs/gch.yaml") # GCH model
# fi = FlorisInterface("inputs/cc.yaml") # CumulativeCurl model


coordinates= [('3559.77', '3506.80'), ('5917.94', '1598.74'), ('5172.71', '2591.54'), ('7386.77', '2262.26'), ('2349.76', '2992.72'), ('1723.28', '771.94'), ('3882.25', '1614.79'), ('1336.05', '2037.73'), ('5307.23', '400.16'), ('2992.13', '128.10')]
x_coordinates = []
y_coordinates = []

for coordinate in coordinates:
    # Remove parentheses and split the string by comma
    #x, y = coordinate.strip('()').split(', ')
    x, y = coordinate
    # Convert x and y coordinates to float and append to respective arrays
    x_coordinates.append(float(x))
    y_coordinates.append(float(y))



# Assume a three-turbine wind farm with 5D spacing. We reinitialize the
# floris object and assign the layout, wind speed and wind direction arrays.
D = fi.floris.farm.rotor_diameters[0] # Rotor diameter for the NREL 5 MW
fi.reinitialize(
    layout_x = x_coordinates,
    layout_y = y_coordinates,
    wind_directions=wd_array,
    wind_speeds=ws_array,
)

print("----------------------------- AEP Calculations ----------------------")
print ()
# Compute the AEP using the default settings
aep = fi.get_farm_AEP(freq=freq)
print("Farm AEP (default options): {:.3f} GWh".format(aep / 1.0e9))



# Compute the AEP again while specifying a cut-in and cut-out wind speed.
# The wake calculations are skipped for any wind speed below respectively
# above the cut-in and cut-out wind speed. This can speed up computation and
# prevent unexpected behavior for zero/negative and very high wind speeds.
# In this example, the results should not change between this and the default
# call to 'get_farm_AEP()'.
aep = fi.get_farm_AEP(
    freq=freq,
    cut_in_wind_speed=4.0,  # Wakes are not evaluated below this wind speed
    cut_out_wind_speed=25.0,  # Wakes are not evaluated above this wind speed
)
print("Farm AEP (with cut_in/out specified): {:.3f} GWh".format(aep / 1.0e9))


# Finally, we can also compute the AEP while ignoring all wake calculations.
# This can be useful to quantity the annual wake losses in the farm. Such
# calculations can be facilitated by enabling the 'no_wake' handle.
aep_no_wake = fi.get_farm_AEP(freq, no_wake=True)
print("Farm AEP (no_wake=True): {:.3f} GWh".format(aep_no_wake / 1.0e9))

# Calculates the wind farm efficiency 
wf_efficiency = aep/aep_no_wake*100
print("The wind farm efficiency (P/Pno_wake): {:.1f} %".format(wf_efficiency))


# Calculates the max theoretical windfarm energy in [GWh]
turbines = fi.floris.farm.n_turbines #Number of turbines
Y = 8760
max_energy = rated_power*Y*turbines
print("Farm max theoretical energy: {:.2f} GWh".format(max_energy / 1.0e6))

# Calculates the capacity factor in % 
capacity_factor = ((aep/1e3)/max_energy)*100
print("The wind farm capacity factor: {:.1f} %".format(capacity_factor))



print ()
print("----------------------------- LCOE Calculation ----------------------")
# Calculates the O&M Costs per year
#aep = (aep/3)*turbines
print("Farm AEP (with cut_in/out specified): {:.3f} GWh".format(aep / 1.0e9))
OM_cost = 0.012;                                  # [$/kWh - yr]
OM_cost_yr = OM_cost*aep*1e-9;                     # [Million $ - yr]

# Calculates the Renting Costs per year
Rent_cost = 20000;                                  # [$/MW-yr]
Rent_cost_yr = Rent_cost*turbines*rated_power*1e-9; # [Million $ - yr]


# Calculates the Operation Costs per year
OpEx_yr = OM_cost_yr + Rent_cost_yr                 # [Million $ - yr]
print("The yearly operational costs of the project is: {:.2f} [$/MWh - yr]".format(OpEx_yr))


# Calculates the Initial investment cost
turbine_cost = 1.6;                                                   #[Million $ / MW]
turbine_cost_total = turbine_cost*turbines*rated_power*1e-3          #[Million $]

# Input the value given by the output file from LandBosse as the total sum of the cost per project column
landbosse_cost = 25.392297                              #[Million $]

CapEx = landbosse_cost + turbine_cost_total                          #[Million $]


# Calculates the payment value for the CapEx
n = 20;
DiscountRate = 0.035
CapEx_yr = npf.pmt(DiscountRate, n, -CapEx)
print("The yearly payment value for the initial investment is: {:.2f} [$/MWh -yr]".format(CapEx_yr))

# Calculates the total cost (OpEx + CapEx)
Cost_total_yr = OpEx_yr + CapEx_yr          #[Million $ - yr]           

# LCOE Calculation
LCOE = (Cost_total_yr*1e6)/(aep*1e-6)    #[$/MWh]
print("The sLCOE of the project is: {:.2f} [$/MWh]".format(LCOE))


print ()
print("--------------- Electricity spot price year variaton  -----------------")

A = [100.5,249.8,154.3,74.3,57.7,53.6,63.1,51.2,38.9,35.9,32.6,32.6,32.6,32.6,32.6,32.6,32.6,32.6,32.6,32.6,32.6]


spot_prices = []  # List to store spot prices

for value in A:
    spot_price = value*(1.235-0.04295*(110/120)**alpha*data.ws)
    spot_price = spot_price*data.freq_val
    spot_price = spot_price.sum()

    spot_prices.append(spot_price)

# Print the spot prices
#for i, price in enumerate(spot_prices):
#   print("Year{} is: {:.2f}".format(2020+i, price))
    

print ()
print("-------------- Internal Rate of Return (IRR) and Profitability Index (PI) ------------")


SellingElec = spot_prices; #seeling electricity #[$/MWh - yr]
SellingElec_yr = [x *1e-6*aep*1e-6 for x in SellingElec]       #[Million $ - yr]
initialInvestment = -CapEx
MCash = [x - OM_cost_yr - Rent_cost_yr for x in SellingElec_yr];
cashFlows = [initialInvestment, MCash[0], MCash[1], MCash[2], MCash[3], MCash[4], MCash[5], MCash[6], MCash[7],
MCash[8], MCash[9], MCash[10], MCash[11], MCash[12], MCash[13], MCash[14], MCash[15], MCash[16], MCash[17], MCash[18],
MCash[19]];


# Calculate the IRR, NPV, PV and Profitability Index
npv = npf.npv(DiscountRate, cashFlows);
pv_future = npv +(-initialInvestment);

irr = (npf.irr(cashFlows))*100;

Profitability_Index = (pv_future)/(-initialInvestment); #greater than 1, the project generates value
print("Internal Rate of Return (IRR):%3.2f%%"%irr);
print("Profitability Index:%3.2f"%Profitability_Index);


# This plot he npv for different values of discount rates to determine and check the IRR graphically 

# discount_rates = np.arange(0.01, 0.06, 0.005)  # Array of discount rates from 1% to 10%
# npv_values = []

# for rate in discount_rates:
#     npv = npf.npv(rate, cashFlows)
#     npv_values.append(npv)

# # Create a new figure and axes
# fig, ax = plt.subplots()

# # Plot the new NPV values
# ax.plot(discount_rates*100, npv_values, label='New NPV')

# # Add a horizontal line at NPV = 0
# ax.axhline(y=0, color='red', linestyle='--', label='NPV = 0')

# ax.set_xlabel('Discount Rate [%]')
# ax.set_ylabel('Net Present Value (NPV)')
# ax.set_title('NPV Vs. Discount Rate')
# ax.grid(True)

# # Display the new plot without interfering with the older plot
# plt.show()

print()    
print(coordinates)    

print()    
print(turbines)    



