import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from floris.tools import FlorisInterface
from floris.tools.visualization import plot_rotor_values, visualize_cut_plane

"""
This example illustrates the main parameters of the Empirical Gaussian
velocity deficit model and their effects on the wind turbine wake.
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.

# Options
show_flow_cuts = True

# Define function for visualizing wakes
def generate_wake_visualization(fi: FlorisInterface, title=None):
    # Using the FlorisInterface functions, get 2D slices.
    x_bounds = [0, 8950]
    y_bounds = [0,6260]
    z_bounds = [0, 500]
    cross_plane_locations = [0, 0, 0]
    horizontal_plane_location = 120
    streamwise_plane_location = 0.0
    # Contour plot colors
    min_ws = 4
    max_ws = 10

    horizontal_plane = fi.calculate_horizontal_plane(
        x_resolution=500,
        y_resolution=500,
        height=horizontal_plane_location,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        yaw_angles=yaw_angles
    )
    y_plane = fi.calculate_y_plane(
        x_resolution=10,
        z_resolution=10,
        crossstream_dist=streamwise_plane_location,
        x_bounds=x_bounds,
        z_bounds=z_bounds,
        yaw_angles=yaw_angles
    )
    cross_planes = []
    for cpl in cross_plane_locations:
        cross_planes.append(
            fi.calculate_cross_plane(
                y_resolution=10,
                z_resolution=10,
                downstream_dist=cpl
            )
        )

    # Create the plots
    # Cutplane settings
    cp_ls = "solid" # line style
    cp_lw = 0.5 # line width
    cp_clr = "black" # line color
    fig = plt.figure()
    fig.set_size_inches(12, 12)

    # Horizontal Profile
    fig1 = plt.figure()  # Create a new figure
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])  # Add an axes object to the figure

    # # x_1 = 3452.06
    # # y_1 = 1863.86
    # # ax1.scatter(x_1, y_1, label='T1', color='black')
    # # ax1.text(x_1 - 100, y_1, 'T1', ha='right', va='center')
    # #
    # x_2 = 3452.06
    # y_2 = 1863.86
    # ax1.scatter(x_2, y_2, label='T2', color='black')
    # ax1.text(x_2 - 100, y_2, 'T2', ha='right', va='center')
    #
    # x_3 = 3426.3
    # y_3 = 3753.17
    # ax1.scatter(x_3, y_3, label='T3', color='black')
    # ax1.text(x_3 - 100, y_3, 'T3', ha='right', va='center')
    #
    # #
    # x_4 = 2725.61
    # y_4 = 3300.48
    # ax1.scatter(x_4, y_4, label='T4', color='black')
    # ax1.text(x_4 - 100, y_4, 'T4', ha='right', va='center')
    # # #
    # x_5 = 2118.2
    # y_5 = 4924.88
    # ax1.scatter(x_5, y_5, label='T5', color='black')
    # ax1.text(x_5 - 100, y_5, 'T5', ha='right', va='center')
    # #
    # # x_6 = 3108.9
    # # y_6 = 3278.78
    # # ax1.scatter(x_6, y_6, label='T8', color='black')
    # # ax1.text(x_6 - 100, y_6, 'T8', ha='right', va='center')
    # #
    # x_7 = 487.6
    # y_7 = 4687.74
    # ax1.scatter(x_7, y_7, label='T7', color='black')
    # ax1.text(x_7 - 100, y_7, 'T7', ha='right', va='center')
    # #
    # x_8 = 2215.03
    # y_8 = 1909.13
    # ax1.scatter(x_8, y_8, label='T8', color='black')
    # ax1.text(x_8- 100, y_8, 'T8', ha='right', va='center')
    # #
    # x_9 = 2417.97
    # y_9 = 3046.31
    # ax1.scatter(x_9, y_9, label='T9', color='black')
    # ax1.text(x_9 - 100, y_9, 'T9', ha='right', va='center')
    #
    # x_10 = 1112.34
    # y_10 = 1669.99
    # ax1.scatter(x_10, y_10, label='T10', color='black')
    # ax1.text(x_10 - 100, y_10, 'T10', ha='right', va='center')
    #
    # # x_11 = 1160.63
    # # y_11 = 4425.23
    # # ax1.scatter(x_11, y_11, label='T11', color='black')
    # # ax1.text(x_11 - 100, y_11, 'T11', ha='right', va='center')
    #
    # x_12 = 1203.57
    # y_12 = 404.64
    # ax1.scatter(x_12, y_12, label='T12', color='black')
    # ax1.text(x_12 - 100, y_12, 'T12', ha='right', va='center')

    visualize_cut_plane(horizontal_plane, ax=ax1, title="Top-down profile ",
                        min_speed=min_ws, max_speed=max_ws)
    ax1.plot(x_bounds, [streamwise_plane_location] * 2, color=cp_clr,
             linewidth=cp_lw, linestyle=cp_ls)
    for cpl in cross_plane_locations:
        ax1.plot([cpl] * 2, y_bounds, color=cp_clr, linewidth=cp_lw,
                 linestyle=cp_ls)

    # Add overall figure title
    if title is not None:
        fig.suptitle(title, fontsize=16)

## Main script

# Set the coordinates for each turbine in the wind farm
coordinates = [(5935.17, 2047.67), (2901.85, 2327.77), (7669.89, 2320.64), (3962.29, 3736.55), (4732.28, 1577.41), (1259.3, 1297.94), (4636.25, 2664.75), (3108.9, 3278.78), (3773.86, 1404.18), (6638.94, 1433.36), (5372.24, 415.35), (1966.97, 403.55), (3031.08, 145.05), (6643.36, 2679.0), (5292.81, 3347.89)]

turbine_names = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15']

# Extract x and y coordinates from the list of tuples
layout_x = [coord[0] for coord in coordinates]
layout_y = [coord[1] for coord in coordinates]

# Reinitialize the FLORIS interface with the updated layout
fi = FlorisInterface("inputs/emgauss.yaml")
# D = fi.floris.farm.rotor_diameters[0]
fi.reinitialize(
    layout_x=layout_x ,
    layout_y=layout_y,
    wind_speeds=[8],
    wind_directions=[270.0]
)
yaw_angles = np.array(  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0] )  # Example yaw angles for each turbine

# # Read the CSV file
# df_1 = pd.read_csv('Optimized yaw angles/yaw_angles_Bayern_IEA_base_3MW_10.csv')
#
# # Filter rows where the first column is equal to 270
# filtered_df = df_1[df_1.iloc[:, 0] == 270]
#
# # Extract the values of the filtered rows except the first column
# result_array = filtered_df.iloc[:, 1:].values
#
# # # Convert the resulting array to one-dimensional
# yaw_angles = np.squeeze(result_array)

print(np.column_stack((turbine_names,coordinates, yaw_angles)))

# Check if the number of yaw angles matches the number of turbines
if len(yaw_angles) != len(coordinates):
    raise ValueError("Number of yaw angles should match the number of turbines")

# Reshape yaw_angles array to match the required shape (1 row, 1 column)
yaw_angles = yaw_angles.reshape(1, 1, -1)

# Calculate wake with individual yaw angles
fi.calculate_wake(yaw_angles=yaw_angles)

# Save dictionary to modify later
fi_dict = fi.floris.as_dict()

# Run wake calculation
fi.calculate_wake()

# Get the powers of each turbine
turbine_powers = fi.get_turbine_powers().flatten() / 1e6

width = 0.1
nw = -2
x = np.array(range(len(turbine_powers))) + width * nw
nw += 1

# Increase the base recovery rate
fi_dict_mod = copy.deepcopy(fi_dict)
fi_dict_mod['wake']['wake_velocity_parameters']['empirical_gauss']\
    ['wake_expansion_rates'] = [0.02, 0.01]
fi = FlorisInterface(fi_dict_mod)
fi.reinitialize(
    wind_speeds=[8],
    wind_directions=[270.0]
)

fig = plt.figure()  # Create a new figure
ax0 = fig.add_subplot(111)  # Add a subplot to the figure
title = "Increase base recovery"
ax0.bar(x, turbine_powers, width=width, label=title)

# Add names to each bar on the x-axis
ax0.set_xticks(x)
ax0.set_xticklabels(turbine_names, rotation=45, ha='right')

# Add names to each coordinate on the graph with larger font size
for i, (coord, name) in enumerate(zip(coordinates, turbine_names)):
    ax0.annotate(name, coord, xytext=(-10, 10), textcoords='offset points', fontsize=80)

if show_flow_cuts:
    generate_wake_visualization(fi, title)

ax0.set_xticks(range(15))
ax0.set_xticklabels(["T{0}".format(t) for t in range(15)])
ax0.legend()
ax0.set_xlabel("Turbine")
ax0.set_ylabel("Power [MW]")

plt.show()
