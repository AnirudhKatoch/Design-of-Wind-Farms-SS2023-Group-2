from floris.tools import FlorisInterface, WindRose
from floris.tools import FlorisInterface
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math

# Load the image
image_ = cv2.imread('images/Bayern_DE_3.jpg')

# Flip the image horizontally (about the y-axis)
image_ = cv2.flip(image_, 0)

# Resize the image to 6000 x 6000 pixels
image = cv2.resize(image_, (6800, 5600))

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the image to get only blue regions
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Find contours of blue regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty canvas for plotting
plot_image = np.zeros_like(image)

# Create an empty list to store the generated points
random_points = []

# Generate a random boundary index
random_boundary_index = random.randint(0, len(contours) - 1)
boundary_count = 0
random_points_coordinates =[]

# Calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

random_points = []
random_points_coordinates = []
D = 166 # Diameter for turbine
min_distance = 5.0 * D  # Minimum distance between two turbines


while len(random_points) < 15:
    # Get the current boundary contour
    contour = contours[random_boundary_index]

    # Calculate the area of the current contour
    area = cv2.contourArea(contour)

    # Generate a random point within the current boundary if the area is greater than 1256.63
    if area > 1256.63:
        x = random.uniform(0, image.shape[1])
        y = random.uniform(0, image.shape[0])
        point = (x, y)
        valid_point = True

        # Check if the point falls within the boundary contour
        if cv2.pointPolygonTest(contour, point, False) > 0:
            # Check the distance between the new point and existing points
            for existing_point in random_points:
                distance = calculate_distance(point, existing_point)
                if distance < min_distance:
                    valid_point = False
                    break

            if valid_point:
                random_points.append(point)
                random_points_coordinates.append((point[0], point[1]))

        # Draw the current boundary contour on the plot image
        cv2.drawContours(plot_image, [contour], -1, (0, 255, 0), 2)

    # Generate a new random boundary index
    random_boundary_index = random.randint(0, len(contours) - 1)

# Set up the figure and axes for plotting the image on the graph
fig, ax = plt.subplots(1, 1)

# Load the background image
background_image = mpimg.imread('images/Bayern_DE_3.jpg')  # Replace 'path_to_your_image.jpg' with the actual path to your image

# Set the background image as the plot's background
ax.imshow(background_image, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)

# Plot the turbine layout with a rectangular boundary
# ax.scatter(layout_x, layout_y, marker='o', color='red', edgecolors='red', s=20)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Wind Farm Layout')

#Result is plotted
plt.figure(2)

# Add title, x-axis label, and y-axis label
plt.title("Wind Farm Layout")
plt.xlabel("Latitude")
plt.ylabel("Longitude")

# Plot the image with boundaries
plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB), origin='lower')

# Plot the random points
random_points = np.array(random_points)
plt.plot(random_points[:, 0], random_points[:, 1], 'r.', markersize=5)


# Print the coordinates of the random points
# Print the coordinates of the random points as a list with two decimal places
print("Random Points Coordinates:")
print([f"({point[0]:.2f}, {point[1]:.2f})" for point in random_points_coordinates])


# Read in the wind rose using the class
wind_rose = WindRose()
wind_rose.read_wind_rose_csv("nc_files/mesotimeseries_Bayern_DK_120m_with_freq/mesotimeseries_Bayern_DK_120m_with_freq.csv")
print(wind_rose)
# Load the FLORIS object
fi = FlorisInterface("inputs/gch.yaml") # GCH model
# fi = FlorisInterface("inputs/cc.yaml") # CumulativeCurl model
print("barbau_LSP_3MW")

#Defining x and y coordinates from random_points_coordinates
layout_x = [coord[0] for coord in random_points_coordinates]
layout_y = [coord[1] for coord in random_points_coordinates]
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

aep = fi.get_farm_AEP_wind_rose_class(
    wind_rose=wind_rose,
    cut_in_wind_speed=4.0,  # Wakes are not evaluated below this wind speed
    cut_out_wind_speed=25.0,  # Wakes are not evaluated above this wind speed
)
print("Farm AEP (with cut_in/out specified): {:.3f} GWh".format(aep / 1.0e9))

# Finally, we can also compute the AEP while ignoring all wake calculations.
aep_no_wake = fi.get_farm_AEP_wind_rose_class(wind_rose=wind_rose, no_wake=True)
print("Farm AEP (no_wake=True): {:.3f} GWh".format(aep_no_wake / 1.0e9))

plt.show()