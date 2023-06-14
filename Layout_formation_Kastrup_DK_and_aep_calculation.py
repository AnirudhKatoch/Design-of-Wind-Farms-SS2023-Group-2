from floris.tools import FlorisInterface, WindRose
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math

# Load the image
image_ = cv2.imread('images/Kastrup_DK_3_8K.jpg')
image_ = cv2.flip(image_, 0)
image = cv2.resize(image_, (8950, 6260))
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])


# Calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# Set up lists to store the results
all_random_points = []
all_aep_results = []

# Run the code 10 times
for _ in range(10):
    # Threshold the image to get only blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plot_image = np.zeros_like(image)
    random_points = []
    random_points_coordinates = []
    D = 166
    min_distance = 7.0 * D

    while len(random_points) < 10:
        random_boundary_index = random.randint(0, len(contours) - 1)
        contour = contours[random_boundary_index]
        area = cv2.contourArea(contour)

        if area > 1256.63:
            x = random.uniform(0, image.shape[1])
            y = random.uniform(0, image.shape[0])
            point = (x, y)
            valid_point = True

            if cv2.pointPolygonTest(contour, point, False) > 0:
                for existing_point in random_points:
                    distance = calculate_distance(point, existing_point)
                    if distance < min_distance:
                        valid_point = False
                        break

                if valid_point:
                    random_points.append(point)
                    random_points_coordinates.append((point[0], point[1]))

                cv2.drawContours(plot_image, [contour], -1, (0, 255, 0), 2)

    # fig, ax = plt.subplots(1, 1)
    # background_image = mpimg.imread('images/Kastrup_DK_3_8K.jpg')
    # ax.imshow(background_image, extent=[0, 8950, 0, 6260], aspect='auto', alpha=1.0)
    # ax.set_xlabel('Latitude')
    # ax.set_ylabel('Longitude')
    # ax.set_title('Wind Farm Layout')
    plt.figure(1)
    plt.title("Wind Farm Layout")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB), origin='lower')
    random_points = np.array(random_points)
    plt.plot(random_points[:, 0], random_points[:, 1], 'r.', markersize=5)
    random_points_coordinates = [(f"{point[0]:.2f}", f"{point[1]:.2f}") for point in random_points_coordinates]
    print("Random Points Coordinates:")
    print(random_points_coordinates)

    wind_rose = WindRose()
    wind_rose.read_wind_rose_csv("nc_files/mesotimeseries_Kastrup-DK120m_with_freq/mesotimeseries_Kastrup-DK120m_with_freq.csv")

    fi = FlorisInterface("inputs/gch.yaml")
    layout_x = [coord[0] for coord in random_points_coordinates]
    layout_y = [coord[1] for coord in random_points_coordinates]
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

    aep = fi.get_farm_AEP_wind_rose_class(wind_rose=wind_rose, cut_in_wind_speed=4.0, cut_out_wind_speed=25.0)
    aep_no_wake = fi.get_farm_AEP_wind_rose_class(wind_rose=wind_rose, no_wake=True)

    print("Farm AEP (with cut_in/out specified): {:.3f} GWh".format(aep / 1.0e9))
    print("Farm AEP (no_wake=True): {:.3f} GWh".format(aep_no_wake / 1.0e9))

    all_random_points.append(random_points_coordinates)
    all_aep_results.append(aep / 1.0e9)

    plt.show()

