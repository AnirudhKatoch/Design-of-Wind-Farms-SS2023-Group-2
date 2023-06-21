# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 04:23:14 2023

@author: 52222
"""

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math


# Function to calculate average turbine spacing
def xcoord_line(line):
    xcoord = line[ : len(line)-1 , 0]
    distances_x = []
    for i in range(len(xcoord) - 1):
        distancex = np.linalg.norm(xcoord[i] - xcoord[i + 1])
        distances_x.append(distancex)

    # Average distance of x values in line     
    total_distancex = np.sum(distances_x)
    av_turb_spacing = (total_distancex / len(distances_x))/156
    return av_turb_spacing


# Function to calculate average row spacing
def ycoord_line (position_to_substation):
    ycoord_tot15 = np.array([ycoord_line1[position_to_substation],ycoord_line2[position_to_substation],ycoord_line3[position_to_substation],ycoord_line4[position_to_substation]])
    distancesy = []
    for i in range(len(ycoord_tot15) - 1):
        distancey = np.linalg.norm(ycoord_tot15[i] - ycoord_tot15[i + 1])
        distancesy.append(distancey)
    total_distancey = np.sum(distancesy)
    av_row_spacing = (total_distancey / len(distancesy))/156
    return av_row_spacing

# Function to calculate average row spacing
def ycoord_line_uptoL3 (position_to_substation):
    ycoord_tot15 = np.array([ycoord_line1[position_to_substation],ycoord_line2[position_to_substation],ycoord_line3[position_to_substation]])
    distancesy = []
    for i in range(len(ycoord_tot15) - 1):
        distancey = np.linalg.norm(ycoord_tot15[i] - ycoord_tot15[i + 1])
        distancesy.append(distancey)
    total_distancey = np.sum(distancesy)
    av_row_spacing = (total_distancey / len(distancesy))/156
    return av_row_spacing



######################################################################
# Distance calculation of the Barbau IEA3MW - 15 Turbines - Layout 5
######################################################################


# Substation and Turbine points in the layout

x= 1050
y= 3130
p0 = [1050,3130]
p1 = [4107.98, 4463.99]
p2 = [2639.93, 4308.56]
p3 = [1107.24, 836.26]
p4 = [4007.99, 1718.64]
p5 = [2095.88, 2959.57]
p6 = [2195.38, 1943.54]
p7 = [3185.68, 3255.72]
p8 = [928.71, 5020.81]
p9 = [5317.70, 3980.26]
p10 = [3054.43, 1038.56]
p11 = [4232.35, 2725.80]
p12 = [5061.41, 1628.29]
p13 = [716.42, 2360.47]
p14 = [5171.35, 2650.82]
p15 = [2041.71, 5039.32]
dots = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15])


# Average Turbine Spacing Calculation 

# Lines cable distances
line1 = np.array([p9,p1,p2,p15,p8, p0])
turb_spacingL1 = xcoord_line(line1)
xcoord_line1 = line1[ : len(line1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 15 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line2 = np.array([p14,p11,p7,p5, p0])
turb_spacingL2 = xcoord_line(line2)
xcoord_line2 = line2[ : len(line2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 15 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line3 = np.array([p12,p4, p10, p6, p0])
turb_spacingL3 = xcoord_line(line3)
xcoord_line3 = line3[ : len(line3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 15 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation 
line4 = np.array([p3,p13, p0])
turb_spacingL4 = xcoord_line(line4)
xcoord_line4 = line4[ : len(line4)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 4 with 15 Turbines is: {:.3f} D".format(turb_spacingL4))
# Average Turbine spacing of IEA 3MW - 15T Layout
total_av_15T = (turb_spacingL1 + turb_spacingL2 + turb_spacingL3 + turb_spacingL4) / 4
print("The average turbine spacing of the 15-T Layout is: {:.3f} D".format(total_av_15T))


# Average row spacing Calculation

ycoord_line1 = line1[ : len(line1)-1 , 1] # Extracting y-coords from line 1
ycoord_line2 = line2[ : len(line2)-1 , 1] # Extracting y-coords from line 1
ycoord_line3 = line3[ : len(line3)-1 , 1] # Extracting y-coords from line 1
ycoord_line4 = line4[ : len(line4)-1 , 1] # Extracting y-coords from line 1
# Turbine Closest to substation Y-Distance Calculation
position_1= -1
av_row_spacing = ycoord_line(position_1)
print("The row spacing of closest turbines to substation is: {:.3f} D".format(av_row_spacing))
# Penultimate Turbine to Substation Y-Distance Calculation
position_2 = -2
av_row_spacing_pen = ycoord_line(position_2)
print("The row spacing of the penultimate turbines to substation is: {:.3f} D".format(av_row_spacing_pen))
# Antepenultimate Turbine to Substation Y-Distance Calculation
position_3 = -3
av_row_spacing_ant = ycoord_line_uptoL3(position_3)
# Total average row spacing of the layout
avg_row_tot = (av_row_spacing+av_row_spacing_pen+av_row_spacing_ant)/3
print("The average row spacing of the 15-T layout is: {:.3f} D".format(avg_row_tot))



# Combined Homerun trench distance calculation

trenchL1 =  math.sqrt((xcoord_line1[len(xcoord_line1)-1] - p0[0])**2 + (ycoord_line1[len(xcoord_line1)-1] - p0[1])**2)
trenchL2 =  math.sqrt((xcoord_line2[len(xcoord_line2)-1] - p0[0])**2 + (ycoord_line2[len(xcoord_line2)-1] - p0[1])**2)
trenchL3 =  math.sqrt((xcoord_line3[len(xcoord_line3)-1] - p0[0])**2 + (ycoord_line3[len(xcoord_line3)-1] - p0[1])**2)
trenchL4 =  math.sqrt((xcoord_line4[len(xcoord_line4)-1] - p0[0])**2 + (ycoord_line4[len(xcoord_line4)-1] - p0[1])**2)
combtrench15 = trenchL1+trenchL2+trenchL3+trenchL4
print("The combined homerun trench distance of the 15-T layout is: {:.3f} km".format(combtrench15/1.0e3))


# Plotting layout

fig, ax = plt.subplots()
background_image = mpimg.imread('images/Layout_5_with_turbinesBARBAU.png')  
ax.imshow(background_image, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Barbau IEA3.3MW - 15 Turbines - Layout 5')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots[:, 0], dots[:, 1], c='blue', marker='o')
plt.plot(line1[:, 0], line1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line2[:, 0], line2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line3[:, 0], line3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.plot(line4[:, 0], line4[:, 1], c='orange', linestyle='-', linewidth=2)
plt.show()



##############################################################
# Distance calculation Barbau IEA 3MW - 13 Turbines - Layout 5
##############################################################

p13_1 = [4238.56, 4584.61]
p13_2 =  [2122.90, 1934.15]
p13_3 =  [5290.32, 4025.61]
p13_4 =  [4097.35, 2901.03]
p13_5 =  [5160.04, 2758.34]
p13_6 =  [1105.24, 1411.80]
p13_7 =  [2012.96, 4856.51]
p13_8 =  [4034.76, 1734.23]
p13_9 =  [2613.21, 3363.64]
p13_10 =  [1008.35, 5007.05]
p13_11 =  [3040.73, 978.68]
p13_12 =  [3226.84, 4685.18]
p13_13 =  [1373.01, 422.85]
dots13 = np.array([p13_1,p13_2,p13_3,p13_4,p13_5,p13_6,p13_7,p13_8,p13_9,p13_10,p13_11,p13_12,p13_13])


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line13_1 = np.array([p13_3,p13_1,p13_12,p13_7, p13_10, p0])
turb_spacingL1 = xcoord_line(line13_1)
xcoord_line1 = line13_1[ : len(line13_1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 13 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line13_2 = np.array([p13_5,p13_4,p13_9, p0])
turb_spacingL2 = xcoord_line(line13_2)
xcoord_line2 = line13_2[ : len(line13_2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 13 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line13_3 = np.array([p13_8,p13_11,p13_2, p0])
turb_spacingL3 = xcoord_line(line13_3)
xcoord_line3 = line13_3[ : len(line13_3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 13 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation 
line13_4 = np.array([p13_13,p13_6, p0])
turb_spacingL4 = xcoord_line(line13_4)
xcoord_line4 = line13_4[ : len(line13_4)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 4 with 15 Turbines is: {:.3f} D".format(turb_spacingL4))
# Average Turbine spacing of IEA 3MW - 15T Layout
total_av_15T = (turb_spacingL1 + turb_spacingL2 + turb_spacingL3 + turb_spacingL4) / 4
print("The average turbine spacing of the 13-T Layout is: {:.3f} D".format(total_av_15T))


# Average row spacing Calculation

ycoord_line1 = line13_1[ : len(line13_1)-1 , 1] # Extracting y-coords from line 1
ycoord_line2 = line13_2[ : len(line13_2)-1 , 1] # Extracting y-coords from line 1
ycoord_line3 = line13_3[ : len(line13_3)-1 , 1] # Extracting y-coords from line 1
ycoord_line4 = line13_4[ : len(line13_4)-1 , 1] # Extracting y-coords from line 1
# Turbine Closest to substation Y-Distance Calculation
position_1= -1
av_row_spacing = ycoord_line(position_1)
print("The row spacing of closest turbines to substation is: {:.3f} D".format(av_row_spacing))
# Penultimate Turbine to Substation Y-Distance Calculation
position_2 = -2
av_row_spacing_pen = ycoord_line(position_2)
print("The row spacing of the penultimate turbines to substation is: {:.3f} D".format(av_row_spacing_pen))
# Antepenultimate Turbine to Substation Y-Distance Calculation
position_3 = -3
av_row_spacing_ant = ycoord_line_uptoL3(position_3)
# Total average row spacing of the layout
avg_row_tot = (av_row_spacing+av_row_spacing_pen+av_row_spacing_ant)/3
print("The average row spacing of the 13-T layout is: {:.3f} D".format(avg_row_tot))


# Combined Homerun trench distance calculation

trenchL1 =  math.sqrt((xcoord_line1[len(xcoord_line1)-1] - p0[0])**2 + (ycoord_line1[len(xcoord_line1)-1] - p0[1])**2)
trenchL2 =  math.sqrt((xcoord_line2[len(xcoord_line2)-1] - p0[0])**2 + (ycoord_line2[len(xcoord_line2)-1] - p0[1])**2)
trenchL3 =  math.sqrt((xcoord_line3[len(xcoord_line3)-1] - p0[0])**2 + (ycoord_line3[len(xcoord_line3)-1] - p0[1])**2)
trenchL4 =  math.sqrt((xcoord_line4[len(xcoord_line4)-1] - p0[0])**2 + (ycoord_line4[len(xcoord_line4)-1] - p0[1])**2)
combtrench15 = trenchL1+trenchL2+trenchL3+trenchL4
print("The combined homerun trench distance of the 13-T layout is: {:.3f} km".format(combtrench15/1.0e3))


# Plotting layout

fig, ax = plt.subplots()
background_image13 = mpimg.imread('images/Layout_5_with_turbines13BARBAU.png')  
ax.imshow(background_image13, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Barbau IEA3.3MW - 13 Turbines - Layout 5')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots13[:, 0], dots13[:, 1], c='blue', marker='o')
plt.plot(line13_1[:, 0], line13_1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line13_2[:, 0], line13_2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line13_3[:, 0], line13_3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.plot(line13_4[:, 0], line13_4[:, 1], c='orange', linestyle='-', linewidth=2)
plt.show()



#########################################################
# Distance calculation Barbau IEA3MW - 10 Turbines - Layout 3
#########################################################

p10_1 = [3917.33, 3130.38]
p10_2 =  [2990.61, 1149.16]
p10_3 =  [974.08, 1954.01]
p10_4 =  [819.76, 4858.50]
p10_5 =  [4458.84, 4948.15]
p10_6 =  [2530.24, 4713.10]
p10_7 =  [1231.68, 422.40]
p10_8 =  [2627.22, 3436.60]
p10_9 =  [4572.49, 2033.99]
p10_10 =  [5331.16, 3967.05]
dots10 = np.array([p10_1,p10_2,p10_3,p10_4,p10_5,p10_6,p10_7,p10_8,p10_9,p10_10])


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line10_1 = np.array([p10_10,p10_5,p10_6,p10_4, p0])
turb_spacingL1 = xcoord_line(line10_1)
xcoord_line1 = line10_1[ : len(line10_1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 10 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line10_2 = np.array([p10_9,p10_1,p10_8, p0])
turb_spacingL2 = xcoord_line(line10_2)
xcoord_line2 = line10_2[ : len(line10_2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 10 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line10_3 = np.array([p10_2,p10_7,p10_3, p0])
turb_spacingL3 = xcoord_line(line10_3)
xcoord_line3 = line10_3[ : len(line10_3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 10 Turbines is: {:.3f} D".format(turb_spacingL3))
# Average Turbine spacing of IEA 3MW - 15T Layout
total_av_15T = (turb_spacingL1 + turb_spacingL2 + turb_spacingL3) / 3
print("The average turbine spacing of the 10-T Layout is: {:.3f} D".format(total_av_15T))


# Average row spacing Calculation

ycoord_line1 = line10_1[ : len(line10_1)-1 , 1] # Extracting y-coords from line 1
ycoord_line2 = line10_2[ : len(line10_2)-1 , 1] # Extracting y-coords from line 2
ycoord_line3 = line10_3[ : len(line10_3)-1 , 1] # Extracting y-coords from line 3

# Turbine Closest to substation Y-Distance Calculation
position_1= -1
av_row_spacing = ycoord_line(position_1)
print("The row spacing of closest turbines to substation is: {:.3f} D".format(av_row_spacing))
# Penultimate Turbine to Substation Y-Distance Calculation
position_2 = -2
av_row_spacing_pen = ycoord_line(position_2)
print("The row spacing of the penultimate turbines to substation is: {:.3f} D".format(av_row_spacing_pen))
# Antepenultimate Turbine to Substation Y-Distance Calculation
position_3 = -3
av_row_spacing_ant = ycoord_line_uptoL3(position_3)
print("The row spacing of the antepenultimate turbines to substation is: {:.3f} D".format(av_row_spacing_ant))
# Total average row spacing of the layout
avg_row_tot = (av_row_spacing+av_row_spacing_pen+av_row_spacing_ant)/3
print("The average row spacing of the 10-T layout is: {:.3f} D".format(avg_row_tot))


# Combined Homerun trench distance calculation

trenchL1 =  math.sqrt((xcoord_line1[len(xcoord_line1)-1] - p0[0])**2 + (ycoord_line1[len(xcoord_line1)-1] - p0[1])**2)
trenchL2 =  math.sqrt((xcoord_line2[len(xcoord_line2)-1] - p0[0])**2 + (ycoord_line2[len(xcoord_line2)-1] - p0[1])**2)
trenchL3 =  math.sqrt((xcoord_line3[len(xcoord_line3)-1] - p0[0])**2 + (ycoord_line3[len(xcoord_line3)-1] - p0[1])**2)
combtrench15 = trenchL1+trenchL2+trenchL3
print("The combined homerun trench distance of the 13-T layout is: {:.3f} km".format(combtrench15/1.0e3))


# Plotting layout

fig, ax = plt.subplots()
background_image10 = mpimg.imread('images/Layout_5_with_turbines10BARBAU.png')  
ax.imshow(background_image10, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Barbau IEA3.3MW - 10 Turbines - Layout 3')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots10[:, 0], dots10[:, 1], c='blue', marker='o')
plt.plot(line10_1[:, 0], line10_1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line10_2[:, 0], line10_2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line10_3[:, 0], line10_3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.show()