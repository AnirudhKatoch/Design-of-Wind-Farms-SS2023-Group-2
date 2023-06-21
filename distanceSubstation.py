# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:53:21 2023

@author: Luis Angel German E.
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
    av_turb_spacing = (total_distancex / len(distances_x))/130
    return av_turb_spacing


# Function to calculate average row spacing
def ycoord_line (position_to_substation):
    ycoord_tot15 = np.array([ycoord_line1[position_to_substation],ycoord_line2[position_to_substation],ycoord_line3[position_to_substation],ycoord_line4[position_to_substation]])
    distancesy = []
    for i in range(len(ycoord_tot15) - 1):
        distancey = np.linalg.norm(ycoord_tot15[i] - ycoord_tot15[i + 1])
        distancesy.append(distancey)
    total_distancey = np.sum(distancesy)
    av_row_spacing = (total_distancey / len(distancesy))/130
    return av_row_spacing



#################################################################
# Distance calculation of the IEA3MW - 15 Turbines - Layout 5
#################################################################


# Substation and Turbine points in the layout

x= 1050
y= 3130
p0 = [1050,3130]
p1 = [3304.81, 4710.94]
p2 = [3452.06, 1863.86] 
p3 = [3426.30, 3753.17]
p4 = [2896.39, 665.65]
p5 = [4398.12, 2530.76]
p6 = [4605.97, 5058.22]
p7 = [1249.99, 4390.38]
p8 = [2215.03, 1909.13]
p9 = [2417.97, 3046.31]
p10 = [1112.34, 1669.99] 
p11 = [5278.39, 4058.06]
p12 = [1203.57, 404.64]
p13 = [3473.79, 2838.37]
p14 = [1982.83, 4998.94]
p15 = [5040.33, 1682.60]
dots = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15]) #Making an array with the turbine coordinates in our layout


# Plotting substation within the Google earth map

fig, ax = plt.subplots()
background_image = mpimg.imread('images/Layout_5_with_image2.png')  
ax.imshow(background_image, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Wind Farm Layout')
plt.scatter(x,y,c='blue',marker='o')
plt.show()


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line1 = np.array([p11,p6,p1,p14,p7, p0]) # Line 1 set-up of turbines
turb_spacingL1 = xcoord_line(line1)
xcoord_line1 = line1[ : len(line1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 15 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line2 = np.array([p15,p5,p13,p3,p9, p0])
turb_spacingL2 = xcoord_line(line2)
xcoord_line2 = line2[ : len(line2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 15 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line3 = np.array([p2,p8, p0])
turb_spacingL3 = xcoord_line(line3)
xcoord_line3 = line3[ : len(line3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 15 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation 
line4 = np.array([p4,p12,p10, p0])
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
# Total average row spacing of the layout
avg_row_tot = (av_row_spacing+av_row_spacing_pen)/2
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
background_image = mpimg.imread('images/Layout_5_with_turbines2.png')  
ax.imshow(background_image, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('IEA3MW - 15 Turbines - Layout 5')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots[:, 0], dots[:, 1], c='blue', marker='o')
plt.plot(line1[:, 0], line1[:, 1], c='blue', linestyle='-', linewidth=2, label='Line 1')
plt.plot(line2[:, 0], line2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line3[:, 0], line3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.plot(line4[:, 0], line4[:, 1], c='orange', linestyle='-', linewidth=2)
plt.show()



##############################################################
# Distance calculation IEA 3MW - 13 Turbines - Layout 5
##############################################################

p13_1 = [2529.82, 4639.66]
p13_2 = [959.96, 4871.71]
p13_3 = [4440.54, 2499.57]
p13_4 = [4567.62, 5014.52]
p13_5 = [3154.93, 2962.56]
p13_6 = [2309.01, 1960.59]
p13_7 = [1204.11, 948.46]
p13_8 = [2676.29, 875.41]
p13_9 = [2150.71, 3081.23]
p13_10 = [827.20, 2403.70]
p13_11 = [3340.83, 1768.35]
p13_12 = [4299.79, 3874.99]
p13_13 = [5067.80, 1710.08]
dots13 = np.array([p13_1,p13_2,p13_3,p13_4,p13_5,p13_6,p13_7,p13_8,p13_9,p13_10,p13_11,p13_12,p13_13])


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line13_1 = np.array([p13_4,p13_12,p13_1,p13_2, p0])
turb_spacingL1 = xcoord_line(line13_1)
xcoord_line1 = line13_1[ : len(line13_1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 13 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line13_2 = np.array([p13_3,p13_5,p13_9, p0])
turb_spacingL2 = xcoord_line(line13_2)
xcoord_line2 = line13_2[ : len(line13_2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 13 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line13_3 = np.array([p13_13,p13_11,p13_6, p0])
turb_spacingL3 = xcoord_line(line13_3)
xcoord_line3 = line13_3[ : len(line13_3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 13 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation 
line13_4 = np.array([p13_8,p13_7,p13_10, p0])
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
av_row_spacing_ant = ycoord_line(position_3)
print("The row spacing of the antepenultimate turbines to substation is: {:.3f} D".format(av_row_spacing_ant))
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
background_image13 = mpimg.imread('images/Layout_5_with_turbines13.png')  
ax.imshow(background_image13, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('IEA3MW - 13 Turbines - Layout 5')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots13[:, 0], dots13[:, 1], c='blue', marker='o')
plt.plot(line13_1[:, 0], line13_1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line13_2[:, 0], line13_2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line13_3[:, 0], line13_3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.plot(line13_4[:, 0], line13_4[:, 1], c='orange', linestyle='-', linewidth=2)
plt.show()



#########################################################
# Distance calculation IEA3MW - 10 Turbines - Layout 3
#########################################################

p10_1 = [4127.32, 2992.05]
p10_2 = [4666.25, 5098.74]
p10_3 = [2222.96, 1799.38]
p10_4 = [2725.61, 3300.48]
p10_5 = [2118.20, 4924.88]
p10_6 = [974.22, 858.94]
p10_7 = [487.60, 4687.74]
p10_8 = [4028.33, 1073.72]
p10_9 = [881.81, 2401.53]
p10_10 = [2652.53, 523.58]
dots10 = np.array([p10_1,p10_2,p10_3,p10_4,p10_5,p10_6,p10_7,p10_8,p10_9,p10_10])


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line10_1 = np.array([p10_2,p10_5,p10_7, p0])
turb_spacingL1 = xcoord_line(line10_1)
xcoord_line1 = line10_1[ : len(line10_1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 10 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line10_2 = np.array([p10_1,p10_4,p10_3, p0])
turb_spacingL2 = xcoord_line(line10_2)
xcoord_line2 = line10_2[ : len(line10_2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 10 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line10_3 = np.array([p10_8,p10_10,p10_6,p10_9, p0])
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
av_row_spacing_ant = ycoord_line(position_3)
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
background_image10 = mpimg.imread('images/Layout_3_with_turbines10.png')  
ax.imshow(background_image10, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('IEA3MW - 10 Turbines - Layout 3')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots10[:, 0], dots10[:, 1], c='blue', marker='o')
plt.plot(line10_1[:, 0], line10_1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line10_2[:, 0], line10_2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line10_3[:, 0], line10_3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.show()
