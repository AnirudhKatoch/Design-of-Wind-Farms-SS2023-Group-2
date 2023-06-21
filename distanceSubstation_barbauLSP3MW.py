# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 05:25:10 2023

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
    av_turb_spacing = (total_distancex / len(distances_x))/166
    return av_turb_spacing


# Function to calculate average row spacing
def ycoord_line (position_to_substation):
    ycoord_tot15 = np.array([ycoord_line1[position_to_substation],ycoord_line2[position_to_substation],ycoord_line3[position_to_substation],ycoord_line4[position_to_substation]])
    distancesy = []
    for i in range(len(ycoord_tot15) - 1):
        distancey = np.linalg.norm(ycoord_tot15[i] - ycoord_tot15[i + 1])
        distancesy.append(distancey)
    total_distancey = np.sum(distancesy)
    av_row_spacing = (total_distancey / len(distancesy))/166
    return av_row_spacing

# Function to calculate average row spacing without L3
def ycoord_line_uptoL3 (position_to_substation):
    ycoord_tot15 = np.array([ycoord_line1[position_to_substation],ycoord_line2[position_to_substation],ycoord_line3[position_to_substation]])
    distancesy = []
    for i in range(len(ycoord_tot15) - 1):
        distancey = np.linalg.norm(ycoord_tot15[i] - ycoord_tot15[i + 1])
        distancesy.append(distancey)
    total_distancey = np.sum(distancesy)
    av_row_spacing = (total_distancey / len(distancesy))/166
    return av_row_spacing

# Function to calculate average row spacing without L2 and L3
def ycoord_line_noL2L3 (position_to_substation,pos_L2L3):
    ycoord_tot15 = np.array([ycoord_line1[position_to_substation],ycoord_line2[pos_L2L3],ycoord_line3[pos_L2L3],ycoord_line4[position_to_substation]])
    distancesy = []
    for i in range(len(ycoord_tot15) - 1):
        distancey = np.linalg.norm(ycoord_tot15[i] - ycoord_tot15[i + 1])
        distancesy.append(distancey)
    total_distancey = np.sum(distancesy)
    av_row_spacing = (total_distancey / len(distancesy))/166
    return av_row_spacing




######################################################################
# Distance calculation of the Barbau IEA3MW - 15 Turbines - Layout 5
######################################################################


# Substation and Turbine points in the layout

x= 1050
y= 3130
p0 = [1050,3130]
p1 =[3290.02, 2751.55]
p2 = [2941.91, 1113.59]
p3 = [4214.39, 2849.12]
p4 = [4514.22, 4063.33]
p5 = [968.84, 1063.40]
p6 = [4761.37, 1678.77]
p7 = [2155.00, 4965.58]
p8 = [1180.32, 2072.96]
p9 = [2585.93, 3485.85]
p10 = [2200.34, 1927.84]
p11 = [1160.63, 4425.23]
p12 = [3199.48, 4651.04]
p13 = [3702.96, 1818.95]
p14 = [4618.76, 5017.60]
p15 = [5142.32, 2684.34]
dots = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15])


# Average Turbine Spacing Calculation 

# Lines cable distances
line1 = np.array([p14,p4,p12,p7,p11, p0])
turb_spacingL1 = xcoord_line(line1)
xcoord_line1 = line1[ : len(line1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 15 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line2 = np.array([p15,p3,p1,p9, p0])
turb_spacingL2 = xcoord_line(line2)
xcoord_line2 = line2[ : len(line2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 15 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line3 = np.array([p6,p13, p10, p0])
turb_spacingL3 = xcoord_line(line3)
xcoord_line3 = line3[ : len(line3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 15 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation 
line4 = np.array([p2,p5, p8, p0])
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
av_row_spacing_ant = ycoord_line(position_3)
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
background_image = mpimg.imread('images/Layout_1_with_turbinesLSP.png')  
ax.imshow(background_image, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Barbau LSP 3.25MW  - 15 Turbines - Layout 1')
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

p13_1 =[4333.81, 3436.43]
p13_2 = [925.41, 4762.79]
p13_3 = [3293.44, 2772.02]
p13_4 = [1331.00, 452.64]
p13_5 = [4729.43, 4585.10]
p13_6 = [3978.70, 1763.64]
p13_7 = [3098.10, 1057.84]
p13_8 = [1098.15, 2025.64]
p13_9 = [5017.16, 1616.89]
p13_10 = [2124.52, 1551.01]
p13_11 = [2207.35, 3040.56]
p13_12 = [2063.91, 4840.29]
p13_13 = [3220.13, 4650.75]
dots13 = np.array([p13_1,p13_2,p13_3,p13_4,p13_5,p13_6,p13_7,p13_8,p13_9,p13_10,p13_11,p13_12,p13_13])


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line13_1 = np.array([p13_5,p13_13,p13_12,p13_2, p0])
turb_spacingL1 = xcoord_line(line13_1)
xcoord_line1 = line13_1[ : len(line13_1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 13 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line13_2 = np.array([p13_1,p13_3,p13_11, p0])
turb_spacingL2 = xcoord_line(line13_2)
xcoord_line2 = line13_2[ : len(line13_2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 13 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line13_3 = np.array([p13_9,p13_6,p13_7, p13_10, p0])
turb_spacingL3 = xcoord_line(line13_3)
xcoord_line3 = line13_3[ : len(line13_3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 13 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation 
line13_4 = np.array([p13_4,p13_8, p0])
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
background_image13 = mpimg.imread('images/Layout_3_with_turbines13LSP.png')  
ax.imshow(background_image13, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Barbau LSP 3.25MW  - 13 Turbines - Layout 3')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots13[:, 0], dots13[:, 1], c='blue', marker='o')
plt.plot(line13_1[:, 0], line13_1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line13_2[:, 0], line13_2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line13_3[:, 0], line13_3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.plot(line13_4[:, 0], line13_4[:, 1], c='orange', linestyle='-', linewidth=2)
plt.show()



#########################################################
# Distance calculation LSP IEA3MW - 10 Turbines - Layout 3
#########################################################

p10_1 =[3052.26, 1159.60]
p10_2 = [999.89, 803.75]
p10_3 = [4550.65, 2092.52]
p10_4 = [2494.92, 3316.56]
p10_5 = [4659.41, 5078.63]
p10_6 = [1255.11, 4189.58]
p10_7 = [4234.04, 3837.26]
p10_8 = [722.51, 2375.45]
p10_9 = [2527.22, 4567.45]
p10_10 = [2155.43, 2034.84]
dots10 = np.array([p10_1,p10_2,p10_3,p10_4,p10_5,p10_6,p10_7,p10_8,p10_9,p10_10])


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line10_1 = np.array([p10_5,p10_9,p10_6, p0])
turb_spacingL1 = xcoord_line(line10_1)
xcoord_line1 = line10_1[ : len(line10_1)-1 , 0] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 10 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line10_2 = np.array([p10_7,p10_4, p0])
turb_spacingL2 = xcoord_line(line10_2)
xcoord_line2 = line10_2[ : len(line10_2)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 10 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line10_3 = np.array([p10_3,p10_10, p0])
turb_spacingL3 = xcoord_line(line10_3)
xcoord_line3 = line10_3[ : len(line10_3)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 10 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation
line10_4 = np.array([p10_1,p10_2, p10_8, p0])
turb_spacingL4 = xcoord_line(line10_4)
xcoord_line4 = line10_4[ : len(line10_4)-1 , 0] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 10 Turbines is: {:.3f} D".format(turb_spacingL4))
# Average Turbine spacing of IEA 3MW - 15T Layout
total_av_15T = (turb_spacingL1 + turb_spacingL2 + turb_spacingL3 + turb_spacingL4) / 4
print("The average turbine spacing of the 10-T Layout is: {:.3f} D".format(total_av_15T))


# Average row spacing Calculation

ycoord_line1 = line10_1[ : len(line10_1)-1 , 1] # Extracting y-coords from line 1
ycoord_line2 = line10_2[ : len(line10_2)-1 , 1] # Extracting y-coords from line 2
ycoord_line3 = line10_3[ : len(line10_3)-1 , 1] # Extracting y-coords from line 3
ycoord_line4 = line10_4[ : len(line10_4)-1 , 1] # Extracting y-coords from line 3

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
av_row_spacing_ant = ycoord_line_noL2L3(position_3,position_2)
print("The row spacing of the antepenultimate turbines to substation is: {:.3f} D".format(av_row_spacing_ant))
# Total average row spacing of the layout
avg_row_tot = (av_row_spacing_pen+av_row_spacing_ant)/2
print("The average row spacing of the 10-T layout is: {:.3f} D".format(avg_row_tot))


# Combined Homerun trench distance calculation

trenchL1 =  math.sqrt((xcoord_line1[len(xcoord_line1)-1] - p0[0])**2 + (ycoord_line1[len(xcoord_line1)-1] - p0[1])**2)
trenchL2 =  math.sqrt((xcoord_line2[len(xcoord_line2)-1] - p0[0])**2 + (ycoord_line2[len(xcoord_line2)-1] - p0[1])**2)
trenchL3 =  math.sqrt((xcoord_line3[len(xcoord_line3)-1] - p0[0])**2 + (ycoord_line3[len(xcoord_line3)-1] - p0[1])**2)
combtrench15 = trenchL1+trenchL2+trenchL3
print("The combined homerun trench distance of the 13-T layout is: {:.3f} km".format(combtrench15/1.0e3))


# Plotting layout

fig, ax = plt.subplots()
background_image10 = mpimg.imread('images/Layout_3_with_turbines10LSP.png')  
ax.imshow(background_image10, extent=[0, 6800, 0, 5600], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Barbau LSP 3.25MW - 10 Turbines - Layout 3')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots10[:, 0], dots10[:, 1], c='blue', marker='o')
plt.plot(line10_1[:, 0], line10_1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line10_2[:, 0], line10_2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line10_3[:, 0], line10_3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.plot(line10_4[:, 0], line10_4[:, 1], c='orange', linestyle='-', linewidth=2)
plt.show()