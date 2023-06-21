# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:44:01 2023

@author: 52222
"""
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

#For Kastrup, 

# Function to calculate average turbine spacing
def xcoord_line(line):
    xcoord = line[ : len(line)-1 , 1]
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



#################################################################
# Distance calculation of the Barbau IEA3MW - 15 Turbines - Layout 5
#################################################################

# Substation and Turbine points in the layout

x= 4900
y= 5300
p0 = [4900,5300]
p1 =[4524.42, 2254.53] 
p2 = [6038.91, 1989.90] 
p3 = [3082.06, 3276.61] 
p4 = [7188.98, 1956.77] 
p5 = [1277.68, 1021.17]
p6 = [4083.38, 1251.55]
p7 = [2998.47, 2266.21]
p8 = [5108.04, 1153.96]
p9 = [3970.05, 3521.04]
p10 = [5386.30, 3095.99]
p11 = [1405.30, 2010.88]
p12 = [6642.94, 2628.60]
p13 = [1987.97, 413.96]
p14 = [7527.96, 2750.20]
p15 = [2995.88, 132.35]
dots = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15])

# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line1 = np.array([p4,p14, p0])
turb_spacingL1 = xcoord_line(line1)
xcoord_line1 = line1[ : len(line1)-1 , 1] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 15 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line2 = np.array([p8,p2,p12, p0])
turb_spacingL2 = xcoord_line(line2)
xcoord_line2 = line2[ : len(line2)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 15 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line3 = np.array([p15,p6, p1, p10, p0])
turb_spacingL3 = xcoord_line(line3)
xcoord_line3 = line3[ : len(line3)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 15 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation 
line4 = np.array([p13,p5,p11, p7, p3, p9, p0])
turb_spacingL4 = xcoord_line(line4)
xcoord_line4 = line4[ : len(line4)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 4 with 15 Turbines is: {:.3f} D".format(turb_spacingL4))
# Average Turbine spacing of IEA 3MW - 15T Layout
total_av_15T = (turb_spacingL1 + turb_spacingL2 + turb_spacingL3 + turb_spacingL4) / 4
print("The average turbine spacing of the 15-T Layout is: {:.3f} D".format(total_av_15T))


# Average row spacing Calculation

ycoord_line1 = line1[ : len(line1)-1 , 0] # Extracting y-coords from line 1
ycoord_line2 = line2[ : len(line2)-1 , 0] # Extracting y-coords from line 1
ycoord_line3 = line3[ : len(line3)-1 , 0] # Extracting y-coords from line 1
ycoord_line4 = line4[ : len(line4)-1 , 0] # Extracting y-coords from line 1
# Turbine Closest to substation Y-Distance Calculation
position_1= -1
av_row_spacing = ycoord_line(position_1)
print("The row spacing of closest turbines to substation is: {:.3f} D".format(av_row_spacing))
# Penultimate Turbine to Substation Y-Distance Calculation
position_2 = -2
av_row_spacing_pen = ycoord_line(position_2)
print("The row spacing of the penultimate turbines to substation is: {:.3f} D".format(av_row_spacing_pen))
# AntePenultimate Turbine to Substation Y-Distance Calculation
#position_3 = -3
#av_row_spacing_ant = ycoord_line(position_3)
#print("The row spacing of the penultimate turbines to substation is: {:.3f} D".format(av_row_spacing_ant))
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
background_image = mpimg.imread('images/Layout_6_with_turbine15DKbarbau.png')  
ax.imshow(background_image, extent=[0, 8950, 0, 6260], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title(' DK BARBAU IEA3MW - 15 Turbines - Layout 6')
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

p13_1 =[6086.34, 2354.11] 
p13_2 = [4506.15, 1671.19] 
p13_3 = [4832.19, 2765.63] 
p13_4 = [5618.38, 1415.11] 
p13_5 = [3543.91, 3828.69] 
p13_6 = [2993.16, 3027.53] 
p13_7 = [7411.59, 2303.29]
p13_8 = [1178.00, 1652.10]
p13_9 = [3231.74, 2119.39]
p13_10 = [6773.83, 1616.74]
p13_11 = [5316.96, 411.93]
p13_12 = [3405.12, 1161.89]
p13_13 = [1933.06, 362.19]
dots13 = np.array([p13_1,p13_2,p13_3,p13_4,p13_5,p13_6,p13_7,p13_8,p13_9,p13_10,p13_11,p13_12,p13_13])

# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line13_1 = np.array([p13_10,p13_7, p0])
turb_spacingL1 = xcoord_line(line13_1)
xcoord_line1 = line13_1[ : len(line13_1)-1 , 1] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 13 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line13_2 = np.array([p13_11,p13_4,p13_1, p0])
turb_spacingL2 = xcoord_line(line13_2)
xcoord_line2 = line13_2[ : len(line13_2)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 13 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line13_3 = np.array([p13_13,p13_12, p13_2, p13_3, p0])
turb_spacingL3 = xcoord_line(line13_3)
xcoord_line3 = line13_3[ : len(line13_3)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 13 Turbines is: {:.3f} D".format(turb_spacingL3))
# Line 4 X Distance Calculation
line13_4 = np.array([p13_8,p13_9,p13_6, p13_5, p0])
turb_spacingL4 = xcoord_line(line13_4)
xcoord_line4 = line13_4[ : len(line13_4)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 4 with 15 Turbines is: {:.3f} D".format(turb_spacingL4))
# Average Turbine spacing of IEA 3MW - 15T Layout
total_av_15T = (turb_spacingL1 + turb_spacingL2 + turb_spacingL3 + turb_spacingL4) / 4
print("The average turbine spacing of the 13-T Layout is: {:.3f} D".format(total_av_15T))


# Average row spacing Calculation

ycoord_line1 = line13_1[ : len(line13_1)-1 , 0] # Extracting y-coords from line 1
ycoord_line2 = line13_2[ : len(line13_2)-1 , 0] # Extracting y-coords from line 1
ycoord_line3 = line13_3[ : len(line13_3)-1 , 0] # Extracting y-coords from line 1
ycoord_line4 = line13_4[ : len(line13_4)-1 , 0] # Extracting y-coords from line 1
# Turbine Closest to substation Y-Distance Calculation
position_1= -1
av_row_spacing = ycoord_line(position_1)
print("The row spacing of closest turbines to substation is: {:.3f} D".format(av_row_spacing))
# Penultimate Turbine to Substation Y-Distance Calculation
position_2 = -2
av_row_spacing_pen = ycoord_line(position_2)
print("The row spacing of the penultimate turbines to substation is: {:.3f} D".format(av_row_spacing_pen))
# Antepenultimate Turbine to Substation Y-Distance Calculation
#position_3 = -3
#av_row_spacing_ant = ycoord_line(position_3)
#print("The row spacing of the antepenultimate turbines to substation is: {:.3f} D".format(av_row_spacing_ant))
# Total average row spacing of the layout
avg_row_tot = (av_row_spacing+av_row_spacing_pen)/3
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
background_image13 = mpimg.imread('images/Layout_1_with_turbines13DKBARBAU.png')  
ax.imshow(background_image13, extent=[0, 8950, 0, 6260], aspect='auto', alpha=1.0)
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

p10_1 =[4397.69, 3098.99] 
p10_2 = [4366.92, 1325.35] 
p10_3 = [5855.05, 2487.19] 
p10_4 = [7752.62, 2356.37] 
p10_5 = [6557.83, 1539.29] 
p10_6 = [2880.62, 2273.82] 
p10_7 = [1281.58, 1037.75]
p10_8 = [3365.46, 3875.01]
p10_9 = [5406.09, 413.40]
p10_10 = [3000.16, 129.42]
dots10 = np.array([p10_1,p10_2,p10_3,p10_4,p10_5,p10_6,p10_7,p10_8,p10_9,p10_10])


# Average Turbine Spacing Calculation 

# Line 1 X Distance Calculation
line10_1 = np.array([p10_9,p10_5,p10_4, p0])
turb_spacingL1 = xcoord_line(line10_1)
xcoord_line1 = line10_1[ : len(line10_1)-1 , 1] # Extracting x-coords from line 1 for trench calculation
print("The average turbine spacing of Line 1 with 10 Turbines is: {:.3f} D".format(turb_spacingL1))
# Line 2 X Distance Calculation
line10_2 = np.array([p10_10,p10_2,p10_3, p0])
turb_spacingL2 = xcoord_line(line10_2)
xcoord_line2 = line10_2[ : len(line10_2)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 2 with 10 Turbines is: {:.3f} D".format(turb_spacingL2))
# Line 3 X Distance Calculation
line10_3 = np.array([p10_7,p10_6,p10_1,p10_8, p0])
turb_spacingL3 = xcoord_line(line10_3)
xcoord_line3 = line10_3[ : len(line10_3)-1 , 1] # Extracting x-coords from line 1
print("The average turbine spacing of Line 3 with 10 Turbines is: {:.3f} D".format(turb_spacingL3))
# Average Turbine spacing of IEA 3MW - 15T Layout
total_av_15T = (turb_spacingL1 + turb_spacingL2 + turb_spacingL3) / 3
print("The average turbine spacing of the 10-T Layout is: {:.3f} D".format(total_av_15T))


# Average row spacing Calculation

ycoord_line1 = line10_1[ : len(line10_1)-1 , 0] # Extracting y-coords from line 1
ycoord_line2 = line10_2[ : len(line10_2)-1 , 0] # Extracting y-coords from line 2
ycoord_line3 = line10_3[ : len(line10_3)-1 , 0] # Extracting y-coords from line 3
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
background_image10 = mpimg.imread('images/Layout_1_with_turbines10DKbarbau.png')  
ax.imshow(background_image10, extent=[0, 8950, 0, 6260], aspect='auto', alpha=1.0)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_title('Barbau IEA3.3MW - 10 Turbines - Layout 3')
plt.scatter(x,y,c='red',marker='o')
plt.scatter(dots10[:, 0], dots10[:, 1], c='blue', marker='o')
plt.plot(line10_1[:, 0], line10_1[:, 1], c='blue', linestyle='-', linewidth=2)
plt.plot(line10_2[:, 0], line10_2[:, 1], c='green', linestyle='-', linewidth=2)
plt.plot(line10_3[:, 0], line10_3[:, 1], c='yellow', linestyle='-', linewidth=2)
plt.show()
