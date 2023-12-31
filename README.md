# Design of Wind Farms for Kastrup and Bayern

The problem statement was to design and optimize two wind farms at locations Bayern and Kastrup with a nominal capacity of up to 50MW to maximize the Profitability Index (PI) and reduce LCOE.

Location Bayern

<img align="center" alt="Coding" width="400" src="https://i.imgur.com/zLzCVTO.png">

Location Kastrup

<img align="center" alt="Coding" width="400" src="https://i.imgur.com/EiGJiWa.png">

Provided data and constraints

* Two locations that are Bayern and Kastrup with their respective specifications such as Wind Shear Exponent, 
	 Line Frequency, Standard Voltage, Road length adder, Rental Costs, along with 15 other specifications.
* Wind data from the New European Wind Atlas from 01/01/2005 to 31/12/2018
* Three turbines' types namely IEA 3.4MW Reference , BAR_BAU- IEA_3.3MW and BAR_BAU_LSP_3.25MW and their specifications
* Market spot price forecast for the next 20 years


 ## Methodology and calculations

 Conducted a comprehensive research study to investigate the impact of various turbine types and quantities. The study involved generating data for wind farms consisting of 10, 13, and 15 turbines, each deployed at Bayern and Kastrup. The main objective of this analysis was to examine the variations and reach a consensus regarding the influencing factors of location, turbine type, and the number of turbines. Economic aspects of these wind farms, considering the Acquired Energy Production (AEP) and the design parameters. Following this, a similar methodology was again carried out, but this time, incorporating wake steering into the equation.


 Tasks

* Collaborated with a team of three for the design and optimization of wind farms utilizing the FLORIS library in Python.
* Developed a Python-based simulation model for processing images of potential sites.
* Developed Excel database from obtained results for wind farm data analysis and optimization.
* Evaluated projects for various economic factors using Python and prepared presentations for review.

## Results

The results were visualized
(Shown below is only part of the results)

AEP

<img align="center" alt="Coding" width="600" src="https://i.imgur.com/9dfKPe2.png">

LCOE

<img align="center" alt="Coding" width="600" src="https://i.imgur.com/1je7PvV.png">

Cost Breakdown

<img align="center" alt="Coding" width="600" src="https://i.imgur.com/EqkNKW7.png">

PI (Profitability Index)

<img align="center" alt="Coding" width="600" src="https://i.imgur.com/hhHSBZA.png">

* So, a total of 36 different scenarios were evaluated based on number of turbines, type of turbines, location and wake steering.
* Out of all the scenario it was concluded that a total of 15 IEA 3.4MW Reference turbines at Kastrup location with wake steering is the most economical feasible project.
* It has AEP of 295.9917 GWh, LCOE of  38.85 $/MWh , IRR of 18.5 and PI of 1.68 . 




