# Wildfire Simulation Project

## Overview
This project simulates the spread of a wildfire using a cellular automaton model based on m:n-CAk. The simulation takes into account various factors such as vegetation type, and humidity levels, to model the propagation of the fire over a landscape. The output of the simulation is a video that visually represents the spread of the wildfire over time, providing insights into how different conditions affect the fire's behavior.

## Files in the Project

### Python Scripts
- **Wildfire_on_m_n-CAk.py**: This file contains the main code for the fire cellular automaton. When executed, it displays a window with a slider that can be used to review the entire evolution of the model.

### Vegetation Layer Raster Files
- **vegetation_map.doc**: A raster file containing the description of the vegetation layer.
- **vegetation_map.img**: A raster file containing the data of the vegetation layer.

### Humidity Layer Raster Files
- **humidity_map.doc**: A raster file containing the description of the humidity layer.
- **humidity_map.img**: A raster file containing the data of the humidity layer.

### Other Files
- **README.txt**: This file provides an overview of the project and explanations for the different files included.

## Goal of Wildfire_on_m_n-CAk.py
The goal of the code in `Wildfire_on_m_n-CAk.py` is to simulate the spread of a wildfire using a cellular automaton model based on m:n-CAk. The simulation takes into account various factors such as vegetation type, and humidity levels to model the propagation of the fire over a landscape. The output of the simulation is a video that visually represents the spread of the wildfire over time, providing insights into how different conditions affect the fire's behavior.
The simulation can be done over Z and over R and the results are equivalent because vicinity, nucleous and combination function are euqivalent and both models are based over the discrete topology.

## Running the Simulation
To run the simulation, execute the `Wildfire_on_m_n-CAk.py` script. A window with a slider will appear, allowing you to review the entire evolution of the wildfire model.

## Dependencies
Ensure you have the necessary dependencies installed. You can install them using the following command:
```bash
pip install -r requirements.txt
