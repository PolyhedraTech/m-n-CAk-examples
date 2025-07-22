
# m:n-CAk Cellular Automaton Demonstration

## Project Overview
This project demonstrates the capabilities of the m:n-CAk cellular automaton over both continuous and discrete spaces. The main goal is not to model a realistic wildfire, but to showcase how the m:n-CAk automaton can be applied to different spatial representations and how its behavior is consistent across these domains. The simulation uses synthetic layers (vegetation, humidity, wind, fire) to illustrate the automaton's flexibility and power.

## Files in the Project

### Python Scripts
- **Wildfire_on_m_n-CAk.py**: This file contains the main code for the fire cellular automaton. When executed, it displays a window with a slider that can be used to review the entire evolution of the model.

### Fire Layer Vector Files
- **fire.dvc**: A vector file containing the description of the fire layer.
- **fire.vec**: A vector file containing the data of the fire layer.

### Vegetation Layer Vector Files
- **vegetation.dvc**: A vector file containing the description of the vegetation layer.
- **vegetation.vec**: A vector file containing the data of the vegetation layer.

### Humidity Layer Vector Files
- **humidity.dvc**: A vector file containing the description of the humidity layer.
- **humidity.vec**: A vector file containing the data of the humidity layer.

### Vegetation Layer Raster Files
- **vegetation.doc**: A raster file containing the description of the vegetation layer.
- **vegetation.img**: A raster file containing the data of the vegetation layer.

### Humidity Layer Raster Files
- **humidity.doc**: A raster file containing the description of the humidity layer.
- **humidity.img**: A raster file containing the data of the humidity layer.

### Wind Layer Raster Files
- **wind.doc**: A raster file containing the description of the wind layer.
- **wind.img**: A raster file containing the data of the wind layer.

### Other Files
- **README.txt**: This file provides an overview of the project and explanations for the different files included.


## Goal of m_n-CAk_Wildfire.py
Demonstrates the spread of a synthetic wildfire using the m:n-CAk cellular automaton. The purpose is to show how the automaton operates over both continuous (R) and discrete (Z) spaces, using artificial layers for vegetation, humidity, and wind. The results illustrate the equivalence of the automaton's behavior in both topologies.

## Goal of m_n-CAk_Drop.py
Demonstrates the evolution of a drop/spread scenario using the m:n-CAk cellular automaton. This example highlights the automaton's ability to model diffusion-like processes and its consistency across continuous and discrete domains.

## Goal of m_n-CAk_Vortex.py
Demonstrates a vortex scenario using the m:n-CAk cellular automaton. This model showcases the automaton's flexibility in representing more complex, dynamic patterns in both continuous and discrete spaces.

## Running the Simulation
To run the simulation, execute the `Wildfire_on_m_n-CAk.py` script. 
You must select to execute on R or on Z. Executing on Z implies the generation of the raster files from the vectorial files.
A window with a slider will appear, allowing you to review the entire evolution of the wildfire model.

## Dependencies
Ensure you have the necessary dependencies installed. You can install them using the following command:
```bash
pip install -r requirements.txt
