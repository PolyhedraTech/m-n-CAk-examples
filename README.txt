
# m:n-CAk Cellular Automaton Wildfire Demonstration

## Project Overview
This project demonstrates the capabilities of the m:n-CAk cellular automaton over both continuous (R²) and discrete (Z²) spaces using a wildfire simulation example. The main goal is not to model a realistic wildfire, but to showcase how the m:n-CAk automaton can be applied to different spatial representations and how its behavior is consistent across these domains. The simulation uses synthetic layers (vegetation, humidity, wind, fire) to illustrate the automaton's flexibility and power.

## Files in the Project

### Python Scripts
- **m_n_-CAk_Wildfire.py**: The main wildfire simulation script using m:n-CAk cellular automaton. When executed, it displays a window with a slider for reviewing the complete evolution of the model over both Z² (discrete) and R² (continuous) domains.
- **m_n_-CAk_Vortex.py**: Demonstration of a vortex scenario using the m:n-CAk cellular automaton, showcasing the automaton's flexibility in representing complex, dynamic patterns.

### Fire Layer Files
#### Vector Files (R²)
- **fire.vdc**: Vector file containing the description of the fire layer.
- **fire.vec**: Vector file containing the data of the fire layer.

#### Raster Files (Z²)
- **fire.doc**: Raster file containing the description of the fire layer.
- **fire.img**: Raster file containing the data of the fire layer.

### Vegetation Layer Files
#### Vector Files (R²)
- **vegetation.dvc**: Vector file containing the description of the vegetation layer.
- **vegetation.vec**: Vector file containing the data of the vegetation layer.

#### Raster Files (Z²)
- **vegetation.doc**: Raster file containing the description of the vegetation layer.
- **vegetation.img**: Raster file containing the data of the vegetation layer.

### Humidity Layer Files
#### Vector Files (R²)
- **humidity.dvc**: Vector file containing the description of the humidity layer.
- **humidity.vec**: Vector file containing the data of the humidity layer.

#### Raster Files (Z²)
- **humidity.doc**: Raster file containing the description of the humidity layer.
- **humidity.img**: Raster file containing the data of the humidity layer.

### Wind Layer Files
#### Vector Files (R²)
- **wind.vdc**: Vector file containing the description of the wind layer.
- **wind.vec**: Vector file containing the data of the wind layer.

#### Raster Files (Z²)
- **wind.doc**: Raster file containing the description of the wind layer.
- **wind.img**: Raster file containing the data of the wind layer.
- **no_wind.img**: Raster file for scenarios without wind effects.

### Simple Layer Files (Example/Test Data)
- **simple.vdc**: Vector file containing the description of a simple test layer.
- **simple.vec**: Vector file containing the data of a simple test layer.

### Configuration and Documentation
- **dependencies.txt**: List of Python dependencies required for the project.
- **README.txt**: This file providing project overview and file descriptions.
- **.gitignore**: Git ignore file for version control.

## Main Script: m_n_-CAk_Wildfire.py
Demonstrates the spread of a synthetic wildfire using the m:n-CAk cellular automaton. The script showcases how the automaton operates equivalently over both continuous (R²) and discrete (Z²) spaces, using artificial layers for vegetation, humidity, wind, and fire. The simulation illustrates the mathematical equivalence of the automaton's behavior across different topological representations.

### Key Features:
- **Dual Domain Support**: Operates on both Z² (discrete raster) and R² (continuous vector) domains
- **Interactive Visualization**: GUI with sliders for temporal evolution review
- **IDRISI Format Support**: Reads both vector (.vdc/.vec) and raster (.doc/.img) file formats
- **Multiple Layer Integration**: Vegetation, humidity, wind, and fire layers
- **Real-time Animation**: Step-by-step evolution visualization
- **Configurable Parameters**: Adjustable simulation parameters and initial conditions

## Additional Script: m_n_-CAk_Vortex.py
Demonstrates a vortex scenario using the m:n-CAk cellular automaton, showcasing the automaton's flexibility in representing complex, dynamic patterns in both continuous and discrete spaces.

## Running the Simulation
To run the wildfire simulation:

1. Execute the `m_n_-CAk_Wildfire.py` script
2. Select the domain: 'Z' for discrete (raster) or 'R' for continuous (vector) execution
3. When running on Z² domain, the script automatically generates raster files from vectorial files
4. Use the interactive GUI window with sliders to review the complete temporal evolution

```bash
python m_n_-CAk_Wildfire.py
```

## Dependencies
The project requires several Python packages. Install them using:

```bash
pip install matplotlib numpy shapely scikit-learn tkinter
```

Or if you have a requirements file:
```bash
pip install -r dependencies.txt
```

### Required Packages:
- **matplotlib**: For visualization and plotting
- **numpy**: For numerical computations and array operations
- **shapely**: For geometric operations in the R² domain
- **scikit-learn**: For clustering algorithms (DBSCAN)
- **tkinter**: For GUI components (usually included with Python)

## File Format Notes
- **Vector files (.vdc/.vec)**: IDRISI vector format for continuous R² domain
- **Raster files (.doc/.img)**: IDRISI raster format for discrete Z² domain
- **Automatic conversion**: The script can convert between vector and raster representations as needed

## Research Context
This implementation serves as a proof-of-concept for the m:n-CAk cellular automaton's capability to maintain behavioral consistency across different spatial representations, supporting research into the mathematical foundations of cellular automata over continuous and discrete domains.
