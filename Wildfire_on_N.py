##############################################################
# Wildfire_on_N.py
# Created on: 2021-06-30
# Author: Pau Fonseca i Casas
# Copyright: Pau Fonseca i Casas
# Description: This script simulates the spread of a wildfire using a m:n-CAk cellular automaton model over Z^2.
##############################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

##############################################################
#Auxiliary functions to obtain data and to represent the data
##############################################################

def read_idrisi_doc_file(doc_path):
    """
    Reads a document file (IDRISI) and extracts metadata from it.

    The document file is expected to have lines in the format "key: value".
    Each line is split into a key and a value, which are then stored in a dictionary.

    Args:
        doc_path (str): The path to the document file.

    Returns:
        dict: A dictionary containing the metadata extracted from the document file.
    """
    metadata = {}
    with open(doc_path, 'r') as doc_file:
        for line in doc_file:
            key, value = line.strip().split(': ')
            metadata[key.strip()] = value.strip()
    return metadata

def read_idrisi_raster_file(img_path):
    """
    Reads an image file (IDRISI) and converts its contents to a NumPy array of integers.

    Parameters:
    img_path (str): The path to the image file.

    Returns:
    numpy.ndarray: A NumPy array containing the image data as integers.
    """
    data = np.loadtxt(img_path).astype(int)
    return data

def animateLayers(fig, ax, raster, layersArray, interval=500, radius=1, color='green', title='No title'):
    #fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.grid(True)

    def init():
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(title)
        plt.grid(True)
        return []

    def animate(i):
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Layer {i + 1}'+title)
        plt.grid(True)
        
        # Get the current layer
        layer = layersArray[i % len(layersArray)]
        # Plot all polygons in the current layer
        if raster==True:
            plotRaster(ax, layer, i, color=color, title=title)
        else:
            plotVectorial(ax, layer, i, radius=radius, color=color, title=title)
        return []

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(layersArray), interval=interval, blit=False, repeat=True)
    plt.show()

def plotVectorial(ax, polygons, id, radius=1, color='green', title='No title'):
    ax.clear()
    
    for polygon in polygons:
        polygon_id = polygon['id']
        points = polygon['points']
        
        # Plot each point with a circle
        for (x, y) in points:
            circle = plt.Circle((x, y), radius, color='blue', fill=False)
            ax.add_patch(circle)
            ax.plot(x, y, 'ro')  # Plot the point
        
        # Plot the polygon with green fill
        polygon_shape = plt.Polygon(points, closed=True, edgecolor=color, facecolor=color, fill=True)
        ax.add_patch(polygon_shape)
        
        # Annotate the polygon with its ID
        centroid_x = sum(x for x, y in points) / len(points)
        centroid_y = sum(y for x, y in points) / len(points)
        ax.text(centroid_x, centroid_y, str(polygon_id), fontsize=12, ha='center', va='center', color='black')
    
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Layer {id} - '+title)
    plt.grid(True)

def plotRaster(ax, matrix, id, color='green', title='No title'):
    """
    Plots a matrix using matplotlib on a given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        matrix (numpy.ndarray): The matrix to plot.
        id (int): The ID to highlight in the plot.
        color (str): The color to use for highlighting the ID.
        title (str): The title of the plot.
    """
    ax.clear()  # Clear the axis to prepare for the new frame
    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    
    # Highlight the cells with the given ID
    highlight = np.where(matrix == id)
    ax.scatter(highlight[1], highlight[0], color=color, label=f'ID {id}', edgecolor='black')
    
    # Add a colorbar
    #plt.colorbar(cax, ax=ax, label='Value')
    
    # Add title and labels
    ax.set_title(f'Layer {id} - '+title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Invert the y-axis to have 0 at the bottom
    ax.invert_yaxis()

    # Add legend
    #ax.legend()

def resultsWindow(fireEvolution, vegetationEvolution, humidityEvolution):
    root = tk.Tk()
    root.title("Select Action")
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
 
    # Slider for selecting the matrix
    def on_slider_change(val, layerEvolution):
        frame = int(float(val))
        plotRaster(ax, layerEvolution[frame], id=0, color='red', title=f'State at Frame {frame}')
        canvas.draw()

    # Label to display the current value of the slider
    slider_label = tk.Label(root, text="Steep")
    slider_label.pack(side=tk.BOTTOM, pady=5)

    # Slider for selecting the matrix
    slider = ttk.Scale(root, from_=0, to=len(fireEvolution) - 1, orient=tk.HORIZONTAL, command=lambda val: on_slider_change(val, fireEvolution))
    slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
     
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=20)

    # Buttons to change the layerEvolution
    def set_fire_evolution():
        slider.config(command=lambda val: on_slider_change(val, fireEvolution))
        slider_label.config(text="Fire")
        plotRaster(ax, fireEvolution[0], id=0, color='red', title='Fire - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.draw()

    def set_vegetation_evolution():
        slider.config(command=lambda val: on_slider_change(val, vegetationEvolution))
        slider_label.config(text="Vegetation")
        plotRaster(ax, vegetationEvolution[0], id=0, color='green', title='Vegetation - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.draw()
        
    def set_humidity_evolution():
        slider.config(command=lambda val: on_slider_change(val, humidityEvolution))
        slider_label.config(text="Humidity")
        plotRaster(ax, humidityEvolution[0], id=0, color='blue', title='Humidity - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.draw()

    # Variable to keep track of the selected option
    selected_option = tk.StringVar(value="Fire")

 # Radio buttons to select the layerEvolution
    radio_fire = ttk.Radiobutton(button_frame, text="Fire", variable=selected_option, value="Fire", command=set_fire_evolution)
    radio_fire.pack(side=tk.LEFT, padx=10)

    radio_vegetation = ttk.Radiobutton(button_frame, text="Vegetation", variable=selected_option, value="Vegetation", command=set_vegetation_evolution)
    radio_vegetation.pack(side=tk.LEFT, padx=10)

    radio_humidity = ttk.Radiobutton(button_frame, text="Humidity", variable=selected_option, value="Humidity", command=set_humidity_evolution)
    radio_humidity.pack(side=tk.LEFT, padx=10)

    root.mainloop()
##############################################################
#m:n-CAk main functions
##############################################################

#Naif Evolution function, propagation of the wildfire.

def Naif_Evolution_function(state, vegetation, humidity):
    new_state = state.copy()
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            #This is what must be part of the evolution function.
            if state[i, j] == BURNING:  # Si la cel·la està burning
                if vegetation[i,j] == 0:
                    new_state[i, j] = BURNED
                else:
                    vegetation[i,j] -= 1
                    if i > 0 and state[i-1,j] == UNBURNED:
                        if humidity[i-1, j] > 0:
                            humidity[i-1, j] -= 1
                        elif vegetation[i-1,j] > 0:
                            new_state[i-1, j] = BURNING 
                    if i < state.shape[0] - 1 and state[i+1, j] == UNBURNED:
                        if humidity[i+1, j] > 0:
                            humidity[i+1, j] -= 1
                        elif vegetation[i+1,j] > 0:
                            new_state[i+1, j] = BURNING 
                    if j > 0 and state[i, j-1] == UNBURNED:
                        if humidity[i, j-1] > 0:
                            humidity[i, j-1] -= 1
                        elif vegetation[i,j-1] > 0:
                            new_state[i, j-1] = BURNING 
                    if j < state.shape[1] - 1 and state[i, j+1] == UNBURNED:
                        if humidity[i, j+1] > 0:
                            humidity[i, j+1] -= 1
                        elif vegetation[i,j+1] > 0:    
                            new_state[i, j+1] = BURNING

    return new_state, vegetation, humidity

def get_vc(point, max_dim):
    """
    Vicinity function. Get the valid coordinates (Moore neighbourhood on N^2) adjacent to a given point within a specified maximum dimension.
    Args:
        point (tuple): A tuple (i, j) representing the coordinates of the point.
        max_dim (tuple): A tuple (max_i, max_j) representing the maximum dimensions of the grid.
    Returns:
        list: A list of tuples representing the valid adjacent coordinates.
    """
    vc= []
    i, j = point
    max_i, max_j = max_dim

    if i > 0: vc.append([i-1,j])
    if i < max_i - 1: vc.append([i+1, j])
    if j > 0:  vc.append([i, j-1])
    if j < max_j - 1: vc.append([i, j+1])

    return vc

def get_nc(point):
    """
    Nucleous function. Returns the input point without any modifications as a nucleous.

    Parameters:
    point (any): The input point to be returned.

    Returns:
    any: The same input point as nucleous.
    """
    return point

def combination_function(point):
    """
    Combination function. Processes a given point and returns it as a combination function.

    Args:
        point: The input point to be processed.

    Returns:
        The same point that was passed as input.
    """
    return point

def Evolution_function(point,state, vegetation, humidity, new_state, new_vegetation, new_humidity):
    def Evolution_function(point, state, vegetation, humidity, new_state, new_vegetation, new_humidity):
        """
        Simulates the evolution of a wildfire in a m:n-CAk cellular automaton model.
        Args:
            point (tuple): The coordinates (i, j) of the current cell.
            state (ndarray): The current state of the grid, where each cell can be BURNING, UNBURNED, or BURNED.
            vegetation (ndarray): The current vegetation levels of the grid.
            humidity (ndarray): The current humidity levels of the grid.
            new_state (ndarray): The state of the grid for the next time step.
            new_vegetation (ndarray): The vegetation levels of the grid for the next time step.
            new_humidity (ndarray): The humidity levels of the grid for the next time step.
        Returns:
            tuple: A tuple containing:
                - new_LP (list): List of points that have changed state.
                - new_state (ndarray): Updated state of the grid.
                - new_vegetation (ndarray): Updated vegetation levels of the grid.
                - new_humidity (ndarray): Updated humidity levels of the grid.
        """
    new_LP = []
    nc = []
    vc = []
    max_dim = [100,100]
    #3Això FORMA PART de la funció d'evolució
    #Getting the nucleous.
    nc = get_nc(point)
    i, j = point
   
    #Getting the vicinity
    vc = get_vc(point, max_dim)
    if state[i, j] == BURNING:  # Si la cel·la està burning
        if vegetation[i,j] == 0:
            new_state[i, j] = BURNED
        else:
            new_vegetation[i,j] -= 1
            new_LP.append([i,j])
            new_LP.extend([elems for elems in get_vc(point, max_dim)])
    elif state[i,j] == UNBURNED:
        for point_vc in vc:
            #We muist acces information contained on other layers, therefore we will use combination funcion
            #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
            #is just returning the point.
            i_vc, j_vc = combination_function(point_vc)
            if state[i_vc,j_vc] == BURNING:
                if humidity[i, j] > 0:
                    new_humidity[i, j] -= 1
                elif vegetation[i,j] > 0:
                    new_state[i, j] = BURNING
                    new_LP.append([i_vc, j_vc])
                    new_LP.extend([elems for elems in get_vc(point_vc, max_dim)])

    return new_LP,new_state, new_vegetation, new_humidity

def wildfire_on_Z():
    # Auxiliary functions to obtain the information of the layers from files (IDRISI 32 format).
    # vegetation layer.
    folder_path = './'
    vegetation_map_doc_path = os.path.join(folder_path, 'vegetation_map2.doc')
    vegetation_map_img_path = os.path.join(folder_path, 'vegetation_map2.img')

    # humidity layer
    humidity_doc_path = os.path.join(folder_path, 'humidity_map2.doc')
    humidity_img_path = os.path.join(folder_path, 'humidity_map2.img')

    # Reading the information of the layers
    # vegetation layer
    vegetation_data = read_idrisi_raster_file(vegetation_map_img_path)

    # humidity layer
    humidity_data = read_idrisi_raster_file(humidity_img_path)

    # defining the size for the layers, the same for all.
    size = (100, 100)

    # Auxiliary functions to convert the vector in a matrix of data for both layers.
    humidity_data = humidity_data.reshape(size)
    vegetation_data = vegetation_data.reshape(size)

    # Modifying the initial conditions to start the wildfire
    initial_fire = np.zeros(size)
    ini_point = [70, 70]
    max_dim = [100, 100]
    i, j = ini_point
    initial_fire[i, j] = BURNING
    LP = []

    # Adding the initial point we change.
    LP.append(ini_point)

    # Also adding the neighborhoods of this point.
    LP.extend([point for point in get_vc(ini_point, max_dim)])

    # Variable that will contain all the states we define on the execution of the model.
    fireEvolution = [initial_fire]
    vegetationEvolution = [vegetation_data]
    humidityEvolution = [humidity_data]

    # Number of steps to execute the evolution function
    n_steps = 100

    # n_steps represents the finish condition of the simulation.
    for _ in range(n_steps):
        LP_rep = []
        LP_new = []
        new_state = fireEvolution[-1].copy()
        new_vegetation = vegetationEvolution[-1].copy()
        new_humidity = humidityEvolution[-1].copy()
        # Event Scheduling simulation engine, where LP is the event list.
        for point in LP:
            LP_new, new_state, new_vegetation, new_humidity = Evolution_function(point, fireEvolution[-1], vegetationEvolution[-1], humidityEvolution[-1], new_state, new_vegetation, new_humidity)
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]
        fireEvolution.append(new_state)
        vegetationEvolution.append(new_vegetation)
        humidityEvolution.append(new_humidity)

    return fireEvolution, vegetationEvolution, humidityEvolution

if __name__ == "__main__":

    # Definition of the function E, the interpretation for the layer state, the main layer.
    UNBURNED = 0
    BURNING = 1
    BURNED = 2

    fireEvolution = []
    vegetationEvolution = []
    humidityEvolution = []

    fireEvolution, vegetationEvolution, humidityEvolution = wildfire_on_Z()

    resultsWindow(fireEvolution, vegetationEvolution, humidityEvolution)