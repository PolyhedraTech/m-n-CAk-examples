##############################################################
# Wildfire_on_N.py
# Created on: 2021-06-30
# Author: Pau Fonseca i Casas
# Copyright: Pau Fonseca i Casas
# Description: This script simulates the spread of a wildfire using a m:n-CAk cellular automaton model over Z^2 and R^2
# All the layer share the same coordinate system, therefore, the funcrion, to change the basis is not needed.
##############################################################

import sys
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
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
import math

##############################################################
#Auxiliary functions to obtain data and to represent the data
##############################################################

# Functions to read IDRISI files
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
    Reads an IDRISI raster image file and converts its contents to a NumPy array of integers.
    The IDRISI raster file is expected to be in ASCII format, where each line represents a row of the image,
    and each value in the line represents a pixel value.
    img_path (str): The path to the IDRISI raster image file.

    Args:
        img_path (str): The path to the image file.

    Returns:
        numpy.ndarray: A NumPy array containing the image data as integers.
    """
    data = np.loadtxt(img_path).astype(int)
    return data.T  # Transpose so [x, y] matches vector convention

def read_idrisi_vector_file(file_path):
    """
    Reads an IDRISI vector file and extracts polygon data.
    The function reads a file containing polygon data in the IDRISI vector format.
    Each polygon is defined by an ID, a number of points, and the coordinates of those points.
    The file is expected to have the following structure:
    - Each polygon starts with a line containing the polygon ID and the number of points.
    - The subsequent lines contain the coordinates of the points (x, y).
    - Each polygon ends with a line containing "0 0".
    Args:
        file_path (str): The path to the IDRISI vector file.
    Returns:
        list: A list of dictionaries, each representing a polygon with the following keys:
            - 'id' (int): The polygon ID.
            - 'points' (list of tuples): A list of (x, y) tuples representing the points of the polygon.
    """
    polygons = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            # Read the polygon ID and number of points
            id_line = lines[i].strip().split()
            polygon_id = int(id_line[0])
            num_points = int(id_line[1])
            i += 1
            
            # Read the points for the polygon
            points = []
            for _ in range(num_points):
                x, y = map(float, lines[i].strip().split())
                points.append((x, y))
                i += 1
            
            # Skip the "0 0" line
            if lines[i].strip() == "0 0":
                i += 1
            
            polygons.append({'id': polygon_id, 'points': points})
    return polygons

# Functions to plot the data
def plot_vectorial(ax, polygons, id, radius=1, color='green', title='No title'):
    """
    Plots a series of polygons on a given matplotlib axis.
    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
        polygons (list of dict): A list of dictionaries, each containing:
            - 'id' (int): The identifier of the polygon.
            - 'points' (list of tuples): A list of (x, y) tuples representing the vertices of the polygon.
        id (int): The identifier for the current layer being plotted.
        radius (float, optional): The radius of the circles to be plotted at each vertex. Default is 1.
        color (str, optional): The color of the circles to be plotted at each vertex. Default is 'green'.
        title (str, optional): The title of the plot. Default is 'No title'.
    Returns:
        None
    """
    ax.clear()
    
    # Calculate the area of each polygon and sort them by area
    polygons = sorted(polygons, key=lambda p: np.abs(np.sum([x0*y1 - x1*y0 for (x0, y0), (x1, y1) in zip(p['points'], p['points'][1:] + [p['points'][0]])])) / 2)
    
    for polygon in polygons:
        polygon_id = polygon['id']
        points = polygon['points']
        
        # Plot the polygon with transparent fill and black edges
        polygon_shape = plt.Polygon(points, closed=True, edgecolor='black', facecolor='none', fill=True)
        ax.add_patch(polygon_shape)
        
        # Plot the edges of the polygon
        for j in range(len(points)):
            x1, y1 = points[j]
            x2, y2 = points[(j + 1) % len(points)]  # Connect to the next point, wrapping around
            ax.plot([x1, x2], [y1, y2], color='black')  # Draw the edge
        
        # Annotate the polygon with its ID
        centroid_x = sum(x for x, y in points) / len(points)
        centroid_y = sum(y for x, y in points) / len(points)
        ax.text(centroid_x, centroid_y, str(polygon_id), fontsize=12, ha='center', va='center', color='black')
    
    # Plot each point with a circle on top of everything
    for polygon in polygons:
        points = polygon['points']
        for (x, y) in points:
            circle = plt.Circle((x, y), radius, color='blue', fill=False)
            ax.add_patch(circle)
            ax.plot(x, y, 'ro')  # Plot the point
    
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
  
    # Define the color scale
    cmap = plt.cm.colors.ListedColormap(['black', 'red', 'green', 'blue'])
    bounds = [0, 1, 2, 3, 4]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
  
    #cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    cax = ax.imshow(matrix, cmap=cmap, norm=norm,  interpolation='nearest')
    
    # Highlight the cells with the given ID
    #highlight = np.where(matrix == id)
    #ax.scatter(highlight[1], highlight[0], color=color, label=f'ID {id}', edgecolor='black')
    
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

def create_idrisi_raster(polygons, output_filename):
    """
    Creates a raster file in IDRISI format with dimensions of 100x100 points,
    using the function find_polygon_id to determine the IDs of the polygons
    that contain each point.

    Args:
        polygons (list): A list of dictionaries, where each dictionary represents a polygon with the keys:
            - 'id' (any): The identifier of the polygon.
            - 'points' (list): A list of tuples representing the vertices of the polygon.
        output_filename (str): The base name of the output file (without extension).
    """
    # Dimensiones del raster
    width, height = 100, 100

    # Crear una matriz de 100x100 puntos
    raster = np.zeros((height, width), dtype=int)

    # Iterar sobre cada punto en la matriz
    for j in range(height):
        for i in range(width):
            point = (i, j)
            polygon_id = find_polygon_id(point, polygons)
            if polygon_id is not None:
                raster[j, i] = polygon_id

    # Guardar la matriz en un archivo en formato IDRISI
    data_filename = f"{output_filename}.img"
    metadata_filename = f"{output_filename}.doc"

    # Guardar el archivo de datos como texto
    with open(data_filename, 'w') as data_file:
        for row in raster:
            data_file.write(' '.join(map(str, row)) + '\n')


    # Crear el archivo de metadatos
    with open(metadata_filename, 'w') as metadata_file:
        metadata_file.write(f"file format : IDRISI Raster A.1\n")
        metadata_file.write(f"file title  : {output_filename}\n")
        metadata_file.write(f"data type   : integer\n")
        metadata_file.write(f"file type   : binary\n")
        metadata_file.write(f"columns     : {width}\n")
        metadata_file.write(f"rows        : {height}\n")
        metadata_file.write(f"ref. system : plane\n")
        metadata_file.write(f"ref. units  : m\n")
        metadata_file.write(f"unit dist.  : 1.0000000\n")
        metadata_file.write(f"min. X      : 0.0000000\n")
        metadata_file.write(f"max. X      : {width}\n")
        metadata_file.write(f"min. Y      : 0.0000000\n")
        metadata_file.write(f"max. Y      : {height}\n")
        metadata_file.write(f"pos'n error : unknown\n")
        metadata_file.write(f"resolution  : 1.0000000\n")
        metadata_file.write(f"min. value  : {raster.min()}\n")
        metadata_file.write(f"max. value  : {raster.max()}\n")
        metadata_file.write(f"display min : {raster.min()}\n")
        metadata_file.write(f"display max : {raster.max()}\n")
        metadata_file.write(f"value units : none\n")
        metadata_file.write(f"value error : unknown\n")
        metadata_file.write(f"flag value  : none\n")
        metadata_file.write(f"flag def'n  : none\n")

def results_window(domain, fireEvolution, vegetationEvolution, humidityEvolution):
    """
    Displays a window with a slider and radio buttons to visualize the evolution of fire, vegetation, and humidity over time.
    Args:
        domain (str): The domain type, either 'Z' for raster or other for vectorial.
        fireEvolution (list): A list of matrices or vectors representing the evolution of fire over time.
        vegetationEvolution (list): A list of matrices or vectors representing the evolution of vegetation over time.
        humidityEvolution (list): A list of matrices or vectors representing the evolution of humidity over time.
    
    The window contains:
        - A matplotlib figure to display the selected evolution state.
        - A slider to navigate through different frames of the selected evolution.
        - Radio buttons to switch between fire, vegetation, and humidity evolutions.
    """
    root = tk.Tk()
    root.title("Select Action")
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
 
    # Slider for selecting the matrix
    def on_slider_change(val, layerEvolution):
        frame = int(float(val))
        if domain == 'Z':
            plotRaster(ax, layerEvolution[frame], id=0, color='red', title=f'State at Frame {frame}')
        else:
            plot_vectorial(ax, layerEvolution[frame], id=0, radius=1, color='red', title=f'State at Frame {frame}')
        canvas.draw()

    # Label to display the current value of the slider
    slider_label = tk.Label(root, text="Steeps")
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
        if domain == 'Z':
            plotRaster(ax, fireEvolution[0], id=0, color='red', title='Fire - Initial State')
        else:
            plot_vectorial(ax, fireEvolution[0], id=0, radius=1, color='red', title='Fire - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.draw()

    def set_vegetation_evolution():
        slider.config(command=lambda val: on_slider_change(val, vegetationEvolution))
        slider_label.config(text="Vegetation")
        if domain == 'Z':
            plotRaster(ax, vegetationEvolution[0], id=0, color='green', title='Vegetation - Initial State')
        else: 
            plot_vectorial(ax, vegetationEvolution[0], id=0, radius=1, color='green', title='Vegetation - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.draw()
        
    def set_humidity_evolution():
        slider.config(command=lambda val: on_slider_change(val, humidityEvolution))
        slider_label.config(text="Humidity")
        if domain == 'Z':
            plotRaster(ax, humidityEvolution[0], id=0, color='blue', title='Humidity - Initial State')
        else:
            plot_vectorial(ax, humidityEvolution[0], id=0, radius=1, color='blue', title='Humidity - Initial State')
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

#Not used but useful for debugging
def animate_layers(layersArray, interval=500, radius=1, color='green', title='No title'):
    """
    Animates a series of layers using matplotlib.
    Args:
        layersArray (list): A list of layers, where each layer contains polygons to be animated.
        interval (int, optional): The delay between frames in milliseconds. Default is 500.
        radius (int, optional): The radius of the plotted points. Default is 1.
        color (str, optional): The color of the plotted points. Default is 'green'.
        title (str, optional): The title of the plot. Default is 'No title'.
    Returns:
        None
    """
    fig, ax = plt.subplots()
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
        plot_vectorial(ax, layer, i, radius=radius, color=color, title=title)
        return []

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(layersArray), interval=interval, blit=False, repeat=True)
    plt.show()

##############################################################
#m:n-CAk on Z specific functions
##############################################################
def evolution_function_on_Z(point,state, vegetation, humidity, new_state, new_vegetation, new_humidity, wind):
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
    #Getting the nucleous.
    nc = get_nc(point)
    i, j = nc
   
    # obtaint the wind speed
    wind_speed = combination_function_on_Z(nc, wind)

    #Getting the vicinity
    vc = get_vc_Z(point, max_dim, wind_speed)
    
    if combination_function_on_Z(nc,state) == BURNING:
        if combination_function_on_Z(nc,vegetation) == 0:
            new_state[i, j] = BURNED
        else:
            new_vegetation[i,j] -= 1
            new_LP.append([i,j])
            new_LP.extend([elems for elems in get_vc_Z(point, max_dim, wind_speed)])
    elif state[i,j] == UNBURNED:
        for point_vc in vc:
            #We muist acces information contained on other layers, therefore we will use combination funcion
            #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
            #is just returning the same point that will be used as center of the vicinity.
            i_vc, j_vc = point_vc
            if state[i_vc,j_vc] == BURNING:
                if humidity[i, j] > 0:
                    new_humidity[i, j] -= 1
                elif vegetation[i,j] > 0:
                    new_state[i, j] = BURNING
                    new_LP.append([i_vc, j_vc])
                    new_LP.extend([elems for elems in get_vc_Z(point_vc, max_dim, wind_speed)])

    return new_LP,new_state, new_vegetation, new_humidity

def event_scheduling_on_Z():
    """
    Simulates the evolution of a wildfire over a specified number of steps using event scheduling.
    This function reads vegetation and humidity data from IDRISI raster files, initializes the wildfire
    conditions, and iteratively updates the state of the wildfire, vegetation, and humidity layers.
    Returns:
        tuple: A tuple containing three lists:
            - fireEvolution: A list of numpy arrays representing the state of the wildfire at each step.
            - vegetationEvolution: A list of numpy arrays representing the state of the vegetation at each step.
            - humidityEvolution: A list of numpy arrays representing the state of the humidity at each step.
    """
    # Auxiliary functions to obtain the information of the layers from files (IDRISI 32 format).
    # vegetation layer.
    folder_path = './'
    vegetation_map_doc_path = os.path.join(folder_path, 'vegetation.doc')
    vegetation_map_img_path = os.path.join(folder_path, 'vegetation.img')

    # humidity layer
    humidity_doc_path = os.path.join(folder_path, 'humidity.doc')
    humidity_img_path = os.path.join(folder_path, 'humidity.img')

    # wind layer
    wind_doc_path = os.path.join(folder_path, 'wind.doc')
    wind_img_path = os.path.join(folder_path, 'wind.img')
    # Uncomment this if you don't want wind (validation purposes)
    #wind_img_path = os.path.join(folder_path, 'no_wind.img')

    # Reading the information of the layers
    # vegetation layer
    vegetation_data = read_idrisi_raster_file(vegetation_map_img_path)

    # humidity layer
    humidity_data = read_idrisi_raster_file(humidity_img_path)

    # wind layer
    wind_data = read_idrisi_raster_file(wind_img_path)
   
    # defining the size for the layers, the same for all.
    size = (100, 100)

    # Auxiliary functions to convert the vector in a matrix of data for both layers.
    humidity_data = humidity_data.reshape(size)
    vegetation_data = vegetation_data.reshape(size)

    # Modifying the initial conditions to start the wildfire
    initial_fire = np.full(size, UNBURNED)
    ini_point = [70, 70]
    max_dim = [100, 100]
    i, j = ini_point
    initial_fire[i, j] = BURNING
    LP = []

    # Adding the initial point we change.
    LP.append(ini_point)

    # Also adding the neighborhoods of this point.
    LP.extend([point for point in get_vc_Z(ini_point, max_dim)])

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
            LP_new, new_state, new_vegetation, new_humidity = evolution_function_on_Z(point, fireEvolution[-1], vegetationEvolution[-1], humidityEvolution[-1], new_state, new_vegetation, new_humidity, wind_data)
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]
        fireEvolution.append(new_state)
        vegetationEvolution.append(new_vegetation)
        humidityEvolution.append(new_humidity)

    return fireEvolution, vegetationEvolution, humidityEvolution

##############################################################
#m:n-CAk on R specific functions
##############################################################

# Functions to work with vectorial maps
def is_point_in_polygon(point, polygon_points):
    """
    Determine if a given point is inside a polygon defined by a list of points.
    Args:
        point (tuple): A tuple representing the coordinates of the point (x, y).
        polygon_points (list): A list of tuples where each tuple represents the coordinates of a vertex of the polygon.
    Returns:
        bool: True if the point is inside the polygon or on its boundary, False otherwise.
    Special Cases:
        - If `polygon_points` contains only one point, the function checks if the given point is the same as that single point.
    """
    # Handle the special case where polygon_points is a single point
    if len(polygon_points) == 1:
        return point == polygon_points[0]
    
    #handle the special case where polygon_points is a line segment
    if len(polygon_points) == 2:
        # Create a line segment from the two points
        line_segment = LineString(polygon_points)
        # Create a point from the given coordinates
        point_geom = Point(point)
        # Check if the point intersects with the line segment
        return line_segment.intersects(point_geom)

    # Create a polygon from the list of points
    polygon = Polygon(polygon_points)
    
    # Check if the polygon contains the point
    return polygon.intersects(Point(point))

def find_polygon_id(point, polygons):
    """
    Finds the ID of the smallest polygon that contains a given point.
    Args:
        point (tuple): A tuple representing the coordinates of the point (x, y).
        polygons (list): A list of dictionaries, where each dictionary represents a polygon with the keys:
            - 'id' (any): The identifier of the polygon.
            - 'points' (list): A list of tuples representing the vertices of the polygon.
    Returns:
        any: The ID of the smallest polygon that contains the point. If no polygon contains the point, returns None.
    """

    i, j = point

    if i == 70 and j == 70:
        a=1

    smallest_polygon_id = None
    smallest_area = float('inf')
    
    for polygon in polygons:
        if is_point_in_polygon(point, polygon['points']):
            if len(polygon['points']) == 1: #if the poligon is a point
                current_area = 0
            elif len(polygon['points']) == 2: #if the poligon is a line segment
                current_area = 0
            else:
                current_polygon = Polygon(polygon['points'])
                current_area = current_polygon.area
            
            if current_area < smallest_area:
                smallest_area = current_area
                smallest_polygon_id = polygon['id']
    
    return smallest_polygon_id

def addVectorialMap(vectorialMap, layersArray):
    """
    Adds a vectorial map to the layers array, ensuring no duplicate points.
    This function checks if the given vectorial map is a single point. If so, it searches for and removes any existing 
    identical point in the layers array. After that, it adds the vectorial map to the layers array if it is not already present.
    Args:
        vectorialMap (list): A list containing a single dictionary with a 'points' key, which is a list of points.
        layersArray (list): A list of vectorial maps, where each vectorial map is a list containing a dictionary with a 'points' key.
    Returns:
        None
    """
    # Verificar si vectorialMap es un punto
    if len(vectorialMap) == 1 and isinstance(vectorialMap[0], dict) and len(vectorialMap[0]['points']) == 1:
        point_to_add = vectorialMap[0]['points'][0]
        
        # Buscar y eliminar el punto en layersArray si existe
        for layer in layersArray:
            if len(layer) == 1 and isinstance(layer[0], dict) and len(layer[0]['points']) == 1:
                existing_point = layer[0]['points'][0]
                if existing_point == point_to_add:
                    layersArray.remove(layer)
                    break
    
    # Añadir el vectorialMap a layersArray
    if vectorialMap not in layersArray:
        layersArray.append(vectorialMap)

def sort_points(points):
    """
    Sorts a list of points in a 2D plane based on their angle with respect to the centroid of the polygon formed by the points.
    We do not want intersections in the edges of the polygon.
    Args:
        points (list of tuple): A list of tuples where each tuple represents a point (x, y) in a 2D plane.
    Returns:
        list of tuple: A list of points sorted by their angle with respect to the centroid of the polygon.
    """
    # Calculate the centroid of the polygon
    centroid_x = sum(x for x, y in points) / len(points)
    centroid_y = sum(y for x, y in points) / len(points)
    
    # Sort points based on the angle with respect to the centroid
    def angle_from_centroid(point):
        x, y = point
        return math.atan2(y - centroid_y, x - centroid_x)
    
    return sorted(points, key=angle_from_centroid)
    
def simplifyVectorialMap(vectorialMap):
    """
    Simplifies a vectorial map by merging polygons that are within a certain distance (defined by the vicinity) of each other.
    Args:
        vectorialMap (list): A list of dictionaries, where each dictionary represents a polygon with an 'id' and 'points'.
                             The 'id' is a unique identifier for the polygon, and 'points' is a list of coordinates.
    Returns:
        list: A simplified vectorial map, where polygons that are close to each other are merged. Each element in the 
              returned list is a dictionary with an 'id' and 'points'. The 'points' represent the exterior coordinates 
              of the merged polygons.
    """
    # Agrupar polígonos por ID
    polygons_by_id = {}
    for polygon in vectorialMap:
        poly_id = polygon['id']
        if poly_id not in polygons_by_id:
            polygons_by_id[poly_id] = []
        if len(polygon['points']) == 1:
            polygons_by_id[poly_id].append(Point(polygon['points'][0]))
        elif len(polygon['points']) == 2:
            polygons_by_id[poly_id].append(Point(polygon['points'][0]))
            polygons_by_id[poly_id].append(Point(polygon['points'][1]))
        else:
            polygons_by_id[poly_id].append(Polygon(polygon['points']))
    
    simplified_vectorial_map = []
    
    # Unir polígonos que cumplen con el criterio de distancia
    for poly_id, polygons in polygons_by_id.items():
        merged_polygons = []
        while polygons:
            base_polygon = polygons.pop(0)
            to_merge = [base_polygon]
            for other_polygon in polygons[:]:
                if base_polygon.distance(other_polygon) <= 1:
                    to_merge.append(other_polygon)
                    polygons.remove(other_polygon)
            merged_polygon = unary_union(to_merge)
            merged_polygons.append(merged_polygon)
        
        # Añadir los polígonos unidos al nuevo vectorialMap
        for merged_polygon in merged_polygons:
            if isinstance(merged_polygon, MultiPolygon):
                for poly in merged_polygon:
                    simplified_vectorial_map.append({'id': poly_id, 'points': list(poly.exterior.coords)})
            elif isinstance(merged_polygon, Polygon):
                simplified_vectorial_map.append({'id': poly_id, 'points': list(merged_polygon.exterior.coords)})
            elif isinstance(merged_polygon, Point):
                simplified_vectorial_map.append({'id': poly_id, 'points': [(merged_polygon.x, merged_polygon.y)]})
            else:
                # Manejar el caso donde merged_polygon es una colección de puntos
                points = []
                for geom in merged_polygon.geoms:
                    if isinstance(geom, Point):
                        #simplified_vectorial_map.append({'id': poly_id, 'points': [(geom.x, geom.y)]})
                        points.append((geom.x, geom.y))
                    elif isinstance(geom, Polygon):
                        #simplified_vectorial_map.append({'id': poly_id, 'points': list(geom.exterior.coords)})
                        points.extend(list(geom.exterior.coords))
                # Create a polygon from the points. To assure not add interior points.
                points_ext = remove_interior_points(points)
                points_ext = sort_points(points_ext)
                simplified_vectorial_map.append({'id': poly_id, 'points': points_ext})

    return simplified_vectorial_map

def remove_interior_points(points):
    """
    Removes interior and duplicate points from a list of points.
    A point is considered interior if there are points above, below, 
    to the left, and to the right of it.
    Args:
        points (list of tuple): A list of points where each point is represented 
                                as a tuple (x, y).
    Returns:
        list of tuple: A list of points with all interior and duplicate points removed.
    """
    def is_interior(point, points_set):
        x, y = point
        above = below = left = right = False
        for px, py in points_set:
            if px == x and py > y:
                above = True
            elif px == x and py < y:
                below = True
            elif py == y and px > x:
                right = True
            elif py == y and px < x:
                left = True
            if above and below and left and right:
                return True
        return False

    #points_set = set(points)
    #result = [point for point in points if not is_interior(point, points_set)]

    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for pt in points:
        if pt not in seen:
            unique_points.append(pt)
            seen.add(pt)

    points_set = set(unique_points)
    result = [point for point in unique_points if not is_interior(point, points_set)]
    return result

#m:n-CAk on R functions
def evolution_function_on_R(point,fire, vegetation, humidity, new_state, new_vegetation, new_humidity, wind):
    """
    Simulates the evolution of a wildfire on a grid based on the current state of fire, vegetation, and humidity.
    Args:
        point (tuple): The coordinates (i, j) of the current cell.
        fire (list): The current state of the fire grid.
        vegetation (list): The current state of the vegetation grid.
        humidity (list): The current state of the humidity grid.
        new_state (list): The updated state of the fire grid.
        new_vegetation (list): The updated state of the vegetation grid.
        new_humidity (list): The updated state of the humidity grid.
    Returns:
        tuple: A tuple containing:
            - new_LP (list): List of points that have been updated.
            - new_state (list): The updated state of the fire grid.
            - new_vegetation (list): The updated state of the vegetation grid.
            - new_humidity (list): The updated state of the humidity grid.
    """
    new_LP = []
    nc = []
    vc = []
    max_dim = [100,100]

    #Getting the nucleous.
    nc = get_nc(point)
    i, j = nc

    if i == 70 and j == 70:
        a=1

    #We must acces information contained on other layers, therefore we will use combination funcion
    #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
    #is just returning the point.

    #we access to the wind layer that is on Z to obtain the wind speed at this point.
    #This is a point, therefore we can use the combination function on Z.
    wind_speed = combination_function_on_Z(nc,wind)

    #Getting the vicinity, it can be defined also over the nucleous.
    vc = get_vc_R(point, max_dim, wind_speed)

    cell_state = combination_function_on_R(nc,fire) 
    if cell_state == BURNING:
        value_veg = combination_function_on_R(nc,vegetation)
        if value_veg == 0:
            points = []
            points.append((i, j))
            addVectorialMap({'id': BURNED, 'points': points},new_state)
        else:
            points = []
            value_veg -= 1
            points.append((i, j))
            addVectorialMap({'id': value_veg, 'points': points},new_vegetation)
            new_LP.append((i,j))
            new_LP.extend([elems for elems in get_vc_R(point, max_dim, wind_speed)])
    elif cell_state == UNBURNED:
        for point_vc in vc:
            #We must acces information contained on other layers, therefore we will use combination funcion
            #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
            #is just returning the point.
            i_vc, j_vc = point_vc
            cell_state = combination_function_on_R(point_vc,fire)
            cell_humidity =  combination_function_on_R(nc,humidity)
            cell_vegetation =  combination_function_on_R(nc,vegetation) 
            if cell_state  == BURNING:
                if cell_humidity > 0:
                    cell_humidity -= 1
                    points = []
                    points.append((i, j))
                    addVectorialMap({'id': cell_humidity, 'points': points},new_humidity)
                elif cell_vegetation > 0:
                    points = []
                    points.append((i, j))
                    addVectorialMap({'id': BURNING, 'points': points},new_state)
                    new_LP.append((i_vc, j_vc))
                    new_LP.extend([elems for elems in get_vc_R(point_vc, max_dim, wind_speed)])

    return new_LP,new_state, new_vegetation, new_humidity

def event_scheduling_on_R():
    """
    Simulates the evolution of a wildfire over a specified number of steps.
    This function reads initial conditions from IDRISI vector files for vegetation, humidity, 
    and the initial fire starting points. It then simulates the spread of the wildfire over 
    a number of steps, updating the state of the fire, vegetation, and humidity at each step.
    Returns:
        tuple: A tuple containing three lists:
            - fireEvolution: A list of states representing the evolution of the fire.
            - vegetationEvolution: A list of states representing the evolution of the vegetation.
            - humidityEvolution: A list of states representing the evolution of the humidity.
    """
    folder_path = './'
    size = (100, 100)

    # Reading vegetation and humidity layers
    fileVegetation = 'vegetation.vec'
    polygonsVegetation = read_idrisi_vector_file(fileVegetation)
    polygonsVegetation = transpose_vectorial_polygons(polygonsVegetation)

    fileHumidity = 'humidity.vec'
    polygonsHumidity = read_idrisi_vector_file(fileHumidity)
    polygonsHumidity = transpose_vectorial_polygons(polygonsHumidity)

    # Create raster versions for compatibility
    create_idrisi_raster(polygonsVegetation, 'vegetation')
    create_idrisi_raster(polygonsHumidity, 'humidity')

    # Wind layer (raster)
    wind_doc_path = os.path.join(folder_path, 'wind.doc')
    wind_img_path = os.path.join(folder_path, 'wind.img')
    # Uncomment this if you don't want wind (validation purposes)
    #wind_img_path = os.path.join(folder_path, 'no_wind.img')

    wind_data = read_idrisi_raster_file(wind_img_path)
    wind_data = transpose_raster_matrix(wind_data)

    # Reading the wildfire starting point
    fileFire = 'fire.vec'
    polygonsFire = read_idrisi_vector_file(fileFire)
    polygonsFire = transpose_vectorial_polygons(polygonsFire)

    max_dim = [100, 100]
    LP = []

    for polygon in polygonsFire:
        polygon_id = polygon['id']
        points = polygon['points']
        for (x, y) in points:
            ini_point = (x, y)
            LP.append(ini_point)
            LP.extend([point for point in get_vc_Z(ini_point, max_dim)])

    fireEvolution = [polygonsFire]
    vegetationEvolution = [polygonsVegetation]
    humidityEvolution = [polygonsHumidity]

    n_steps = 100

    for _ in range(n_steps):
        LP_rep = []
        LP_new = []
        new_state = fireEvolution[-1].copy()
        new_vegetation = vegetationEvolution[-1].copy()
        new_humidity = humidityEvolution[-1].copy()

        for point in LP:
            LP_new, new_state, new_vegetation, new_humidity = evolution_function_on_R(
                point, fireEvolution[-1], vegetationEvolution[-1], humidityEvolution[-1],
                new_state, new_vegetation, new_humidity, wind_data
            )
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]

        fireEvolution.append(simplifyVectorialMap(new_state))
        vegetationEvolution.append(simplifyVectorialMap(new_vegetation))
        humidityEvolution.append(simplifyVectorialMap(new_humidity))

    return fireEvolution, vegetationEvolution, humidityEvolution

##############################################################
#m:n-CAk main functions
##############################################################

def get_vc_Z(point, max_dim, wind_speed=0):
    """
    Vicinity function. Get the valid coordinates (Moore neighbourhood) adjacent to a given point within a specified maximum dimension.
    The same for Z^2 and R^2 in this example.
    Args:
        point (tuple): A tuple (i, j) representing the coordinates of the point.
        max_dim (tuple): A tuple (max_i, max_j) representing the maximum dimensions of the grid.
    Returns:
        list: A list of tuples representing the valid adjacent coordinates.
    """
    vc= []
    i, j = point
    max_i, max_j = max_dim

    if i > 0: vc.append((i-1,j))
    if i < max_i - 1: vc.append((i+1, j))
    if j > 0:  vc.append((i, j-1))
    if j < max_j - 1: vc.append((i, j+1))

    #consider the case of wind
    wind_speed = wind_speed*1.5  # Assuming wind_speed is a multiplier for the number of steps to consider
    # Add the wind effect by extending the vicinity in the direction of the wind
    while wind_speed > 0: 
        j=j+1
        if j < max_j - 1: vc.append((i, j+1))
        wind_speed -= 1

    return vc

def get_vc_R(point, max_dim, wind_speed=0):
    """
    Vicinity function. Get the valid coordinates (Moore neighbourhood) adjacent to a given point within a specified maximum dimension.
    The same for Z^2 and R^2 in this example.
    Args:
        point (tuple): A tuple (i, j) representing the coordinates of the point.
        max_dim (tuple): A tuple (max_i, max_j) representing the maximum dimensions of the grid.
    Returns:
        list: A list of tuples representing the valid adjacent coordinates.
    """
    vc= []
    i, j = point
    max_i, max_j = max_dim

    if i > 0: vc.append((i-1,j))
    if i < max_i - 1: vc.append((i+1, j))
    if j > 0:  vc.append((i, j-1))
    if j < max_j - 1: vc.append((i, j+1))

    #consider the case of wind
    wind_speed = wind_speed*1.5  # Assuming wind_speed is a multiplier for the number of steps to consider
    # Add the wind effect by extending the vicinity in the direction of the wind
    # we extend j because on numpy the first coordinate is the y coordinate and the second is the x coordinate.
    if wind_speed > 0: 
        if i+wind_speed < max_i - 1: vc.append((i+wind_speed,j))

    return vc

def get_nc(point):
    """
    Nucleous function. Returns the input point without any modifications as a nucleous.
    The same for Z^2 and R^2 in this example.
    
    Args:
        point (any): The input point to be returned.

    Returns:
        any: The same input point as nucleous.
    """
    return point

def set_nc(point, layer, value):
    """
    Nucleous function. Sets the value for the nucleous.
    The same for Z^2 and R^2 in this example.

    Args:
        point (any): The input point to be returned.

    Returns:
        any: The same input point as nucleous.
    """
    x, y = point
    layer[x, y] = value
    return point

def combination_function_on_R(point, layer):
    """
    Retrieves the value from the specified layer at the given point coordinates.
    All the layers in this example share the same coordinates, so the E function is the identity function.
    To simplify the example implementation we define just a single combination_function that uses a a parameter the layer.
    Determines the polygon ID for a given point within a specified layer.
    
    Args:
        point (tuple): A tuple representing the coordinates of the point (e.g., (x, y)).
        layer (object): The layer object containing polygon data.
    Returns:
        int: The ID of the polygon that contains the point.
    """

    return find_polygon_id(point, layer)

def combination_function_on_Z(point, layer):
    """
    Retrieves the value from the specified layer at the given point coordinates.
    All the layers in this example share the same coordinates, so the E function is the identity function.
    To simplify the example implementation we define just a single combination_function that uses a a parameter the layer.

    Args:
        point (tuple): A tuple containing the coordinates (i, j) of the point.
        layer (numpy.ndarray): A 2D array representing the layer from which the value is to be retrieved.

    Returns:
        The value from the layer at the specified point coordinates.
    """
    i, j = point
    # Check if the values i and j are integers, if not use the integer part
    if not isinstance(i, int):
        i = int(i)
    if not isinstance(j, int):
        j = int(j)
    # Ensure i and j are within the bounds of the layer
    if i < 0 or i >= layer.shape[0] or j < 0 or j >= layer.shape[1]:
        #return the maximum value of the layer if out of bounds
        return 0

    return layer[i,j]
    #return point

def transpose_vectorial_polygons(polygons):
    """
    Transposes the coordinates of all points in all polygons from (x, y) to (y, x).
    Args:
        polygons (list): List of polygon dictionaries with 'points' as (x, y) tuples.
    Returns:
        list: List of polygons with transposed points.
    """
    transposed = []
    for poly in polygons:
        new_points = [(y, x) for (x, y) in poly['points']]
        transposed.append({'id': poly['id'], 'points': new_points})
    return transposed

def transpose_raster_matrix(matrix):
    """
    Transposes a raster matrix so that coordinates (x, y) become (y, x).
    Args:
        matrix (numpy.ndarray): The raster matrix to transpose.
    Returns:
        numpy.ndarray: The transposed raster matrix.
    """
    return matrix.T

##############################################################
#Main
##############################################################

if __name__ == "__main__":

    # Definition of S_fire for fire main layer.
    UNBURNED = 2
    BURNING = 1
    BURNED = 0

    domain = 'Z'

    domain = input(".git\nPlease, select the domain to work on (Z or R): ")
    if domain != 'Z' and domain != 'R':
        print("Invalid domain. Please select 'Z' or 'R'.")
    else:
        fireEvolution = []
        vegetationEvolution = []
        humidityEvolution = []

        if domain == 'Z':
            fireEvolution, vegetationEvolution, humidityEvolution = event_scheduling_on_Z()
        else:
            fireEvolution, vegetationEvolution, humidityEvolution = event_scheduling_on_R()

        results_window(domain, fireEvolution, vegetationEvolution, humidityEvolution)