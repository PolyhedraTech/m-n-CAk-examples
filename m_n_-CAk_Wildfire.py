##############################################################
# Wildfire_on_N.py
# Created on: 2021-06-30
# Author: Pau Fonseca i Casas
# Copyright: Pau Fonseca i Casas
# Description: This script simulates the spread of a wildfire using a m:n-CAk cellular automaton model over Z^2 and R^2
# All the layer share the same coordinate system, therefore, the funcrion, to change the basis is not needed.
##############################################################

import sys
from xmlrpc.client import boolean
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
# Constants and Global Variables
##############################################################

# Fire state constants
UNBURNED = 2
BURNING = 1
BURNED = 0

# Global structures to organize points by ID and layer type
STATE_POINTS_BY_ID = {}      # {'0': [(x1,y1), (x2,y2), ...], '1': [...], '2': [...]}
HUMIDITY_POINTS_BY_ID = {}   # {'id1': [(x1,y1), ...], 'id2': [...], ...}
VEGETATION_POINTS_BY_ID = {} # {'id1': [(x1,y1), ...], 'id2': [...], ...}

STATE_POINTS_BY_ID_FINAL = {}      # {'0': [(x1,y1), (x2,y2), ...], '1': [...], '2': [...]}
HUMIDITY_POINTS_BY_ID_FINAL = {}   # {'id1': [(x1,y1), ...], 'id2': [...], ...}
VEGETATION_POINTS_BY_ID_FINAL = {} # {'id1': [(x1,y1), ...], 'id2': [...], ...}

#DEBUG Variables
DEBUG = False  # Set to True to enable debugging output
D_i = 70  # i component of the point to debug
D_j = 69  # j component of the point to debug
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
    return data  # No transpose; keep original orientation

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
def plot_vectorial(ax, polygons, id, radius=1, color='green', title='No title', exclusive_plot=[]):
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
        exclusive_plot (list, optional): List of polygon IDs to show exclusively. If empty (default), shows ALL polygons including ID 0. 
                                         If contains specific IDs, shows only those polygons. Default is [].
    Returns:
        None
    """
    ax.clear()
    
    # Calculate the area of each polygon and sort them by area
    polygons = sorted(polygons, key=lambda p: np.abs(np.sum([x0*y1 - x1*y0 for (x0, y0), (x1, y1) in zip(p['points'], p['points'][1:] + [p['points'][0]])])) / 2)
    
    for polygon in polygons:
        polygon_id = polygon['id']
        points = polygon['points']
        
        # Filter polygons based on exclusive_plot list or default behavior
        if exclusive_plot is not None and len(exclusive_plot) > 0:
            # If exclusive_plot is provided with specific IDs, only show polygons with IDs in the list
            if polygon_id not in exclusive_plot:
                continue
        elif exclusive_plot == [] or exclusive_plot is None:
            # Default behavior when exclusive_plot is empty: show ALL polygons including ID 0
            pass  # Don't filter anything - show all polygons
        else:
            # Fallback: show all polygons
            pass
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
        
        # Plot each point with a circle on top of everything (only for displayed polygons)
        circle = False
        if circle:
            for (x, y) in points:
                circle = plt.Circle((x, y), radius, color='blue', fill=False)
                ax.add_patch(circle)
                ax.plot(x, y, 'ro')  # Plot the point
        
    ax.set_aspect('equal', adjustable='box')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Layer {id} - '+title)
    plt.grid(True)

def plot_raster(ax, matrix, id, type="fire", color='green', title='No title'):
    """
    Plots a matrix using matplotlib on a given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        matrix (numpy.ndarray): The matrix to plot.
        id (int): The ID to highlight in the plot.
        color (str): The color to use for highlighting the ID.
        title (str): The title of the plot.
    """

    # Limpiar el eje y eliminar leyendas y títulos previos
    ax.clear()
    # Eliminar leyenda previa si existe
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # Eliminar título previo (set a string vacío)
    ax.set_title("")

    # Definir el color y la normalización según el tipo
    from matplotlib.patches import Patch
    if type == "fire":
        cmap = plt.cm.colors.ListedColormap(['black', 'red', 'green'])
        bounds = [0, 1, 2, 3]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    elif type == "vegetation":
        cmap = plt.cm.colors.ListedColormap(['black', (0.5, 1, 0.5), (0, 0.5, 0), (0, 1, 0)])
        bounds = [0, 1, 2, 3, 21]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    elif type == "humidity":
        cmap = plt.cm.colors.ListedColormap(['black', (0,0,0.5), (0,0,0.7)])
        bounds = [0, 1, 2, 3]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    elif type == "wind":
        cmap = plt.cm.colors.ListedColormap(['black', (0,0.5,0.5), (0,0.5,0.7), (0,0.5,1)])
        bounds = [0, 1, 2, 3, 21]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    else:
        cmap = plt.cm.viridis
        norm = None

    # Debug de valores únicos
    unique_values = np.unique(matrix)
    print(f"DEBUG - Unique values in matrix: {unique_values}")

    # Contar estados
    burned_count = np.sum(matrix == 0)
    burning_count = np.sum(matrix == 1)
    unburned_count = np.sum(matrix == 2)
    if type == "fire":
        print(f"DEBUG - Counts: BURNED={burned_count}, BURNING={burning_count}, UNBURNED={unburned_count}")
    elif type == "vegetation":
        print(f"DEBUG - Counts: DEAD={burned_count}, ALIVE={unburned_count}")
    elif type == "humidity":
        print(f"DEBUG - Counts: DRY={burned_count}, HUMID={unburned_count}")
    elif type == "wind":
        print(f"DEBUG - Counts: CALM={burned_count}, STRONG={unburned_count}")

    # Mostrar la matriz
    cax = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')

    # Título y etiquetas
    if type == "fire":
        ax.set_title(f'Layer {id} - {title}\n(Black=BURNED, Red=BURNING, Green=UNBURNED)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='black', label=f'BURNED ({burned_count})'),
            Patch(facecolor='red', label=f'BURNING ({burning_count})'),
            Patch(facecolor='green', label=f'UNBURNED ({unburned_count})')
        ]
    elif type == "vegetation":
        ax.set_title(f'Layer {id} - {title}\n(Black=DEAD, Green=ALIVE)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='black', label=f'DEAD ({burned_count})'),
            Patch(facecolor='green', label=f'ALIVE ({unburned_count})')
        ]
    elif type == "humidity":
        ax.set_title(f'Layer {id} - {title}\n(Blue=HUMID, Yellow=DRY)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='black', label=f'DRY ({burned_count})'),
            Patch(facecolor='blue', label=f'HUMID ({unburned_count})')
        ]
    elif type == "wind":
        ax.set_title(f'Layer {id} - {title}\n(Purple=CALM, Red=STRONG)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='blue', label=f'CALM ({burned_count})'),
            Patch(facecolor='darkblue', label=f'STRONG ({unburned_count})')
        ]
    else:
        ax.set_title(f'Layer {id} - {title}')
        legend_elements = []

    # Invertir el eje y para que 0 esté abajo
    ax.invert_yaxis()

    # Añadir la leyenda solo si corresponde
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))


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

def results_window(domain, fireEvolution, vegetationEvolution, humidityEvolution, windEvolution):
    """
    Displays a window with a slider and radio buttons to visualize the evolution of fire, vegetation, humidity, and wind over time.
    Args:
        domain (str): The domain type, either 'Z' for raster or other for vectorial.
        fireEvolution (list): A list of matrices or vectors representing the evolution of fire over time.
        vegetationEvolution (list): A list of matrices or vectors representing the evolution of vegetation over time.
        humidityEvolution (list): A list of matrices or vectors representing the evolution of humidity over time.
        windEvolution (list): A list of matrices or vectors representing the evolution of wind over time.
    
    The window contains:
        - A matplotlib figure to display the selected evolution state.
        - A slider to navigate through different frames of the selected evolution.
        - Radio buttons to switch between fire, vegetation, humidity, and wind evolutions.
    """
    root = tk.Tk()
    root.title("Select Action")
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
 
    # Slider for selecting the matrix
    def on_slider_change(val, layerEvolution, layer_type, use_domain=domain):
        frame = int(float(val))
        if use_domain == 'Z':
            plot_raster(ax, layerEvolution[frame], id=0, type=layer_type, color='red', title=f'State at Frame {frame}')
        else:
            plot_vectorial(ax, layerEvolution[frame], id=0, radius=1, color='red', title=f'State at Frame {frame}')
        # Forzar actualización completa del canvas
        canvas.flush_events()
        canvas.draw_idle()

    # Label to display the current value of the slider
    slider_label = tk.Label(root, text="Steeps")
    slider_label.pack(side=tk.BOTTOM, pady=5)

    # Slider for selecting the matrix
    # Inicialmente para fire
    slider = ttk.Scale(root, from_=0, to=len(fireEvolution) - 1, orient=tk.HORIZONTAL, command=lambda val: on_slider_change(val, fireEvolution, "fire"))
    slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
     
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=20)

    # Buttons to change the layerEvolution
    def set_fire_evolution():
        slider.config(command=lambda val: on_slider_change(val, fireEvolution, "fire"))
        slider_label.config(text="Fire")
        if domain == 'Z':
            plot_raster(ax, fireEvolution[0], id=0, type="fire", color='red', title='Fire - Initial State')
        else:
            plot_vectorial(ax, fireEvolution[0], id=0, radius=1, color='red', title='Fire - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.flush_events()
        canvas.draw_idle()

    def set_vegetation_evolution():
        slider.config(command=lambda val: on_slider_change(val, vegetationEvolution, "vegetation"))
        slider_label.config(text="Vegetation")
        if domain == 'Z':
            plot_raster(ax, vegetationEvolution[0], id=0, type="vegetation", color='green', title='Vegetation - Initial State')
        else: 
            plot_vectorial(ax, vegetationEvolution[0], id=0, radius=1, color='green', title='Vegetation - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.flush_events()
        canvas.draw_idle()
        
    def set_humidity_evolution():
        slider.config(command=lambda val: on_slider_change(val, humidityEvolution, "humidity"))
        slider_label.config(text="Humidity")
        if domain == 'Z':
            plot_raster(ax, humidityEvolution[0], id=0, type="humidity", color='blue', title='Humidity - Initial State')
        else:
            plot_vectorial(ax, humidityEvolution[0], id=0, radius=1, color='blue', title='Humidity - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.flush_events()
        canvas.draw_idle()

    def set_wind_evolution():
        slider.config(command=lambda val: on_slider_change(val, windEvolution, "wind"))
        slider_label.config(text="Wind")
        plot_raster(ax, windEvolution[0], id=0, type="wind", color='purple', title='Wind - Initial State')
        slider.set(0)  # Reset slider to the beginning
        canvas.flush_events()
        canvas.draw_idle()

    # Variable to keep track of the selected option
    selected_option = tk.StringVar(value="Fire")

 # Radio buttons to select the layerEvolution
    radio_fire = ttk.Radiobutton(button_frame, text="Fire", variable=selected_option, value="Fire", command=set_fire_evolution)
    radio_fire.pack(side=tk.LEFT, padx=10)

    radio_vegetation = ttk.Radiobutton(button_frame, text="Vegetation", variable=selected_option, value="Vegetation", command=set_vegetation_evolution)
    radio_vegetation.pack(side=tk.LEFT, padx=10)

    radio_humidity = ttk.Radiobutton(button_frame, text="Humidity", variable=selected_option, value="Humidity", command=set_humidity_evolution)
    radio_humidity.pack(side=tk.LEFT, padx=10)

    radio_wind = ttk.Radiobutton(button_frame, text="Wind", variable=selected_option, value="Wind", command=set_wind_evolution)
    radio_wind.pack(side=tk.LEFT, padx=10)

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
def evolution_function_on_Z(point,state, vegetation, humidity, wind, new_state, new_vegetation, new_humidity, new_wind):
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

    global DEBUG, D_i, D_j, BURNING, UNBURNED, BURNED

    new_LP = []
    nc = []
    vc = []
    max_dim = [100,100]
    #Getting the nucleous.
    nc = get_nc(point)
    i, j = nc
   
    # obtaint the wind speed
    wind_speed = combination_function_on_Z(nc, wind)

    if DEBUG:
        global D_i, D_j
        if i == D_i and i == D_j:
            print(f"DEBUG - Processing point {point} with state {state[i, j]} and vegetation {vegetation[i, j]}")
    
    cell_state = combination_function_on_Z(nc,state)
    #Getting the vicinity
    vc = get_vc_Z(point, max_dim, wind_speed, cell_state)

    if cell_state == BURNING:
        cell_vegetion = combination_function_on_Z(nc,vegetation)
        if cell_vegetion <= 0:
            new_state[i, j] = BURNED
            new_LP.append([i, j])
            new_LP.extend([elems for elems in get_vc_Z(point, max_dim, wind_speed)])
        else:
            # Decreasing the vegetation.
            new_vegetation[i, j] -= 1
            new_LP.append([i, j])
            #new_LP.extend([elems for elems in get_vc_Z(point, max_dim, wind_speed)])
    elif cell_state == UNBURNED:
        for point_vc in vc:
            #We must acces information contained on other layers, therefore we will use combination funcion
            #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
            #is just returning the same point that will be used as center of the vicinity.
            i_vc, j_vc = point_vc
            vicinity_cell_state = combination_function_on_Z(point_vc, state)
            if vicinity_cell_state == BURNING:
                if humidity[i, j] > 0:
                    new_humidity[i, j] -= 1
                    new_LP.append([i, j])
                elif vegetation[i, j] > 0:
                    new_state[i, j] = BURNING
                    # new_wind=increase_wind_in_vicinity(nc, wind,max_dim,10)
                    new_LP.append([i, j])
                    new_LP.extend([elems for elems in get_vc_Z(point, max_dim, wind_speed)])

    return new_LP,new_state, new_vegetation, new_humidity, new_wind

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
    vegetation_img_path = os.path.join(folder_path, 'vegetation.img')

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
    vegetation_data = read_idrisi_raster_file(vegetation_img_path)

    # humidity layer
    humidity_data = read_idrisi_raster_file(humidity_img_path)

    # wind layer
    wind_data = read_idrisi_raster_file(wind_img_path)
   
    # defining the size for the layers, the same for all.
    size = (100, 100)

    # Auxiliary functions to convert the vector in a matrix of data for all layers.
    humidity_data = humidity_data.reshape(size)
    vegetation_data = vegetation_data.reshape(size)
    wind_data = wind_data.reshape(size)

    # Modifying the initial conditions to start the wildfire
    initial_fire = np.full(size, UNBURNED)
    ini_point = [70, 70]
    max_dim = [100, 100]
    i, j = ini_point
    initial_fire[i, j] = BURNING
    LP = []

    # Adding the initial point we change.
    LP.append(ini_point)

    # obtaint the wind speed
    wind_speed = combination_function_on_Z(ini_point, wind_data)

    #Getting the vicinity
    vc = get_vc_Z(ini_point, max_dim, wind_speed)

    # Also adding the neighborhoods of this point.
    LP.extend([point for point in vc])

    # Variable that will contain all the states we define on the execution of the model.
    fireEvolution = [initial_fire]
    vegetationEvolution = [vegetation_data]
    humidityEvolution = [humidity_data]
    windEvoution = [wind_data]

    # Number of steps to execute the evolution function
    n_steps = 100

    # n_steps represents the finish condition of the simulation.
    for _ in range(n_steps):
        LP_rep = []
        LP_new = []
        new_state = fireEvolution[-1].copy()
        new_vegetation = vegetationEvolution[-1].copy()
        new_humidity = humidityEvolution[-1].copy()
        new_wind = windEvoution[-1].copy()
        # Event Scheduling simulation engine, where LP is the event list.
        for point in LP:
            LP_new, new_state, new_vegetation, new_humidity, new_wind = evolution_function_on_Z(point, fireEvolution[-1], vegetationEvolution[-1], humidityEvolution[-1], windEvoution[-1], new_state, new_vegetation, new_humidity, new_wind)
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]
        
        fireEvolution.append(new_state)
        vegetationEvolution.append(new_vegetation)
        humidityEvolution.append(new_humidity)
        windEvoution.append(new_wind)

    return fireEvolution, vegetationEvolution, humidityEvolution, windEvoution

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

def find_polygon_id(point, polygons, type=None):
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
    global DEBUG, D_i, D_j

    i, j = point
    # Start debugging
    if DEBUG:
        if(i == D_i and j == D_j):
            print(f"DEBUG - Searching for point {point} in polygons")
    # end debugging

    smallest_polygon_id = None
    smallest_area = float('inf')
    smallest_polygon_id = float('inf')
    current_polygon_id = float('inf')
    current_area = float('inf')
    
    # Verificar si el punto está en la estructura global identificada por type
    polygon_id_dictionary = None
    if type is not None:
        polygon_id_dictionary = get_point_id_from_global_structure(point, type)

    # If the point is in the global structure, return the ID directly
    if polygon_id_dictionary is not None:
        return polygon_id_dictionary
    else:
        # If the point is not in the global structure, proceed to check polygons
        for polygon in polygons:
            if is_point_in_polygon(point, polygon['points']):
                if len(polygon['points']) == 1: #if the poligon is a point
                    current_area = -1
                    current_polygon_id = polygon['id']
                elif len(polygon['points']) == 2: #if the poligon is a line segment
                    current_area = 0
                    current_polygon_id = polygon['id']
                else:
                    current_polygon = Polygon(polygon['points'])
                    current_area = current_polygon.area
                    current_polygon_id = polygon['id']
                # Check if the current polygon is smaller than the smallest found so far
                if current_area < smallest_area:
                    smallest_area = current_area
                    smallest_polygon_id = current_polygon_id
        # Si no se encontró ningún polígono, warning y devuelve -1
        if smallest_polygon_id is None or smallest_polygon_id == float('inf'):
            print(f"WARNING: No polygon found for point {point} with type '{type}'. Returning -1.")
            return -1
        return smallest_polygon_id

def remove_points_from_global_structure(points_to_remove, type):
    """
    Elimina una lista de puntos de toda la estructura global del tipo especificado.
    Busca cada punto en points_to_remove a través de todos los IDs y los elimina.
    
    Args:
        points_to_remove (list): Lista de puntos (tuplas) a eliminar
        type (str): Tipo de estructura global ("state", "humidity", "vegetation")
    
    Returns:
        None
    """
    global DEBUG, D_i, D_j, STATE_POINTS_BY_ID, HUMIDITY_POINTS_BY_ID, VEGETATION_POINTS_BY_ID
    
    # Seleccionar la estructura global correspondiente según el tipo
    if type == "state":
        points_by_id = STATE_POINTS_BY_ID
    elif type == "humidity":
        points_by_id = HUMIDITY_POINTS_BY_ID
    elif type == "vegetation":
        points_by_id = VEGETATION_POINTS_BY_ID
    else:
        raise ValueError(f"Tipo de capa no válido: {type}. Debe ser 'state', 'humidity' o 'vegetation'")
    
    # Convertir points_to_remove a set para búsqueda más eficiente
    points_to_remove_set = set(points_to_remove)
    ids_to_remove = []  # Lista para almacenar IDs que quedan vacíos
    
    # Iterar por todos los IDs en la estructura
    for id_key in list(points_by_id.keys()):
        # Filtrar puntos que no estén en points_to_remove
        points_by_id[id_key] = [pt for pt in points_by_id[id_key] 
                               if pt not in points_to_remove_set]
        if DEBUG:
            for point in points_to_remove_set:
                if point[0] == D_i and point[1] == D_j:
                    print(f"DEBUG - Removing point {point} with ID {id_key} from {type}")

        # Si el ID queda vacío, marcarlo para eliminación
        if len(points_by_id[id_key]) == 0:
            ids_to_remove.append(id_key)
    
    # Eliminar IDs que quedaron vacíos
    for id_key in ids_to_remove:
        del points_by_id[id_key]

def add_points_to_global_structure(points_to_add, target_id, type):
    """
    Añade una lista de puntos al ID especificado en el diccionario global, evitando duplicados.
    
    Args:
        points_to_add (list): Lista de puntos (tuplas) a añadir
        target_id (str): ID donde añadir los puntos
        type (str): Tipo de estructura global ("state", "humidity", "vegetation")
    
    Returns:
        None
    """
    global DEBUG, D_i, D_j, STATE_POINTS_BY_ID, HUMIDITY_POINTS_BY_ID, VEGETATION_POINTS_BY_ID
    
    # Seleccionar la estructura global correspondiente según el tipo
    if type == "state":
        points_by_id = STATE_POINTS_BY_ID
    elif type == "humidity":
        points_by_id = HUMIDITY_POINTS_BY_ID
    elif type == "vegetation":
        points_by_id = VEGETATION_POINTS_BY_ID
    else:
        raise ValueError(f"Tipo de capa no válido: {type}. Debe ser 'state', 'humidity' o 'vegetation'")
    
    # Add the entry if it does not exists
    if target_id not in points_by_id:
        points_by_id[target_id] = []
    
    # Add each point if it does not already exist in this ID
    for point_to_add in points_to_add:
        if not any(are_points_equal(existing_pt, point_to_add, dist=0) for existing_pt in points_by_id[target_id]):
            if DEBUG:
                if point_to_add[0] == D_i and point_to_add[1] == D_j:
                    print(f"DEBUG - Adding point {point_to_add} to ID {target_id} of type {type}")
            points_by_id[target_id].append(point_to_add)

def are_points_equal(point1, point2, dist=0):
    """
    Compares two points and returns True if they are equal or within a specified distance.
    
    Args:
        point1 (tuple): First point coordinates (x, y).
        point2 (tuple): Second point coordinates (x, y).
        dist (float, optional): Maximum distance tolerance. Default is 0 (exact equality).
    
    Returns:
        bool: True if points are equal or within distance tolerance, False otherwise.
    """
    # Convert both points to float for robust comparison
    x1, y1 = float(point1[0]), float(point1[1])
    x2, y2 = float(point2[0]), float(point2[1])
    if dist == 0:
        # Exact equality check (with float)
        return x1 == x2 and y1 == y2
    else:
        # Distance-based comparison
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance <= dist

def addVectorialMap(vectorialMap, layersArray, type):
    """
    Adds a vectorial map to the layers array and maintains global point structures by ID and type.
    
    This function:
    1. Manages three global dictionaries (STATE_POINTS_BY_ID, HUMIDITY_POINTS_BY_ID, VEGETATION_POINTS_BY_ID)
    2. Ensures each point exists in only one ID per type
    3. Removes duplicate points from other IDs when adding new ones
    
    Args:
        vectorialMap (dict or list): A dictionary with 'id' and 'points' keys, or a list of such dictionaries
        layersArray (list): A list of vectorial maps for the layer
        type (str): The type of layer ("state", "humidity", or "vegetation")
    Returns:
        None
    """
    global D_i, D_j, DEBUG
    
    def remove_points_from_layers_array(points_to_remove, layersArray):
        """
        Función auxiliar que elimina una lista de puntos únicamente del layersArray.
        
        Args:
            points_to_remove (list): Lista de puntos (tuplas) a eliminar
            layersArray (list): Array de layers de donde eliminar los puntos
        
        Returns:
            None
        """
        # Eliminar puntos del layersArray
        layers_to_remove = []  # Lista de layers que quedan vacíos
        
        for layer_index, layer in enumerate(layersArray):
            # Cada layer es una lista de vectorial maps (diccionarios)
            if isinstance(layer, list):
                if DEBUG:
                    print("DEBUG - ERROR on Layer %d: Expected dict, got list", layer_index)
                continue            
            # Si el layer es directamente un diccionario con puntos
            elif isinstance(layer, dict) and 'points' in layer:
                # Filtrar puntos que no estén en points_to_remove
                original_points = layer['points'][:]
                filtered_points = []
                
                for point in original_points:
                    should_keep = True
                    for point_to_remove in points_to_remove:
                        if are_points_equal(point, point_to_remove, dist=0):
                            should_keep = False
                            break
                    if should_keep:
                        filtered_points.append(point)
                
                # Actualizar los puntos del elemento
                layer['points'] = filtered_points
                
                # Si el elemento queda vacío, marcarlo para eliminación
                if len(filtered_points) == 0:
                    layers_to_remove.append(layer_index)
        
        # Eliminar layers vacíos (en orden inverso para no alterar índices)
        for layer_index in reversed(layers_to_remove):
            layersArray.pop(layer_index)


    # Convertir vectorialMap a lista si es un diccionario único
    if isinstance(vectorialMap, dict):
        vectorialMap_copy = [vectorialMap]
    
    # Usar la función auxiliar para eliminar todos los puntos del vectorialMap del layersArray
    for vectorial_element in vectorialMap_copy:
        if isinstance(vectorial_element, dict) and 'id' in vectorial_element and 'points' in vectorial_element:
            new_id = str(vectorial_element['id'])  # Convertir ID a string para consistencia
            new_points = vectorial_element['points']
            #remove_points_from_layers_array(new_points, layersArray)

            # 1. Eliminar estos puntos de cualquier ID en la estructura global
            remove_points_from_global_structure(new_points, type)
            
            # 2. Añadir los puntos al ID correspondiente
            add_points_to_global_structure(new_points, new_id, type)

    # Añadir el vectorialMap a layersArray si no existe ya
    if vectorialMap not in layersArray:
        layersArray.append(vectorialMap)

    '''
   # Convertir vectorialMap a lista si es un diccionario único
    if isinstance(vectorialMap, dict):
        vectorialMap = [vectorialMap]
    
    # Procesar cada elemento del vectorialMap
    for vectorial_element in vectorialMap:
        if isinstance(vectorial_element, dict) and 'id' in vectorial_element and 'points' in vectorial_element:
            new_id = str(vectorial_element['id'])  # Convertir ID a string para consistencia
            new_points = vectorial_element['points']

            # 1. Eliminar estos puntos de cualquier otro ID en la estructura global
            remove_points_from_global_structure(new_points, new_id, type)
            
            # 2. Añadir los puntos al ID correspondiente
            add_points_to_global_structure(new_points, new_id, type)
    '''

def get_point_id_from_global_structure(point, type):
    """
    Busca un punto en la estructura global y devuelve su ID si lo encuentra.
    
    Args:
        point (tuple): Punto (x, y) a buscar
        type (str): Tipo de estructura global ("state", "humidity", "vegetation")
    
    Returns:
        str or None: El ID del punto si se encuentra, None si no se encuentra
    """
    global STATE_POINTS_BY_ID_FINAL, HUMIDITY_POINTS_BY_ID_FINAL, VEGETATION_POINTS_BY_ID_FINAL

    if type == "state":
        points_by_id = STATE_POINTS_BY_ID_FINAL
    elif type == "humidity":
        points_by_id = HUMIDITY_POINTS_BY_ID_FINAL
    elif type == "vegetation":
        points_by_id = VEGETATION_POINTS_BY_ID_FINAL
    else:
        raise ValueError(f"Tipo de capa no válido: {type}. Debe ser 'state', 'humidity' o 'vegetation'")

    # Buscar el punto en la estructura global
    for id_key, points_list in points_by_id.items():
        for existing_point in points_list:
            if are_points_equal(point, existing_point, dist=0):
                return id_key
    
    # Si no se encuentra el punto, devolver None
    return None

def print_points_by_id_debug(type=None):
    """
    Función de debug para imprimir las estructuras de puntos por ID.
    
    Args:
        type (str, optional): Tipo específico a mostrar. Si es None, muestra todos.
    """
    global STATE_POINTS_BY_ID, HUMIDITY_POINTS_BY_ID, VEGETATION_POINTS_BY_ID
    
    if type is None or type == "state":
        print(f"STATE_POINTS_BY_ID: {STATE_POINTS_BY_ID}")
    if type is None or type == "humidity":
        print(f"HUMIDITY_POINTS_BY_ID: {HUMIDITY_POINTS_BY_ID}")
    if type is None or type == "vegetation":
        print(f"VEGETATION_POINTS_BY_ID: {VEGETATION_POINTS_BY_ID}")

def print_global_structures_summary():
    """
    Imprime un resumen de las estructuras globales de puntos por ID.
    Muestra estadísticas sobre la distribución de puntos por tipo y ID.
    """
    global STATE_POINTS_BY_ID, HUMIDITY_POINTS_BY_ID, VEGETATION_POINTS_BY_ID
    
    print("\n" + "="*60)
    print("RESUMEN DE ESTRUCTURAS GLOBALES DE PUNTOS POR ID")
    print("="*60)
    
    # Estadísticas para STATE
    print(f"\nSTATE POINTS BY ID:")
    total_state_points = 0
    for id_key, points in STATE_POINTS_BY_ID.items():
        count = len(points)
        total_state_points += count
        print(f"  ID {id_key}: {count} puntos")
    print(f"  TOTAL: {total_state_points} puntos")
    
    # Mostrar puntos en estado BURNING (ID 1)
    if '1' in STATE_POINTS_BY_ID and len(STATE_POINTS_BY_ID['1']) > 0:
        print(f"\n  PUNTOS EN ESTADO BURNING (ID 1):")
        burned_points = STATE_POINTS_BY_ID['1']
        if len(burned_points) <= 2000:  # Si hay pocos puntos, mostrar todos
            for i, point in enumerate(burned_points):
                print(f"    {i+1:2d}. {point}")
        else:  # Si hay muchos puntos, mostrar solo los primeros 20
            for i, point in enumerate(burned_points[:20]):
                print(f"    {i+1:2d}. {point}")
            print(f"    ... y {len(burned_points) - 20} puntos más")
    
    # Estadísticas para HUMIDITY
    print(f"\nHUMIDITY POINTS BY ID:")
    total_humidity_points = 0
    for id_key, points in HUMIDITY_POINTS_BY_ID.items():
        count = len(points)
        total_humidity_points += count
        print(f"  ID {id_key}: {count} puntos")
    print(f"  TOTAL: {total_humidity_points} puntos")
    
    # Estadísticas para VEGETATION
    print(f"\nVEGETATION POINTS BY ID:")
    total_vegetation_points = 0
    for id_key, points in VEGETATION_POINTS_BY_ID.items():
        count = len(points)
        total_vegetation_points += count
        print(f"  ID {id_key}: {count} puntos")
    print(f"  TOTAL: {total_vegetation_points} puntos")
    
    print("\n" + "="*60 + "\n")

def sort_points(points):
    """
    Sorts a list of points in a 2D plane based on their angle with respect to the centroid of the polygon formed by the points.
    We do not want intersections in the edges of the polygon.
    Args:
        points (list of tuple): A list of tuples where each tuple represents a point (x, y) in a 2D plane.
    Returns:
        list of tuple: A list of points sorted by their angle with respect to the centroid of the polygon.
    """
    import math
    if len(points) < 3:
        return points
    centroid_x = sum(x for x, y in points) / len(points)
    centroid_y = sum(y for x, y in points) / len(points)
    def angle_and_distance(point):
        x, y = point
        angle = math.atan2(y - centroid_y, x - centroid_x)
        dist = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
        return (angle, dist)
    # Ordena primero por ángulo, luego por distancia al centroide (para desempatar)
    return sorted(points, key=angle_and_distance)
    
def simplifyVectorialMap(vectorialMap):
    """
    Simplifies a vectorial map by merging polygons with the same ID using union operations.

    This function groups polygons by ID and merges them if they meet any of these criteria:
        1. They have points in common (intersect)
        2. One is inside the other (containment relationship)
        3. They are adjacent or overlapping
        4. (Nuevo) Si algún punto de dos polígonos está a menos de 'tolerance' distancia, se unen

    For polygons with the same ID, this creates unified geometries using the union operation.

    OPTIMIZACIÓN: Agrupar por ID y usar unary_union de Shapely para cada grupo, agrupando por clusters de proximidad.
    Parámetro:
        tolerance (float): distancia máxima para unir clusters (default=1.5)
    """
    from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiPolygon, GeometryCollection
    from shapely.ops import unary_union
    import numpy as np
    from sklearn.cluster import DBSCAN
    import inspect
    def get_all_points(geom):
        from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon, GeometryCollection
        if isinstance(geom, Polygon):
            return list(geom.exterior.coords)
        elif isinstance(geom, LineString):
            return list(geom.coords)
        elif isinstance(geom, Point):
            return [(geom.x, geom.y)]
        elif isinstance(geom, MultiPoint):
            return [(pt.x, pt.y) for pt in geom.geoms]
        elif isinstance(geom, MultiPolygon):
            pts = []
            for poly in geom.geoms:
                pts.extend(list(poly.exterior.coords))
            return pts
        elif isinstance(geom, GeometryCollection):
            pts = []
            for g in geom.geoms:
                pts.extend(get_all_points(g))
            return pts
        else:
            return []

    import numpy as np
    from shapely.geometry import Polygon, Point, LineString
    from shapely.ops import unary_union
    # Paso 1: Agrupar por ID (0 a 3)
    tolerance = 1.5
    simplified_vectorial_map = []
    for poly_id in range(0, 4):
        # Extraer todos los elementos de este ID
        elements = [p for p in vectorialMap if p['id'] == poly_id]
        if not elements:
            continue
        groups = []
        used = set()
        for i, elem in enumerate(elements):
            if i in used:
                continue
            group = [elem]
            used.add(i)
            pts1 = np.array(elem['points'])
            for j, other in enumerate(elements):
                if j == i or j in used:
                    continue
                pts2 = np.array(other['points'])
                # Comprobar si algún punto está a distancia < tolerance
                close = False
                for p1 in pts1:
                    for p2 in pts2:
                        if np.linalg.norm(np.array(p1) - np.array(p2)) < tolerance:
                            close = True
                            break
                    if close:
                        break
                if close:
                    group.append(other)
                    used.add(j)
            groups.append(group)
        # Para cada grupo, unir los puntos, eliminar interiores y crear el nuevo elemento
        for group in groups:
            all_points = []
            for g in group:
                # g['points'] puede ser un punto, segmento o polígono
                if isinstance(g['points'], (list, tuple)):
                    all_points.extend(g['points'])
                else:
                    all_points.append(g['points'])
            filtered_points = remove_interior_points(all_points)
            # Ordenar los puntos para formar un contorno (ángulo respecto al centroide)
            if filtered_points:
                if len(filtered_points) > 2:
                    ordered_points = sort_points(filtered_points)
                else:
                    ordered_points = filtered_points
                simplified_vectorial_map.append({'id': poly_id, 'points': ordered_points})
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

    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for pt in points:
        if pt not in seen:
            unique_points.append(pt)
            seen.add(pt)

    points_set = set(unique_points)
    filtered_points = [point for point in unique_points if not is_interior(point, points_set)]
    # Si el filtrado elimina todos los puntos, devolver los originales únicos
    if not filtered_points:
        return unique_points
    return filtered_points

#m:n-CAk on R functions
def evolution_function_on_R(point,fire, vegetation, humidity, new_state, new_vegetation, new_humidity, new_wind):
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

    global D_i, D_j, DEBUG, BURNING, UNBURNED, BURNED

    new_LP = []
    nc = []
    vc = []
    max_dim = [100,100]

    #Getting the nucleous.
    nc = get_nc(point)
    i, j = nc
    #Start Debugging information
    if DEBUG:
        if i == D_i and j == D_j:
            print(f"DEBUG - Searching for point {point} in fire layer")
    # End Debugging information

    #We must acces information contained on other layers, therefore we will use combination funcion
    #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
    #is just returning the point.

    #we access to the wind layer that is on Z to obtain the wind speed at this point.
    #This is a point, therefore we can use the combination function on Z.
    wind_speed = combination_function_on_Z(nc,new_wind)

    cell_state = combination_function_on_R(nc,fire, "state") 

    #Getting the vicinity, it can be defined also over the nucleous.
    vc = get_vc_R(point, max_dim, wind_speed, cell_state)
    #debugging information
    if DEBUG:
        if i == D_i and j == D_j and cell_state == BURNING:
            print(f"DEBUG - Fire state at point {point} is {cell_state}")
    #End debugging information
    cell_vegetation = combination_function_on_R(nc,vegetation, "vegetation")
    #Start Debugging information
    #if cell_vegetation == 20:
    #    print(f"DEBUG - Vegetation at point {point} is {cell_vegetation}")
    # End Debugging information
    if cell_state == BURNING:
        if cell_vegetation <= 0:
            points = []
            points.append((i, j))
            if DEBUG:
                if(i == D_i and j == D_j):
                    print(f"DEBUG - Point {point} is burning and has no vegetation, setting to BURNED")
            addVectorialMap({'id': BURNED, 'points': points},new_state, "state")
            new_LP.append([i, j])
            #new_LP.extend([elems for elems in get_vc_R(point, max_dim, wind_speed)])
            new_LP.extend([[elems[0], elems[1]] for elems in get_vc_R(point, max_dim, wind_speed)])     
        else:
            points = []
            cell_vegetation -= 1
            points.append((i, j))
            addVectorialMap({'id': cell_vegetation, 'points': points},new_vegetation, "vegetation")
            new_LP.append([i, j])
            # new_LP.extend([elems for elems in get_vc_R(point, max_dim, wind_speed)])
    elif cell_state == UNBURNED:
        cell_humidity =  combination_function_on_R(nc,humidity, "humidity")
        new_cell_humidity = cell_humidity
        for point_vc in vc:
            #We must acces information contained on other layers, therefore we will use combination funcion
            #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
            #is just returning the point.
            vicinity_cell_state = combination_function_on_R(point_vc,fire, "state")
            if vicinity_cell_state  == BURNING:
                if cell_humidity > 0:
                    new_cell_humidity -= 1
                    points = []
                    points.append((i, j))
                    addVectorialMap({'id': max(new_cell_humidity,0), 'points': points},new_humidity, "humidity")
                    new_LP.append([i, j])
                    # No añadir vicinity cuando solo se reduce humedad
                elif cell_vegetation > 0:
                    points = []
                    points.append((i, j))
                    addVectorialMap({'id': BURNING, 'points': points},new_state, "state")
                    new_LP.append([i, j])
                    #new_LP.extend([elems for elems in get_vc_R(point, max_dim, wind_speed)])
                    new_LP.extend([[elems[0], elems[1]] for elems in get_vc_R(point, max_dim, wind_speed)])

    return new_LP,new_state, new_vegetation, new_humidity, new_wind


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

    global STATE_POINTS_BY_ID, HUMIDITY_POINTS_BY_ID, VEGETATION_POINTS_BY_ID, STATE_POINTS_BY_ID_FINAL, HUMIDITY_POINTS_BY_ID_FINAL, VEGETATION_POINTS_BY_ID_FINAL
    
    folder_path = './'
    size = (100, 100)

    # Reading vegetation and humidity layers

    fileVegetation = 'vegetation.vec'
    polygonsVegetation = read_idrisi_vector_file(fileVegetation)

    fileHumidity = 'humidity.vec'
    polygonsHumidity = read_idrisi_vector_file(fileHumidity)

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

    max_dim = [100, 100]
    LP = []

    for polygon in polygonsFire:
        polygon_id = polygon['id']
        points = polygon['points']
        if polygon_id == BURNING:
            for (x, y) in points:
                ini_point = (x, y)
                LP.append(ini_point)
                LP.extend([point for point in get_vc_Z(ini_point, max_dim)])

    fireEvolution = [polygonsFire]
    vegetationEvolution = [polygonsVegetation]
    humidityEvolution = [polygonsHumidity]
    windEvolution = [wind_data]

    n_steps = 100

    for _ in range(n_steps):
        LP_rep = []
        LP_new = []
        new_state = fireEvolution[-1].copy()
        new_vegetation = vegetationEvolution[-1].copy()
        new_humidity = humidityEvolution[-1].copy()
        new_wind = windEvolution[-1].copy()

        for point in LP:
            LP_new, new_state, new_vegetation, new_humidity, new_wind = evolution_function_on_R(point, fireEvolution[0], vegetationEvolution[0], humidityEvolution[0], new_state, new_vegetation, new_humidity, new_wind)
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]

        simplify=True
        if simplify:
            fireEvolution.append(simplifyVectorialMap(new_state))
            vegetationEvolution.append(simplifyVectorialMap(new_vegetation))
            humidityEvolution.append(simplifyVectorialMap(new_humidity))
            windEvolution.append(new_wind)
        else:
            fireEvolution.append(new_state)
            vegetationEvolution.append(new_vegetation)
            humidityEvolution.append(new_humidity)
            windEvolution.append(new_wind)


        import copy
        STATE_POINTS_BY_ID_FINAL = copy.deepcopy(STATE_POINTS_BY_ID)
        HUMIDITY_POINTS_BY_ID_FINAL = copy.deepcopy(HUMIDITY_POINTS_BY_ID)
        VEGETATION_POINTS_BY_ID_FINAL = copy.deepcopy(VEGETATION_POINTS_BY_ID)


        print(f"Step {_+1}/{n_steps} completed. Fire evolution size: {len(fireEvolution[-1])}, Vegetation size: {len(vegetationEvolution[-1])}, Humidity size: {len(humidityEvolution[-1])}, Wind size: {len(windEvolution[-1])}")

    return fireEvolution, vegetationEvolution, humidityEvolution, windEvolution

##############################################################
#m:n-CAk main functions
##############################################################

def get_vc_Z(point, max_dim, wind_speed=0, cell_state=BURNING):
    """
    Vicinity function. Get the valid coordinates (Von Neumann neighbourhood) adjacent to a given point within a specified maximum dimension.
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
    # wind_speed = 0
    # Add the wind effect by extending the vicinity in the direction of the wind
    if cell_state==BURNING:
        j=j+1
        while wind_speed > 0: 
            j=j+1
            if j < max_j - 1: vc.append((i, j))
            wind_speed -= 1
    else:
        j=j-1
        while wind_speed > 0: 
            j=j-1
            if j < max_j - 1: vc.append((i, j))
            wind_speed -= 1

    return vc

def increase_wind_in_vicinity(point, wind_layer, max_dim, increment=1):
    """
    Increases the wind values by a specified increment in the vicinity of a given point.
    
    Args:
        point (tuple): A tuple (i, j) representing the coordinates of the center point.
        wind_layer (numpy.ndarray): The wind layer matrix to be modified.
        max_dim (tuple): A tuple (max_i, max_j) representing the maximum dimensions of the grid.
        increment (int, optional): The value to add to each cell in the vicinity. Default is 1.
    
    Returns:
        numpy.ndarray: The modified wind layer with increased values in the vicinity.
    """
    # Get the current wind speed at the center point for vicinity calculation
    current_wind_speed = combination_function_on_Z(point, wind_layer)
    
    # Get the vicinity points using the get_vc_Z function
    vicinity_points = get_vc_Z(point, max_dim, current_wind_speed)
    
    # Create a copy of the wind layer to avoid modifying the original
    modified_wind_layer = wind_layer.copy()
    
    # Increase wind values in the vicinity
    for vc_point in vicinity_points:
        i, j = vc_point
        # Ensure the point is within bounds (additional safety check)
        if 0 <= i < max_dim[0] and 0 <= j < max_dim[1]:
            modified_wind_layer[i, j] += increment
    
    return modified_wind_layer

def get_vc_R(point, max_dim, wind_speed=0, cell_state=BURNING):
    """
    Vicinity function. Get the valid coordinates (Von Neumann neighbourhood) adjacent to a given point within a specified maximum dimension.
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

    if DEBUG:
        if i == 0 and j == 0:
            print(f"DEBUG - Adding point 0,0!")

    #consider the case of wind
    wind_speed = wind_speed*1.5  # Assuming wind_speed is a multiplier for the number of steps to consider
    # Add the wind effect by extending the vicinity in the direction of the wind
    # we extend j because on numpy the first coordinate is the y coordinate and the second is the x coordinate.
    if cell_state==BURNING:
        i=i+1
        while wind_speed > 0: 
            i=i+1
            if i < max_i - 1: vc.append((i, j))
            wind_speed -= 1
    else:
        i=i-1
        while wind_speed > 0: 
            i=i-1
            if i < max_i - 1: vc.append((i, j))
            wind_speed -= 1

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

def combination_function_on_R(point, layer, type=None):
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
    return int(find_polygon_id(point, layer, type))

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
#Report functions
##############################################################

def print_burning_cells_report(fireEvolution, domain):
    """
    Prints a console report showing the cells that are in BURNING state at the end of the simulation.
    
    Args:
        fireEvolution (list): List of fire states from the simulation.
        domain (str): The domain type ('Z' for raster or 'R' for vectorial).
    """

    global BURNING, BURNED, UNBURNED

    print("\n" + "="*60)
    print("FIRE SIMULATION REPORT")
    print("="*60)
    
    if len(fireEvolution) == 0:
        print("No simulation data available.")
        return
    
    final_state = fireEvolution[-1]
    print(f"Domain: {domain}")
    print(f"Total simulation steps: {len(fireEvolution) - 1}")
    
    if domain == 'Z':
        # For raster domain (Z)
        burning_cells = []
        burned_cells = []
        unburned_cells = []
        other_values = []
        
        rows, cols = final_state.shape
        for i in range(rows):
            for j in range(cols):
                cell_value = final_state[i, j]
                if cell_value == BURNING:
                    burning_cells.append((i, j))
                elif cell_value == BURNED:
                    burned_cells.append((i, j))
                elif cell_value == UNBURNED:
                    unburned_cells.append((i, j))
                else:
                    other_values.append((i, j, cell_value))
        
        print(f"\nSTATE VALUES VERIFICATION:")
        print(f"- BURNING = {BURNING} (should show as RED)")
        print(f"- BURNED = {BURNED} (should show as BLACK)")
        print(f"- UNBURNED = {UNBURNED} (should show as GREEN)")
        

        print(f"\nFINAL STATE SUMMARY:")
        print(f"- BURNING cells: {len(burning_cells)}")
        print(f"- BURNED cells: {len(burned_cells)}")
        print(f"- UNBURNED cells: {len(unburned_cells)}")
        if other_values:
            print(f"- Cells with unexpected values: {len(other_values)}")
        print(f"- Total cells: {rows * cols}")
        
        # Show some sample values around burning cells for verification
        if burning_cells:
            print(f"\nCELLS IN BURNING STATE:")
            for i, (row, col) in enumerate(burning_cells):
                actual_value = final_state[row, col]
                print(f"  {i+1:3d}. Cell ({row:2d}, {col:2d}) = {actual_value}")
                if i >= 4:  # Limit to first 5 for readability
                    if len(burning_cells) > 5:
                        print(f"  ... and {len(burning_cells) - 5} more cells")
                    break
        else:
            print(f"\nNo cells are currently in BURNING state.")
            
        # Show unexpected values if any
        if other_values:
            print(f"\nCELLS WITH UNEXPECTED VALUES:")
            for i, (row, col, value) in enumerate(other_values[:5]):
                print(f"  {i+1:3d}. Cell ({row:2d}, {col:2d}) = {value}")
            if len(other_values) > 5:
                print(f"  ... and {len(other_values) - 5} more")
        
        # Sample matrix values around center for debugging
        center_i, center_j = rows//2, cols//2
        print(f"\nSAMPLE VALUES AROUND CENTER ({center_i}, {center_j}):")
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = center_i + di, center_j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    value = final_state[ni, nj]
                    print(f"({ni:2d},{nj:2d})={value}", end="  ")
            print()  # New line after each row
            
    else:
        # For vectorial domain (R)
        burning_polygons = []
        burned_polygons = []
        unburned_polygons = []
        
        for polygon in final_state:
            if polygon['id'] == BURNING:
                burning_polygons.append(polygon)
            elif polygon['id'] == BURNED:
                burned_polygons.append(polygon)
            elif polygon['id'] == UNBURNED:
                unburned_polygons.append(polygon)
        
        print(f"\nFINAL STATE SUMMARY:")
        print(f"- BURNING polygons: {len(burning_polygons)}")
        print(f"- BURNED polygons: {len(burned_polygons)}")
        print(f"- UNBURNED polygons: {len(unburned_polygons)}")
        print(f"- Total polygons: {len(final_state)}")
 
    print("="*60 + "\n")

def create_verification_map(fireEvolution, domain):
    """
    Creates a simple text-based verification map for debugging visualization issues.
    
    Args:
        fireEvolution (list): List of fire states from the simulation.
        domain (str): The domain type ('Z' for raster or 'R' for vectorial).
    """
    if domain != 'Z' or len(fireEvolution) == 0:
        return
        
    final_state = fireEvolution[-1]
    print("TEXT-BASED VERIFICATION MAP (COMPLETE 100x100 GRID):")
    print("Legend: . = UNBURNED, X = BURNED, * = BURNING")
    
    rows, cols = final_state.shape
    
    print(f"\nShowing complete grid ({rows}x{cols}):")
    
    # Print column headers (every 10th column)
    print("     ", end="")
    for j in range(0, cols, 10):
        print(f"{j:10d}", end="")
    print()
    
    print("     ", end="")
    for j in range(cols):
        print(f"{j%10}", end="")
    print()
    
    # Print the complete grid
    for i in range(rows):
        print(f"{i:3d}: ", end="")
        for j in range(cols):
            value = final_state[i, j]
            if value == UNBURNED:
                print(".", end="")
            elif value == BURNED:
                print("X", end="")
            elif value == BURNING:
                print("*", end="")
            else:
                print("?", end="")
        print()  # New line after each row
    
    # Summary statistics
    total_cells = rows * cols
    burned_count = np.sum(final_state == BURNED)
    burning_count = np.sum(final_state == BURNING)
    unburned_count = np.sum(final_state == UNBURNED)
    other_count = total_cells - burned_count - burning_count - unburned_count
    
    print(f"\nMAP SUMMARY:")
    print(f"- Total cells: {total_cells}")
    print(f"- BURNED (X): {burned_count} cells")
    print(f"- BURNING (*): {burning_count} cells")
    print(f"- UNBURNED (.): {unburned_count} cells")
    if other_count > 0:
        print(f"- OTHER (?): {other_count} cells")
    
    print("\n" + "="*60 + "\n")

##############################################################
#Main
##############################################################

if __name__ == "__main__":

    domain = 'Z'

    domain = input(".git\nPlease, select the domain to work on (Z or R): ")
    if domain != 'Z' and domain != 'R':
        print("Invalid domain. Please select 'Z' or 'R'.")
    else:
        fireEvolution = []
        vegetationEvolution = []
        humidityEvolution = []
        windEvolution = []

        if domain == 'Z':
            fireEvolution, vegetationEvolution, humidityEvolution, windEvolution = event_scheduling_on_Z()
        else:
            fireEvolution, vegetationEvolution, humidityEvolution, windEvolution = event_scheduling_on_R()

        # Print report of burning cells at the end of simulation
        print_burning_cells_report(fireEvolution, domain)
        
        # Print global structures summary
        print_global_structures_summary()
        
        # Create verification map for debugging
        create_verification_map(fireEvolution, domain)

        results_window(domain, fireEvolution, vegetationEvolution, humidityEvolution, windEvolution)