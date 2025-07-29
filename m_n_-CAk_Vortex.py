##############################################################
# Simple_on_N.py
# Created on: 2021-06-30
# Author: Pau Fonseca i Casas
# Copyright: Pau Fonseca i Casas
# Description: This script simulates the spread of a propagation using an m:n-CAk cellular automaton model over Z^2 and R^2.
# All layers share the same coordinate system; therefore, the function to change the basis is not needed.
# This example uses a nucleus composed of a set of points in the discrete case, and in the continuous case, it can be represented by a set of polygons.
# Also, the nucleus changes by adding the last vicinity obtained.
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

# Global variable to store the maximum radius found
maxRadius = 0.0

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

    Parameters:
    img_path (str): The path to the image file.

    Returns:
    numpy.ndarray: A NumPy array containing the image data as integers.
    """
    data = np.loadtxt(img_path).astype(int)
    return data

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
    Parameters:
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
    # Raster dimensions
    width, height = 100, 100

    # Create a 100x100 point matrix
    raster = np.zeros((height, width), dtype=int)

    # Iterate over each point in the matrix
    for j in range(height):
        for i in range(width):
            point = (i, j)
            polygon_id = find_polygon_id(point, polygons)
            if polygon_id is not None:
                raster[j, i] = polygon_id

    # Save the matrix in IDRISI format
    data_filename = f"{output_filename}.img"
    metadata_filename = f"{output_filename}.doc"

    # Save the data file as text
    with open(data_filename, 'w') as data_file:
        for row in raster:
            data_file.write(' '.join(map(str, row)) + '\n')


    # Create the metadata file
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

def results_window(domain, simpleEvolution, vortex_speed_evolution):
    """
    Displays a window with a slider and radio buttons to visualize the evolution over time.
    Parameters:
    domain (str): The domain type, either 'Z' for raster or other for vectorial.
    simpleEvolution (list): A list of matrices or vectors representing the evolution of the propagation over time.
    vortex_speed_evolution (list): A list of vortex speeds for each step.
    The window contains:
    - A matplotlib figure to display the selected evolution state.
    - A slider to navigate through different frames of the selected evolution.
    - Radio buttons to switch between layers.
    """
    root = tk.Tk()
    root.title("Select Action")
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
 
    # Slider for selecting the matrix
    def on_slider_change(val, layerEvolution):
        frame = int(float(val))
        title = f'State at Frame {frame}, Vortex Speed: {vortex_speed_evolution[frame]:.2f}'
        if domain == 'Z':
            plotRaster(ax, layerEvolution[frame], id=0, color='red', title=title)
        else:
            plot_vectorial(ax, layerEvolution[frame], id=0, radius=1, color='red', title=title)
        canvas.draw()

    # Label to display the current value of the slider
    slider_label = tk.Label(root, text="Steeps")
    slider_label.pack(side=tk.BOTTOM, pady=5)

    # Slider for selecting the matrix
    slider = ttk.Scale(root, from_=0, to=len(simpleEvolution) - 1, orient=tk.HORIZONTAL, command=lambda val: on_slider_change(val, simpleEvolution))
    slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
     
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=20)

    # Buttons to change the layerEvolution
    def set_simple_evolution():
        slider.config(command=lambda val: on_slider_change(val, simpleEvolution))
        slider_label.config(text="Evolution")
        title = f'Evolution - Initial State, Vortex Speed: {vortex_speed_evolution[0]:.2f}'
        if domain == 'Z':
            plotRaster(ax, simpleEvolution[0], id=0, color='red', title=title)
        else:
            plot_vectorial(ax, simpleEvolution[0], id=0, radius=1, color='red', title=title)
        slider.set(0)  # Reset slider to the beginning
        canvas.draw()

    # Variable to keep track of the selected option
    selected_option = tk.StringVar(value="Evolution")

 # Radio buttons to select the layerEvolution
    radio_evolution = ttk.Radiobutton(button_frame, text="Evolution", variable=selected_option, value="Evolution", command=set_simple_evolution)
    radio_evolution.pack(side=tk.LEFT, padx=10)

    root.mainloop()

#Not used but useful for debugging
def animate_layers(layersArray, interval=500, radius=1, color='green', title='No title'):
    """
    Animates a series of layers using matplotlib.
    Parameters:
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
def evolution_function_on_Z(points, state, new_state, ini_point):
    new_LP_set = set()
    max_dim = [100, 100]
    min_vortex_speed_this_step = 0.0

    # Calculate speed for all active points
    if points:
        min_vortex_speed_this_step = min(get_vortex_speed(p, ini_point) for p in points)

    for point in points:
        x, y = point
        vc = get_vc_Z([point], max_dim)

        if combination_function_on_Z(point, state) == FULL:
            new_LP_set.add(point)
            new_LP_set.update(vc)
        elif state[y, x] == EMPTY:
            for point_vc in vc:
                x_vc, y_vc = point_vc
                if state[y_vc, x_vc] == FULL:
                    if get_vortex_speed(point, ini_point) > 10:
                        new_state[y, x] = FULL
                        new_LP_set.add(point_vc)
                        new_LP_set.update(get_vc_Z([point_vc], max_dim))
                    break
    return list(new_LP_set), new_state, min_vortex_speed_this_step

def event_scheduling_on_Z():
    """
    Simulates a simple propagation using event scheduling.
    Returns:
        tuple: A tuple containing two lists:
            - simpleEvolution: A list of numpy arrays representing the state of the evolution at each step.
            - vortex_speed_evolution: A list of floats representing the maximum vortex speed at each step.
    """
    # ... existing code ...
    size = (100, 100)
    max_dim = size

    # Modifying the initial conditions to start the propagation
    initial_propagation = np.full(size, EMPTY)
    ini_point = (70, 70)  # (x, y)
    x, y = ini_point
    initial_propagation[y, x] = FULL
    
    # LP is the list of active nucleous for the next step
    LP = get_nc_Z([ini_point], [])
    
    # Get the initial vicinity to schedule the first step
    vicinity = get_vc_Z(LP, max_dim)
    
    # The event list for the first step is the union of the nucleous and its vicinity
    event_list = list(set(LP) | set(vicinity))

    # Variable that will contain all the states we define on the execution of the model.
    simpleEvolution = [initial_propagation]
    vortex_speed_evolution = [0.0]

    # Number of steps to execute the evolution function
    n_steps = 100

    # n_steps represents the finish condition of the simulation.
    for step in range(n_steps):
        new_state = simpleEvolution[-1].copy()

        # The new nucleous is the entire set of points that changed state in the last step
        nucleous = event_list
        
        # Calculate the new state and the next set of points to check
        LP_new, new_state, max_vortex_speed = evolution_function_on_Z(nucleous, simpleEvolution[-1], new_state, ini_point)
        
        # The new event list is the new nucleous
        event_list = LP_new

        simpleEvolution.append(new_state)
        vortex_speed_evolution.append(max_vortex_speed)

    return simpleEvolution, vortex_speed_evolution

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
    Adds a vectorial map (polygon) to the layers array, ensuring no duplicate points.
    This function checks if the given vectorialMap is a single point. If so, it searches for and removes any existing 
    identical point in the layers array. After that, it adds the vectorial map to the layers array if it is not already present.
    Args:
        vectorialMap (dict): A dictionary representing a polygon with 'id' and 'points'.
        layersArray (list): A list of vectorial maps, where each map is a dictionary.
    Returns:
        None
    """
    # Check if vectorialMap is a single point
    if len(vectorialMap['points']) == 1:
        point_to_add = vectorialMap['points'][0]
        
        # Search for and remove the point in layersArray if it exists
        for layer in layersArray:
            if len(layer['points']) == 1:
                existing_point = layer['points'][0]
                if existing_point == point_to_add:
                    layersArray.remove(layer)
                    break
    
    # Add the vectorialMap to layersArray
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
    
def simplifyVectorialMap(vectorialMap, ini_point):
    """
    Simplifies a vectorial map by merging polygons that are within a certain distance (defined by the vicinity) of each other.
    Args:
        vectorialMap (list): A list of dictionaries, where each dictionary represents a polygon with an 'id' and 'points'.
                             The 'id' is a unique identifier for the polygon, and 'points' is a list of coordinates.
        ini_point (tuple): The initial point of the simulation, used for distance calculations.
    Returns:
        list: A simplified vectorial map, where polygons that are close to each other are merged. Each element in the 
              returned list is a dictionary with an 'id' and 'points'. The 'points' represent the exterior coordinates 
              of the merged polygons.
    """
    # Group polygons by their IDs
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
            seen = set()
            unique_points = []
            for pt in polygon['points']:
                if pt not in seen:
                    unique_points.append(pt)
                    seen.add(pt)
            polygons_by_id[poly_id].append(Polygon(unique_points))
    
    simplified_vectorial_map = []
    
    # Join polygons with the same ID that are within a certain distance
    for poly_id, polygons in polygons_by_id.items():
        merged_polygons = []
        while polygons:
            base_polygon = polygons.pop(0)
            to_merge = [base_polygon]
            for other_polygon in polygons[:]:
                if base_polygon.distance(other_polygon) <= 5:
                    to_merge.append(other_polygon)
                    polygons.remove(other_polygon)
            merged_polygon = unary_union(to_merge)
            merged_polygons.append(merged_polygon)
        
        # Add the merged polygons to the new vectorialMap
        for merged_polygon in merged_polygons:
            if isinstance(merged_polygon, MultiPolygon):
                for poly in merged_polygon.geoms:
                    points = list(poly.exterior.coords)
                    if points:
                        max_dist = max(get_distance(p, ini_point) for p in points)
                        filtered_points = [p for p in points if get_distance(p, ini_point) >= max_dist]
                        if filtered_points:
                            simplified_vectorial_map.append({'id': poly_id, 'points': filtered_points})
            elif isinstance(merged_polygon, Polygon):
                points = list(merged_polygon.exterior.coords)
                if points:
                    max_dist = max(get_distance(p, ini_point) for p in points)
                    filtered_points = [p for p in points if get_distance(p, ini_point) >= max_dist]
                    if filtered_points:
                        simplified_vectorial_map.append({'id': poly_id, 'points': filtered_points})
            elif isinstance(merged_polygon, Point):
                simplified_vectorial_map.append({'id': poly_id, 'points': [(merged_polygon.x, merged_polygon.y)]})
            else:
                # Handle the case where merged_polygon is a collection of points
                points = []
                for geom in merged_polygon.geoms:
                    if isinstance(geom, Point):
                        points.append((geom.x, geom.y))
                    elif isinstance(geom, Polygon):
                        points.extend(list(geom.exterior.coords))

                if points:
                    # Create a polygon from the points. To assure not add interior points.
                    points_ext = remove_interior_points(points)
                    points_ext = sort_points(points_ext)
                    max_dist = max(get_distance(p, ini_point) for p in points_ext)
                    filtered_points = [p for p in points_ext if get_distance(p, ini_point) >= max_dist]
                    if filtered_points:
                        simplified_vectorial_map.append({'id': poly_id, 'points': filtered_points})

    return simplified_vectorial_map

def remove_interior_points(points):
    """
    Removes interior points from a list of points.
    A point is considered interior if there are points above, below, 
    to the left, and to the right of it.
    Args:
        points (list of tuple): A list of points where each point is represented 
                                as a tuple (x, y).
    Returns:
        list of tuple: A list of points with all interior points removed.
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

    points_set = set(points)
    result = [point for point in points if not is_interior(point, points_set)]
    return result

def get_center(points):
    """
    Calculates the centroid of a set of points. If there is only one point, it returns the point itself.
    Args:
        points (list of tuples): A list of (x, y) coordinates.
    Returns:
        tuple: The (x, y) coordinates of the centroid, or the point itself if it's a single point, or None if the list is empty.
    """
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    count = len(points)
    return (sum_x / count, sum_y / count)

#m:n-CAk on R functions
def evolution_function_on_R(points, evolution, new_state, ini_point):
    """
    Simulates the evolution of a simple propagation on a grid based on the cell state.
    Args:
        points (list): The list of points to process.
        evolution (list): The current state of the evolution grid.
        new_state (list): The updated state of the evolution grid.
    Returns:
        tuple: (new_LP, new_state, min_vortex_speed)
    """
    new_LP_set = set()
    max_dim = [100, 100]
    min_vortex_speed_this_step = 0.0

    # Calculate speed for all active points
    if points:
        min_vortex_speed_this_step = min(get_vortex_speed(p, ini_point) for p in points)

    for point in points:
        i, j = point
        vc = get_vc_R([point], max_dim, ini_point)
        
        # Check current point's state
        if combination_function_on_R(point, evolution) == FULL:
            # If the point is already FULL, it remains active, and so do its neighbors
            new_LP_set.add(point)
            new_LP_set.update(vc)
        else: # If the point is EMPTY
            # Check its neighbors
            for point_vc in vc:
                # If a neighbor is FULL
                if combination_function_on_R(point_vc, evolution) == FULL:
                    if get_vortex_speed(point, ini_point) > 10:
                        # The current point becomes FULL
                        addVectorialMap({'id': FULL, 'points': [(i,j)]}, new_state)
                        
                        # The neighbor that caused the change, and its neighbors, become active
                        new_LP_set.add(point_vc)
                        new_LP_set.update(get_vc_R([point_vc], max_dim, ini_point))
                        break # Move to the next point in `points`
    
    #we must do the same for the new_state, remove the points that have a distance that is smaller to the maximum distance found between any point and ini_point
    #first we calculate the max_distance, but now for the points in new_state
    all_points_in_new_state = [point for polygon in new_state for point in polygon['points']]
    if not all_points_in_new_state:
        max_distance = 0
    else:
        max_distance = max(get_distance(ini_point, p) for p in all_points_in_new_state)
    
    # Filter out polygons in new_state that have any point within max_distance from ini_point
    new_state = [polygon for polygon in new_state if any(get_distance(p, ini_point) == max_distance for p in polygon['points'])]

    #Prior to returning, we need to remove the points that have a distance that is smaller to the maximum distance found between any point and ini_point
    # Calculate the maximum distance from ini_point to any point in new_LP_set
    #First we chack if theres any point in new_LP_set, since the expansion will end due to speed is small.
    if not new_LP_set:
        return [], new_state, min_vortex_speed_this_step
    max_distance_lp = max(get_distance(ini_point, p) for p in new_LP_set)

    new_LP_set = {p for p in new_LP_set if get_distance(p, ini_point) >= max_distance_lp}


    return list(new_LP_set), new_state, min_vortex_speed_this_step

def event_scheduling_on_R():
    """
    Simulates a simple evolution over a specified number of steps.
    This function reads initial conditions from IDRISI vector files for propagation.
    Returns:
        tuple: A tuple containing two lists:
            - propagationEvolution: A list of states representing the evolution of the propagation.
            - vortex_speed_evolution: A list of floats representing the maximum vortex speed at each step.
    """
    # Read initial propagation data
    fileEvolution = 'simple.vec'
    polygonsPropagation = read_idrisi_vector_file(fileEvolution)
    max_dim = [100, 100]

    # Initialize the list of active points (nucleous) from the initial polygon points
    initial_points = []
    for polygon in polygonsPropagation:
        initial_points.extend(polygon['points'])
    
    # The initial nucleous is the set of starting points
    nucleous = get_nc_R(initial_points, [])
    ini_point = get_center(nucleous)

    # The initial vicinity
    vicinity = get_vc_R(nucleous, max_dim, ini_point)
    
    # The event list for the first step is the union of the nucleous and its vicinity
    event_list = list(set(nucleous) | set(vicinity))

    # Variable that will contain all the states of the model's execution.
    propagationEvolution = [polygonsPropagation]
    vortex_speed_evolution = [0.0]

    # Number of steps to execute the evolution function
    n_steps = 100

    for step in range(n_steps):
        # new_state must be initialized empty for each iteration
        new_state = []
        
        # The points to process in this step are the ones in the event list
        points_to_process = event_list
        
        # Calculate the new state and the next set of points to check
        LP_new, new_polygons, max_vortex_speed = evolution_function_on_R(points_to_process, propagationEvolution[-1], new_state, ini_point)
        
        event_list = LP_new

        # Combine the previous state with the new polygons generated
        combined_state = propagationEvolution[-1] + new_polygons
        
        # Add the new state to the evolution list
        propagationEvolution.append(simplifyVectorialMap(combined_state, ini_point))
        vortex_speed_evolution.append(max_vortex_speed)

    return propagationEvolution, vortex_speed_evolution

##############################################################
#m:n-CAk main functions
##############################################################

def get_vc_Z(points, max_dim):
    """
    Vicinity function. Given a list of points, returns the union of their Moore neighborhoods (8 neighbors, excluding the center), avoiding duplicates.
    Args:
        points (list of tuples): list of (x, y)
        max_dim (tuple): (max_x, max_y)
    Returns:
        list: List of (x, y) tuples (vicinity)
    """
    max_x, max_y = max_dim
    points_set = set(points)
    vicinity = []
    for x, y in points:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Exclude the center
                nx, ny = x + dx, y + dy
                if 0 <= nx < max_x and 0 <= ny < max_y:
                    neighbor = (nx, ny)
                    if neighbor not in points_set:
                        vicinity.append(neighbor)
    # Remove duplicates
    vicinity = list(set(vicinity))
    return vicinity

def get_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    #Rounded to the first decimal to avoid problems with the calculation of the distance
    return round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2), 1)

def get_cardinal_points(point, radius=1.0, variablePoints=True, point_distance=0.3):
    """
    Given a point, returns points on the perimeter of a circle at a given radius.

    If variablePoints is False (default), it returns 8 cardinal and diagonal points.
    If variablePoints is True, it generates a variable number of points based on the
    circle's perimeter and the desired distance between points.

    Args:
        point (tuple): (x, y) coordinates of the center.
        radius (float): The radius of the circle.
        variablePoints (bool): If True, generate points based on point_distance.
        point_distance (float): The desired distance between points on the perimeter. A small value (e.g., 0.5) is recommended for better resolution. Buet small values impacts deeply on the performance.

    Returns:
        list: A list of tuples representing the points on the circle.
    """
    x, y = point
    
    if variablePoints:
        points = []
        perimeter = 2 * math.pi * radius
        num_points = int(perimeter / point_distance)
        if num_points == 0:
            return []
            
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append((px, py))
        return points
    else:
        diag_dist = radius / math.sqrt(2)
        return [
            # Cardinal points
            (x, y + radius),
            (x, y - radius),
            (x - radius, y),
            (x + radius, y),
            # Diagonal points
            (x + diag_dist, y + diag_dist),
            (x - diag_dist, y + diag_dist),
            (x - diag_dist, y - diag_dist),
            (x + diag_dist, y - diag_dist)
        ]

def get_vc_R(points, max_dim, ini_point):
    """
    Vicinity function for the continuous case (R^2).
    For each point, it generates a circle of neighbors. The radius of the circle is the maximum distance
    found between any point in the input list and the ini_point, and this value is updated globally.
    Args:
        points (list of tuples): list of (x, y)
        max_dim (tuple): (max_x, max_y)
        ini_point (tuple): The origin point of the simulation for distance comparison.
    Returns:
        list: List of (x, y) tuples (vicinity), without duplicates.
    """
    global maxRadius
    max_x, max_y = max_dim
    vicinity_set = set()

    if not points:
        return []
    
    # Calculate the maximum distance from ini_point to any point in the input list.
    current_max_radius = max(get_distance(p, ini_point) for p in points) if points else 0

    # Update the global maxRadius if the current one is larger
    if current_max_radius > maxRadius:
        maxRadius = current_max_radius

    radius_to_use = maxRadius if maxRadius > 0 else 1.0
    
    for point in points:
        #neighbors = get_cardinal_points(point, radius=radius_to_use)
        neighbors = get_cardinal_points(point)
        for neighbor in neighbors:
            nx, ny = neighbor
            if 0 <= nx < max_x and 0 <= ny < max_y:
                vicinity_set.add(neighbor)

    # Ensure the original points are not in their own vicinity
    points_set = set(points)
    
    return list(vicinity_set - points_set)


def get_nc_Z(prev_nucleous, last_vicinity):
    """
    Nucleous function for the discrete case (Z^2).
    Given the last vicinity (list of points) and the previous nucleous (list of points),
    returns the union of both as the new nucleous, deduplicated.
    Args:
        last_vicinity (list of tuples): The last computed vicinity points.
        prev_nucleous (list of tuples): The nucleous points from the previous iteration.
    Returns:
        list: List of (x, y) tuples representing the new nucleous (deduplicated).
    """
    return list(set(last_vicinity) | set(prev_nucleous))

def get_nc_R(prev_nucleous, last_vicinity):
    """
    Nucleous function for the continuous case (R^2). Returns the Moore neighborhood (continuous) of the point in the previous iteration, excluding the center.
    Given the last vicinity (list of points) and the previous nucleous (list of points),
    returns the union of both as the new nucleous, deduplicated.
    Args:
        last_vicinity (list of tuples): The last computed vicinity points.
        prev_nucleous (list of tuples): The nucleous points from the previous iteration.
    Returns:
        list: List of (x, y) tuples representing the new nucleous (deduplicated).
    """
    return list(set(last_vicinity) | set(prev_nucleous))

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
    x, y = point
    return layer[y, x]
    #return point

def get_vortex_speed(point, origen):
    """
    Calculates the tangential speed of a point in a Rankine vortex.
    Args:
        point (tuple): (x, y) coordinates of the point.
        origen (tuple): (x, y) coordinates of the vortex center.
    Returns:
        float: The tangential speed of the point.
    """
    gamma, radius = get_vortex_params()
    
    x1, y1 = point
    x2, y2 = origen
    r = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    if r == 0:
        # If the point is at the center of the vortex, return a very high speed (or handle as needed)
        # This is a special case, as the speed at the center of a Rankine vortex is theoretically infinite.
        # but we return a large value to avoid division by zero.
        return 10000.0
    
    # Inside the vortex core (solid-body rotation)
    if r <= radius:
        return (gamma * r) / (2 * math.pi * radius**2)
    # Outside the vortex core (potential vortex)
    else:
        return gamma / (2 * math.pi * r)

def get_vortex_params():
    """
    Defines the parameters for the Rankine vortex model.
    Returns:
        tuple: (gamma, radius)
            - gamma (float): The circulation of the vortex.
            - radius (float): The radius of the vortex core.
    """
    gamma = 1000.0  # Circulation constant
    radius = 1.0   # Radius of the vortex core
    return gamma, radius

##############################################################
#Main
##############################################################

if __name__ == "__main__":

    # Definition of S_state for propagation main layer.
    FULL = 1
    EMPTY = 0

    domain = 'Z'

    domain = input(".git\nPlease, select the domain to work on (Z or R): ")
    if domain != 'Z' and domain != 'R':
        print("Invalid domain. Please select 'Z' or 'R'.")
    else:

        if domain == 'Z':
            simpleEvolution, vortex_speed_evolution = event_scheduling_on_Z()
        else:
            simpleEvolution, vortex_speed_evolution = event_scheduling_on_R()

        results_window(domain, simpleEvolution, vortex_speed_evolution)