##############################################################
# m_n-CAk_Wildfire.py
# Created on: 2025-06-30
# Author: Pau Fonseca i Casas
# Copyright: Pau Fonseca i Casas
# Description: This script simulates the spread of a wildfire using an m:n-CAk cellular automaton model over Z^2 and R^2
# All the layers share the same coordinate system, therefore, the function to change the basis is not needed.
# This code is intended as a proof of concept for using the m:n-CAk cellular automaton.
##############################################################

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union

##############################################################
# Model execution parameters
##############################################################

WIND_MULT = 1.5  # Multiplier for wind speed

STEEPS = 100  # Number of simulation steps

wind_as_main_layer = True  # Set to True to use wind as the main layer; retrofeedback can appear due to executing the two functions.

# Experimental feature: set to True to enable simplification in the representations of the polygons in the vectorial case.
plot_polygons = False  # This allows representing only the perimeter of the polygons.

# DEBUG Variables
DEBUG = False  # Set to True to enable debugging output
D_i = 100  # i component of the point to debug
D_j = 70  # j component of the point to debug

##############################################################
# Constants and Global Variables
##############################################################

# Fire state constants
UNBURNED = 2
BURNING = 1
BURNED = 0

# wind state constants
CALM = 0
WIND = 1

##############################################################
# SimulationState class to encapsulate point structures
##############################################################
class SimulationState:
    """
    Encapsulates the simulation state.
    Organizes nuclei (points in this example) by ID and layer type for the continuous case.
    """
    def __init__(self):
        # Working structures for the current step
        self.fire_points_by_id = {}      # {'0': [(x1,y1), (x2,y2), ...], '1': [...], '2': [...]}
        self.humidity_points_by_id = {}   # {'id1': [(x1,y1), ...], 'id2': [...], ...}
        self.vegetation_points_by_id = {} # {'id1': [(x1,y1), ...], 'id2': [...], ...}
        self.wind_points_by_id = {}      # {'id1': [(x1,y1), ...], 'id2': [...], ...}
        
        # Final structures for the completed step
        self.fire_points_by_id_final = {}      # {'0': [(x1,y1), (x2,y2), ...], '1': [...], '2': [...]}
        self.humidity_points_by_id_final = {}   # {'id1': [(x1,y1), ...], 'id2': [...], ...}
        self.vegetation_points_by_id_final = {} # {'id1': [(x1,y1), ...], 'id2': [...], ...}
        self.wind_points_by_id_final = {}      # {'id1': [(x1,y1), ...], 'id2': [...], ...}
    
    def get_points_by_id(self, layer_type):
        """Get the working points dictionary for a given layer type."""
        if layer_type == "fire":
            return self.fire_points_by_id
        elif layer_type == "humidity":
            return self.humidity_points_by_id
        elif layer_type == "vegetation":
            return self.vegetation_points_by_id
        elif layer_type == "wind":
            return self.wind_points_by_id
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")
    
    def get_points_by_id_final(self, layer_type):
        """Get the final points dictionary for a given layer type."""
        if layer_type == "fire":
            return self.fire_points_by_id_final
        elif layer_type == "humidity":
            return self.humidity_points_by_id_final
        elif layer_type == "vegetation":
            return self.vegetation_points_by_id_final
        elif layer_type == "wind":
            return self.wind_points_by_id_final
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")
    
    def finalize_step(self):
        """Copy working structures to final structures."""
        import copy
        self.fire_points_by_id_final = copy.deepcopy(self.fire_points_by_id)
        self.humidity_points_by_id_final = copy.deepcopy(self.humidity_points_by_id)
        self.vegetation_points_by_id_final = copy.deepcopy(self.vegetation_points_by_id)
        self.wind_points_by_id_final = copy.deepcopy(self.wind_points_by_id)


##############################################################
# Auxiliary functions to obtain data and to represent the data
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

    Args:
        img_path (str): The path to the IDRISI raster image file.

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
def plot_vectorial_polygons(ax, polygons, id, radius=1, type='fire', title='No title', exclusive_plot=[]):
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
    def safe_area(p):
        pts = p.get('points', [])
        if not isinstance(pts, list) or len(pts) < 3:
            return 0  # Area 0 for points or lines or empty
        try:
            return np.abs(np.sum([x0*y1 - x1*y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + [pts[0]])])) / 2
        except Exception:
            return 0
    polygons = sorted(polygons, key=safe_area)

    # For the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='BURNING'),
        Line2D([0], [0], color='black', lw=2, label='BURNED'),
        Line2D([0], [0], color='green', lw=2, label='UNBURNED')
    ]

    for polygon in polygons:
        polygon_id = polygon['id']
        points = polygon['points']

        # Filter polygons based on exclusive_plot list or default behavior
        if exclusive_plot is not None and len(exclusive_plot) > 0:
            if polygon_id not in exclusive_plot:
                continue
        # Determine color based on the ID
        if polygon_id == 1:
            edge_color = 'red'      # BURNING
        elif polygon_id == 0:
            edge_color = 'black'    # BURNED
        elif polygon_id == 2:
            edge_color = 'green'    # UNBURNED
        else:
            edge_color = 'gray'     # Others

        # Plot the polygon with transparent fill and colored edges
        polygon_shape = plt.Polygon(points, closed=True, edgecolor=edge_color, facecolor='none', fill=True)
        ax.add_patch(polygon_shape)

        # Plot the edges of the polygon
        for j in range(len(points)):
            x1, y1 = points[j]
            x2, y2 = points[(j + 1) % len(points)]
            ax.plot([x1, x2], [y1, y2], color=edge_color)

        # Annotate the polygon with its ID
        centroid_x = sum(x for x, y in points) / len(points)
        centroid_y = sum(y for x, y in points) / len(points)
        ax.text(centroid_x, centroid_y, str(polygon_id), fontsize=12, ha='center', va='center', color='black')

        # Plot each point with a circle on top of everything (only for displayed polygons)
        circle = False
        if circle:
            for (x, y) in points:
                circ = plt.Circle((x, y), radius, color='blue', fill=False)
                ax.add_patch(circ)
                ax.plot(x, y, 'ro')

    ax.set_aspect('equal', adjustable='box')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Layer {id} - '+title)
    plt.grid(True)

    # Add the legend
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

def plot_vectorial_dots(ax, polygons, id, radius=1, type='fire', title='No title', exclusive_plot=[]):
    """
    Draws only the points of the polygons in a scatter plot, grouping by color for maximum efficiency.
    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis where to draw.
        polygons (list of dict): List of dictionaries with 'id' and 'points'.
        id (int): Identifier of the current layer.
        radius (float, optional): Ignored, only for compatibility.
        type (str, optional): Default type if ID is not used.
        title (str, optional): Title of the plot.
        exclusive_plot (list, optional): List of IDs to show exclusively.
    """
    ax.clear()
    from matplotlib.lines import Line2D
    if type=="fire":
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='BURNING'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='BURNED'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, label='UNBURNED')
        ]
        colors = {1: 'red', 0: 'black', 2: 'white'}
    elif type=="vegetation":
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='VEGETATION')
        ]
        colors = {1: 'green', 0: 'black', 2: 'white'}
    elif type=="humidity":
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=8, label='HUMIDITY')
        ]
        colors = {1: 'lightblue', 0: 'black', 2: 'darkblue'}
    elif type=="wind":
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, label='CALM'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=8, label='WIND')
        ]
        colors = {1: 'lightblue', 0: 'white', 2: 'blue'}

    grouped_points = {1: [], 0: [], 2: [], 'other': []}
    # Group points by ID
    for polygon in polygons:
        polygon_id = polygon['id']
        points = polygon['points']
        if exclusive_plot is not None and len(exclusive_plot) > 0:
            if polygon_id not in exclusive_plot:
                continue
        if polygon_id in grouped_points:
            grouped_points[polygon_id].extend(points)
        else:
            grouped_points['other'].extend(points)
    # Draw all points of each group in a single scatter call
    point_size = 4  # Reduced size for visual efficiency
    for pid, pts in grouped_points.items():
        if pts:
            xs, ys = zip(*pts)
            c = colors.get(pid, "lightsteelblue")
            ax.scatter(xs, ys, s=point_size, color=c, label=f'ID {pid}')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Layer {id} - '+title)
    ax.grid(True)
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

def plot_vectorial(ax, polygons, id, type="fire", radius=1, title='No title', exclusive_plot=[]):
    if plot_polygons:
        plot_vectorial_polygons(ax, polygons, id, radius=radius, type=type, title=title, exclusive_plot=exclusive_plot)
    else:
        # plot_continuous_dots(ax, id, type, radius=radius, color=color, title=title, exclusive_plot=exclusive_plot)
        plot_vectorial_dots(ax, polygons, id, radius=radius, type=type, title=title, exclusive_plot=exclusive_plot)

def plot_continuous_dots(ax, id, sim_state, type="fire", radius=1, color='fire', title='No title', exclusive_plot=[]):
    """
    Draws the points from a *_POINTS_BY_ID_FINAL structure in a scatter plot, grouping by color according to the ID.
    Chooses the dictionary according to the 'type' argument from sim_state.
    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis where to draw.
        id (int): Identifier of the current layer (only for title).
        sim_state (SimulationState): The simulation state object
        type (str): Type of layer ('fire', 'humidity', 'vegetation', 'wind').
        radius (float, optional): Ignored, only for compatibility.
        color (str, optional): Default color if ID is not used.
        title (str, optional): Title of the plot.
        exclusive_plot (list, optional): List of IDs to show exclusively.
    """
    ax.clear()
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='BURNING'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='BURNED'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='UNBURNED')
    ]
    colors = {1: 'red', 0: 'black', 2: 'green'}
    grouped_points = {1: [], 0: [], 2: [], 'other': []}
    # Get the corresponding structure from sim_state according to the type
    points_by_id_final = sim_state.get_points_by_id_final(type)
    # Group points by ID
    for id_key, points in points_by_id_final.items():
        if exclusive_plot is not None and len(exclusive_plot) > 0:
            if id_key not in exclusive_plot:
                continue
        if id_key in grouped_points:
            grouped_points[id_key].extend(points)
        else:
            grouped_points['other'].extend(points)
    # Draw all points of each group in a single scatter call
    point_size = 4  # Reduced size for visual efficiency
    for pid, pts in grouped_points.items():
        if pts:
            xs, ys = zip(*pts)
            c = colors.get(pid, color)
            ax.scatter(xs, ys, s=point_size, color=c, label=f'ID {pid}')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Layer {id} - '+title)
    ax.grid(True)
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))


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

    # Clear the axis and remove previous legends and titles
    ax.clear()
    # Remove previous legend if it exists
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # Remove previous title (set to empty string)
    ax.set_title("")

    # Define the color and normalization according to the type
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
        cmap = plt.cm.colors.ListedColormap(['black', 'lightblue', 'darkblue'])
        bounds = [0, 1, 2, 3]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    elif type == "wind":
        cmap = plt.cm.colors.ListedColormap(['black', 'lightblue', 'darkblue', 'cyan'])
        bounds = [0, 1, 2, 21]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    else:
        cmap = plt.cm.viridis
        norm = None

    # Debug unique values
    unique_values = np.unique(matrix)
    print(f"DEBUG - Unique values in matrix: {unique_values}")

    # Count states
    zero_count = np.sum(matrix == 0)
    one_count = np.sum(matrix == 1)
    two_count = np.sum(matrix == 2)
    if type == "fire":
        print(f"DEBUG - Counts: BURNED={zero_count}, BURNING={one_count}, UNBURNED={two_count}")
    elif type == "vegetation":
        print(f"DEBUG - Counts: DEAD={zero_count}, ALIVE={two_count}")
    elif type == "humidity":
        print(f"DEBUG - Counts: DRY={zero_count}, HUMID={two_count}")
    elif type == "wind":
        print(f"DEBUG - Counts: CALM={zero_count}, WIND={one_count}")

    # Display the matrix
    cax = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')

    # Title and labels
    if type == "fire":
        ax.set_title(f'Layer {id} - {title}\n(Black=BURNED, Red=BURNING, Green=UNBURNED)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='black', label=f'BURNED ({zero_count})'),
            Patch(facecolor='red', label=f'BURNING ({one_count})'),
            Patch(facecolor='green', label=f'UNBURNED ({two_count})')
        ]
    elif type == "vegetation":
        ax.set_title(f'Layer {id} - {title}\n(Black=DEAD, Green=ALIVE)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='black', label=f'DEAD ({zero_count})'),
            Patch(facecolor='green', label=f'ALIVE ({two_count})')
        ]
    elif type == "humidity":
        ax.set_title(f'Layer {id} - {title}\n(Light Blue=HUMID, Yellow=DRY)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='black', label=f'DRY ({zero_count})'),
            Patch(facecolor='darkblue', label=f'HUMID ({two_count})')
        ]
    elif type == "wind":
        ax.set_title(f'Layer {id} - {title}\n(Light Blue=CALM, Dark Blue=WIND)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        legend_elements = [
            Patch(facecolor='black', label=f'CALM ({zero_count})'),
            Patch(facecolor='lightblue', label=f'WIND ({10000-zero_count})')            
        ]
    else:
        ax.set_title(f'Layer {id} - {title}')
        legend_elements = []

    # Invert the y-axis so that 0 is at the bottom
    ax.invert_yaxis()

    # Add the legend only if appropriate
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

    # Save the matrix to a file in IDRISI format
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
            # if layer_type == "wind":
            #    plot_raster(ax, layerEvolution[frame], id=0, type=layer_type, color='red', title=f'State at Frame {frame}')
            #else:
            plot_vectorial(ax, layerEvolution[frame], id=0, type=layer_type, radius=1, title=f'State at Frame {frame}')
        # Force complete canvas update
        canvas.flush_events()
        canvas.draw_idle()

    # Label to display the current value of the slider
    slider_label = tk.Label(root, text="Steeps")
    slider_label.pack(side=tk.BOTTOM, pady=5)

    # Slider for selecting the matrix
    # Initially for fire
    slider = ttk.Scale(root, from_=0, to=len(fireEvolution) - 1, orient=tk.HORIZONTAL, command=lambda val: on_slider_change(val, fireEvolution, "fire"))
    slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
     
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=20)

    # Functions to change the layerEvolution
    def set_fire_evolution():
        slider.config(command=lambda val: on_slider_change(val, fireEvolution, "fire"))
        slider_label.config(text="Fire")
        if domain == 'Z':
            plot_raster(ax, fireEvolution[0], id=0, type="fire", color='red', title='Fire - Initial State')
        else:
            plot_vectorial(ax, fireEvolution[0], id=0, type="fire", radius=1, title='Fire - Initial State')
        slider.set(0)
        canvas.flush_events()
        canvas.draw_idle()

    def set_wind_evolution():
        slider.config(command=lambda val: on_slider_change(val, windEvolution, "wind"))
        slider_label.config(text="Wind")
        if domain == 'Z':
            plot_raster(ax, windEvolution[0], id=0, type="wind", color='purple', title='Wind - Initial State')
        else:
            plot_vectorial(ax, windEvolution[0], id=0, type="wind", radius=1, title='Wind - Initial State')
        slider.set(0)
        canvas.flush_events()
        canvas.draw_idle()

    def set_vegetation_evolution():
        slider.config(command=lambda val: on_slider_change(val, vegetationEvolution, "vegetation"))
        slider_label.config(text="Vegetation")
        if domain == 'Z':
            plot_raster(ax, vegetationEvolution[0], id=0, type="vegetation", color='green', title='Vegetation - Initial State')
        else:
            plot_vectorial(ax, vegetationEvolution[0], id=0, type="vegetation", radius=1, title='Vegetation - Initial State')
        slider.set(0)
        canvas.flush_events()
        canvas.draw_idle()

    def set_humidity_evolution():
        slider.config(command=lambda val: on_slider_change(val, humidityEvolution, "humidity"))
        slider_label.config(text="Humidity")
        if domain == 'Z':
            plot_raster(ax, humidityEvolution[0], id=0, type="humidity", color='darkblue', title='Humidity - Initial State')
        else:
            plot_vectorial(ax, humidityEvolution[0], id=0, type="humidity", radius=1, title='Humidity - Initial State')
        slider.set(0)
        canvas.flush_events()
        canvas.draw_idle()

    # Variable to keep track of the selected option
    selected_option = tk.StringVar(value="Fire")

    # Main Layers label
    main_label = ttk.Label(button_frame, text="Main Layers:")
    main_label.pack(side=tk.LEFT, padx=5)

    radio_fire = ttk.Radiobutton(button_frame, text="Fire", variable=selected_option, value="Fire", command=set_fire_evolution)
    radio_fire.pack(side=tk.LEFT, padx=10)

    # Wind radio button, disabled if wind_as_main_layer is False
    radio_wind = ttk.Radiobutton(button_frame, text="Wind", variable=selected_option, value="Wind", command=set_wind_evolution)
    radio_wind.pack(side=tk.LEFT, padx=10)
    if not wind_as_main_layer:
        radio_wind.state(['disabled'])

    # Secondary Layers label
    secondary_label = ttk.Label(button_frame, text="Secondary Layers:")
    secondary_label.pack(side=tk.LEFT, padx=5)

    radio_vegetation = ttk.Radiobutton(button_frame, text="Vegetation", variable=selected_option, value="Vegetation", command=set_vegetation_evolution)
    radio_vegetation.pack(side=tk.LEFT, padx=10)

    radio_humidity = ttk.Radiobutton(button_frame, text="Humidity", variable=selected_option, value="Humidity", command=set_humidity_evolution)
    radio_humidity.pack(side=tk.LEFT, padx=10)

    root.mainloop()

##############################################################
# m:n-CAk on Z specific functions
##############################################################
def evolution_function_on_Z(point, fire, vegetation, humidity, wind, new_state, new_vegetation, new_humidity, new_wind):
    """
    Simulates the evolution of a wildfire in an m:n-CAk cellular automaton model.
    Args:
        point (tuple): The coordinates (i, j) of the current cell.
        fire (ndarray): The current fire state of the grid, where each cell can be BURNING, UNBURNED, or BURNED.
        vegetation (ndarray): The current vegetation levels of the grid.
        humidity (ndarray): The current humidity levels of the grid.
        wind (ndarray): The current wind levels of the grid.
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
    # Getting the nucleus.
    nc = get_nc(point)
    i, j = nc
   
    # Obtain the wind speed
    wind_speed = combination_function_on_Z(nc, wind)

    if DEBUG:
        global D_i, D_j
        if i == D_i and i == D_j:
            print(f"DEBUG - Processing point {point} with state {fire[i, j]} and vegetation {vegetation[i, j]}")
    
    # Evolution function for fire layer
    cell_fire = combination_function_on_Z(nc,fire)
    # Getting the vicinity
    vc = get_vc_fire_Z(point, max_dim, wind_speed, cell_fire)

    if cell_fire == BURNING:
        cell_vegetion = combination_function_on_Z(nc,vegetation)
        if cell_vegetion <= 0:
            new_state[i, j] = BURNED
            new_LP.append([i, j])
            #new_LP.extend([elems for elems in get_vc_fire_Z(point, max_dim, wind_speed)])
            new_LP.extend([elems for elems in get_vc_fire_Z(point, max_dim, wind_speed, BURNED)])
        else:
            # Decrease the vegetation.
            new_vegetation[i, j] -= 1
            new_LP.append([i, j])
    elif cell_fire == UNBURNED:
        for point_vc in vc:
            # We must access information contained in other layers, therefore we will use the combination function
            # In this case, the points we will use are georeferenced by the same coordinates, therefore the combination function
            # is just returning the same point that will be used as the center of the vicinity.
            vicinity_cell_fire = combination_function_on_Z(point_vc, fire)
            if vicinity_cell_fire == BURNING:
                if humidity[i, j] > 0:
                    new_humidity[i, j] -= 1
                    new_LP.append([i, j])
                elif vegetation[i, j] > 0:
                    new_state[i, j] = BURNING
                    new_LP.append([i, j])
                    #new_LP.extend([elems for elems in get_vc_fire_Z(point, max_dim, wind_speed, cell_fire)])
                    new_LP.extend([elems for elems in get_vc_fire_Z(point, max_dim, wind_speed)])

    if wind_as_main_layer:
        # Evolution function for wind layer
        cell_wind = combination_function_on_Z(nc,wind)
        # Getting the vicinity.
        vc = get_vc_wind(point, max_dim)

        if cell_wind == CALM:
            for point_vc in vc:
                vicinity_cell_wind = combination_function_on_Z(point_vc,wind)
                if vicinity_cell_wind == WIND:
                    new_wind[i, j] = WIND
                    new_LP.append([i, j])
                    new_LP.extend([elems for elems in get_vc_wind(point, max_dim)])
                    break
        else: #wind is not CALM
            if cell_fire == BURNING:
                new_wind[i, j] += 1

    return new_LP,new_state, new_vegetation, new_humidity, new_wind

def event_scheduling_on_Z():
    """
    Simulates the evolution of a wildfire over a specified number of steps using event scheduling.
    This function reads vegetation and humidity data from IDRISI raster files, initializes the wildfire
    conditions, and iteratively updates the state of the wildfire, vegetation, and humidity layers.
    Returns:
        tuple: A tuple containing four lists:
            - fireEvolution: A list of numpy arrays representing the state of the wildfire at each step.
            - vegetationEvolution: A list of numpy arrays representing the state of the vegetation at each step.
            - humidityEvolution: A list of numpy arrays representing the state of the humidity at each step.
            - windEvolution: A list of numpy arrays representing the state of the wind at each step.
    """
    global STEEPS
    # Auxiliary functions to obtain layer information from files (IDRISI 32 format).
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
    wind_img_path = os.path.join(folder_path, 'no_wind.img')

    # Reading layer information
    # vegetation layer
    vegetation_data = read_idrisi_raster_file(vegetation_img_path)

    # humidity layer
    humidity_data = read_idrisi_raster_file(humidity_img_path)

    # wind layer
    wind_data = read_idrisi_raster_file(wind_img_path)
   
    # Define the size for the layers, the same for all
    size = (100, 100)

    # Auxiliary functions to convert the vector into a matrix of data for all layers.
    humidity_data = humidity_data.reshape(size)
    vegetation_data = vegetation_data.reshape(size)
    wind_data = wind_data.reshape(size)

    # Modify the initial conditions to start the wildfire
    initial_fire = np.full(size, UNBURNED)
    ini_point = [70, 70]
    max_dim = [100, 100]
    i, j = ini_point
    initial_fire[i, j] = BURNING
    LP = []

    # Add the initial point we change.
    LP.append(ini_point)

    # Obtain the wind speed
    wind_speed = combination_function_on_Z(ini_point, wind_data)

    # Getting the vicinity for fire main layer
    vc = get_vc_fire_Z(ini_point, max_dim, wind_speed)

    # Also adding the neighborhoods of this point, avoid duplicates
    for point in vc:
        if point not in LP:
            LP.append(point)

    if wind_as_main_layer:
        initial_point_wind = [50,50]
        i_w, j_w = initial_point_wind
        wind_data[i_w, j_w]=WIND
        # Adding the initial point we change.
        LP.append(initial_point_wind)
        #Getting the vicinity for wind main layer
        vc = get_vc_wind(initial_point_wind, max_dim)
        # Also adding the neighborhoods of this point, avoid duplicates
        for point in vc:
            if point not in LP:
                LP.append(point)

    # Variable that will contain all the states we define during the execution of the model.
    fire_evolution = [initial_fire]
    vegetation_evolution = [vegetation_data]
    humidity_evolution = [humidity_data]
    wind_evolution = [wind_data]

    # Number of steps to execute the evolution function
    n_steps = STEEPS

    # n_steps represents the termination condition of the simulation.
    for _ in range(n_steps):
        LP_rep = []
        LP_new = []
        new_state = fire_evolution[-1].copy()
        new_vegetation = vegetation_evolution[-1].copy()
        new_humidity = humidity_evolution[-1].copy()
        new_wind = wind_evolution[-1].copy()
        # Event Scheduling simulation engine, where LP is the event list.
        for point in LP:
            LP_new, new_state, new_vegetation, new_humidity, new_wind = evolution_function_on_Z(point, fire_evolution[-1], vegetation_evolution[-1], humidity_evolution[-1], wind_evolution[-1], new_state, new_vegetation, new_humidity, new_wind)
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]
        
        fire_evolution.append(new_state)
        vegetation_evolution.append(new_vegetation)
        humidity_evolution.append(new_humidity)
        wind_evolution.append(new_wind)

        print(f"Step {_+1}/{n_steps} completed. Fire evolution size: {len(fire_evolution[-1])}, Vegetation size: {len(vegetation_evolution[-1])}, Humidity size: {len(humidity_evolution[-1])}, Wind size: {len(wind_evolution[-1])}")

    return fire_evolution, vegetation_evolution, humidity_evolution, wind_evolution

##############################################################
# m:n-CAk on R specific functions
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
    
    # Handle the special case where polygon_points is a line segment
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

def find_polygon_id(point, polygons, type=None, sim_state=None):
    """
    Finds the ID of the smallest polygon that contains a given point.
    Args:
        point (tuple): A tuple representing the coordinates of the point (x, y).
        polygons (list): A list of dictionaries, where each dictionary represents a polygon with the keys:
            - 'id' (any): The identifier of the polygon.
            - 'points' (list): A list of tuples representing the vertices of the polygon.
        type (str): The type of layer for searching in simulation state.
        sim_state (SimulationState): The simulation state object.
    Returns:
        any: The ID of the smallest polygon that contains the point. If no polygon contains the point, returns None.
    """
    global DEBUG, D_i, D_j

    i, j = point
    # Start debugging
    if DEBUG:
        if(i == D_i and j == D_j):
            print(f"DEBUG - Searching for point {point} in polygons")
    # End debugging

    smallest_polygon_id = None
    smallest_area = float('inf')
    smallest_polygon_id = float('inf')
    current_polygon_id = float('inf')
    current_area = float('inf')
    
    # Check if the point is in the simulation state identified by type
    polygon_id_dictionary = None
    if type is not None and sim_state is not None:
        polygon_id_dictionary = get_point_id_from_global_structure(point, type, sim_state)

    # If the point is in the simulation state, return the ID directly
    if polygon_id_dictionary is not None:
        return polygon_id_dictionary
    else:
        # If the point is not in the simulation state, proceed to check polygons
        for polygon in polygons:
            if is_point_in_polygon(point, polygon['points']):
                if len(polygon['points']) == 1:  # if the polygon is a point
                    current_area = -1
                    current_polygon_id = polygon['id']
                elif len(polygon['points']) == 2:  # if the polygon is a line segment
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
    # If no polygon was found, print warning and return -1
        if smallest_polygon_id is None or smallest_polygon_id == float('inf'):
            print(f"WARNING: No polygon found for point {point} with type '{type}'. Returning -1.")
            return -1
        return smallest_polygon_id

def remove_points_from_global_structure(points_to_remove, type, sim_state):
    """
    Removes a list of points from the simulation state structure of the specified type.
    Searches for each point in points_to_remove across all IDs and removes them.
    
    Args:
        points_to_remove (list): List of points (tuples) to remove
        type (str): Type of structure ("fire", "humidity", "vegetation", "wind")
        sim_state (SimulationState): The simulation state object
    Returns:
        None
    """
    global DEBUG, D_i, D_j
    
    # Get the corresponding structure according to the type
    points_by_id = sim_state.get_points_by_id(type)
    
    # Convert points_to_remove to a set for more efficient search
    points_to_remove_set = set(points_to_remove)
    ids_to_remove = []  # List to store IDs that become empty
    
    # Iterate over all IDs in the structure
    for id_key in list(points_by_id.keys()):
    # Filter points that are not in points_to_remove
        points_by_id[id_key] = [pt for pt in points_by_id[id_key] 
                               if pt not in points_to_remove_set]
        if DEBUG:
            for point in points_to_remove_set:
                if point[0] == D_i and point[1] == D_j:
                    print(f"DEBUG - Removing point {point} with ID {id_key} from {type}")

    # If the ID becomes empty, mark it for removal
        if len(points_by_id[id_key]) == 0:
            ids_to_remove.append(id_key)
    
    # Remove IDs that became empty
    for id_key in ids_to_remove:
        del points_by_id[id_key]

def add_points_to_global_structure(points_to_add, target_id, type, sim_state):
    """
    Adds a list of points to the specified ID in the simulation state, avoiding duplicates.
    
    Args:
        points_to_add (list): List of points (tuples) to add
        target_id (str): ID to which the points will be added
        type (str): Type of structure ("fire", "humidity", "vegetation", "wind")
        sim_state (SimulationState): The simulation state object
    Returns:
        None
    """
    global DEBUG, D_i, D_j

    # Get the corresponding structure according to the type
    points_by_id = sim_state.get_points_by_id(type)
    
    # Add the entry if it does not exist
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

def add_vectorial_map(vectorial_map, layers_array, type, sim_state):
    """
    Adds a vectorial map to the layers array and maintains point structures by ID and type.
    
    This function:
    1. Manages four simulation state dictionaries (fire, humidity, vegetation, wind)
    2. Ensures each point exists in only one ID per type
    3. Removes duplicate points from other IDs when adding new ones
    
    Args:
        vectorialMap (dict or list): A dictionary with 'id' and 'points' keys, or a list of such dictionaries
        layersArray (list): A list of vectorial maps for the layer
        type (str): The type of layer ("fire", "humidity", "vegetation", "wind")
        sim_state (SimulationState): The simulation state object
    Returns:
        None
    """
    global D_i, D_j, DEBUG
    
    # NOT USED
    def remove_points_from_layers_array(points_to_remove, layersArray):
        """
        Auxiliary function that removes a list of points only from layersArray.
        
        Args:
            points_to_remove (list): List of points (tuples) to remove
            layersArray (list): Array of layers from which to remove the points
        Returns:
            None
        """
        # Remove points from layersArray
        layers_to_remove = []  # List of layers that become empty
        
        for layer_index, layer in enumerate(layersArray):
            # Each layer is a list of vectorial maps (dictionaries)
            if isinstance(layer, list):
                if DEBUG:
                    print("DEBUG - ERROR on Layer %d: Expected dict, got list", layer_index)
                continue            
            # If the layer is directly a dictionary with points
            elif isinstance(layer, dict) and 'points' in layer:
                # Filter points that are not in points_to_remove
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
                
                # Update the points of the element
                layer['points'] = filtered_points
                
                # If the element becomes empty, mark it for removal
                if len(filtered_points) == 0:
                    layers_to_remove.append(layer_index)
        
    # Remove empty layers (in reverse order to avoid index issues)
        for layer_index in reversed(layers_to_remove):
            layersArray.pop(layer_index)


    # Convert vectorialMap to a list if it is a single dictionary
    if isinstance(vectorial_map, dict):
        vectorialMap_copy = [vectorial_map]
    
    # Use the auxiliary function to remove all points of vectorialMap from layersArray
    for vectorial_element in vectorialMap_copy:
        if isinstance(vectorial_element, dict) and 'id' in vectorial_element and 'points' in vectorial_element:
            new_id = str(vectorial_element['id'])  # Convert ID to string for consistency
            new_points = vectorial_element['points']
            #remove_points_from_layers_array(new_points, layersArray)

            # 1. Remove these points from any ID in the global structure
            remove_points_from_global_structure(new_points, type, sim_state)
            
            # 2. Añadir los puntos al ID correspondiente
            add_points_to_global_structure(new_points, new_id, type, sim_state)

    # Add vectorialMap to layersArray if it does not already exist
    if vectorial_map not in layers_array:
        layers_array.append(vectorial_map)

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

def get_point_id_from_global_structure(point, type, sim_state):
    """
    Searches for a point in the simulation state and returns its ID if found.
    
    Args:
        point (tuple): Point (x, y) to search for
        type (str): Type of structure ("fire", "humidity", "vegetation", "wind")
        sim_state (SimulationState): The simulation state object
    Returns:
        str or None: The ID of the point if found, None if not found
    """
    points_by_id = sim_state.get_points_by_id_final(type)

    # Search for the point in the structure
    for id_key, points_list in points_by_id.items():
        for existing_point in points_list:
            if are_points_equal(point, existing_point, dist=0.5):
                return id_key
    
    # If point not found, return None
    return None

def get_point_id_from_global_structure_temp(point, type, sim_state):
    """
    Searches for a point in the temporary simulation state and returns its ID if found.
    
    Args:
        point (tuple): Point (x, y) to search for
        type (str): Type of structure ("fire", "humidity", "vegetation", "wind")
        sim_state (SimulationState): The simulation state object
    Returns:
        str or None: The ID of the point if found, None if not found
    """
    points_by_id = sim_state.get_points_by_id(type)

    # Search for the point in the structure
    for id_key, points_list in points_by_id.items():
        for existing_point in points_list:
            if are_points_equal(point, existing_point, dist=0):
                return id_key
    
    # If point not found, return None
    return None

# NOT USED
def print_points_by_id_debug(sim_state, type=None):
    """
    Debug function to print the structures of points by ID.
    
    Args:
        sim_state (SimulationState): The simulation state object
        type (str, optional): Specific type to display. If None, shows all.
    """
    
    if type is None or type == "fire":
        print(f"FIRE_POINTS_BY_ID: {sim_state.get_points_by_id('fire')}")
    if type is None or type == "humidity":
        print(f"HUMIDITY_POINTS_BY_ID: {sim_state.get_points_by_id('humidity')}")
    if type is None or type == "vegetation":
        print(f"VEGETATION_POINTS_BY_ID: {sim_state.get_points_by_id('vegetation')}")
    if type is None or type == "wind":
        print(f"WIND_POINTS_BY_ID: {sim_state.get_points_by_id('wind')}")

def print_global_structures_summary(sim_state):
    """
    Prints a summary of the simulation state structures of points by ID.
    Shows statistics about the distribution of points by type and ID.
    
    Args:
        sim_state (SimulationState): The simulation state object
    """

    print("\n" + "="*60)
    print("SUMMARY OF SIMULATION STATE STRUCTURES OF POINTS BY ID")
    print("="*60)
    
    # Statistics for FIRE
    print(f"\nFIRE POINTS BY ID:")
    total_fire_points = 0
    fire_points_by_id = sim_state.get_points_by_id("fire")
    for id_key, points in fire_points_by_id.items():
        count = len(points)
        total_fire_points += count
        print(f"  ID {id_key}: {count} points")
    print(f"  TOTAL: {total_fire_points} points")
    
    # Show points in BURNING state (ID 1)
    if '1' in fire_points_by_id and len(fire_points_by_id['1']) > 0:
        print(f"\n  POINTS IN BURNING STATE (ID 1):")
        burned_points = fire_points_by_id['1']
        if len(burned_points) <= 2000:  # If there are few points, show all
            for i, point in enumerate(burned_points):
                print(f"    {i+1:2d}. {point}")
        else:  # If there are many points, show only the first 20
            for i, point in enumerate(burned_points[:20]):
                print(f"    {i+1:2d}. {point}")
            print(f"    ... and {len(burned_points) - 20} more points")
    
    # Statistics for HUMIDITY
    print(f"\nHUMIDITY POINTS BY ID:")
    total_humidity_points = 0
    humidity_points_by_id = sim_state.get_points_by_id("humidity")
    for id_key, points in humidity_points_by_id.items():
        count = len(points)
        total_humidity_points += count
        print(f"  ID {id_key}: {count} points")
    print(f"  TOTAL: {total_humidity_points} points")
    
    # Statistics for VEGETATION
    print(f"\nVEGETATION POINTS BY ID:")
    total_vegetation_points = 0
    vegetation_points_by_id = sim_state.get_points_by_id("vegetation")
    for id_key, points in vegetation_points_by_id.items():
        count = len(points)
        total_vegetation_points += count
        print(f"  ID {id_key}: {count} points")
    print(f"  TOTAL: {total_vegetation_points} points")

    # Statistics for WIND
    print(f"\nWIND POINTS BY ID:")
    total_wind_points = 0
    wind_points_by_id = sim_state.get_points_by_id("wind")
    for id_key, points in wind_points_by_id.items():
        count = len(points)
        total_wind_points += count
        print(f"  ID {id_key}: {count} points")
    print(f"  TOTAL: {total_wind_points} points")

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
    import numpy as np
    # Eliminar duplicados preservando el orden
    seen = set()
    unique_points = []
    for pt in points:
        if pt not in seen:
            unique_points.append(pt)
            seen.add(pt)
    if len(unique_points) < 3:
        return unique_points

    # Recorrido tipo nearest neighbor probando todos los puntos como inicio
    def nearest_neighbor_best_path(pts):
        if len(pts) < 3:
            return pts
        pts = list(pts)
        best_order = None
        best_dist = float('inf')
        for start in pts:
            ordered = [start]
            used = set([start])
            total_dist = 0.0
            for _ in range(1, len(pts)):
                last = ordered[-1]
                candidates = [p for p in pts if p not in used]
                if not candidates:
                    break
                next_pt = min(candidates, key=lambda p: np.hypot(p[0]-last[0], p[1]-last[1]))
                total_dist += np.hypot(next_pt[0]-last[0], next_pt[1]-last[1])
                ordered.append(next_pt)
                used.add(next_pt)
            # Close the cycle (optional, for polygons)
            total_dist += np.hypot(ordered[0][0]-ordered[-1][0], ordered[0][1]-ordered[-1][1])
            if total_dist < best_dist:
                best_dist = total_dist
                best_order = ordered
        return best_order

    ordered_points = nearest_neighbor_best_path(unique_points)

    # Comprobar si el polígono es cóncavo (autointersección)
    try:
        from shapely.geometry import Polygon
        poly = Polygon(ordered_points)
        if not poly.is_valid or poly.exterior.is_simple is False:
            # Si el polígono es inválido o tiene autointersecciones, usar ConvexHull
            try:
                from scipy.spatial import ConvexHull, QhullError
                pts = np.array(unique_points)
                hull = ConvexHull(pts)
                hull_points = [tuple(pts[v]) for v in hull.vertices]
                return hull_points
            except Exception:
                # Si falla ConvexHull, devolver el orden nearest neighbor
                return ordered_points
        else:
            return ordered_points
    except ImportError:
        # Si no está shapely, devolver el orden nearest neighbor
        return ordered_points
    
def simplify_vectorial_map(vectorialMap, type):
    """
    Simplifies a vectorial map by merging polygons with the same ID using union operations.

    This function groups polygons by ID and merges them if they meet any of these criteria:
        1. They have points in common (intersect)
        2. One is inside the other (containment relationship)
        3. They are adjacent or overlapping
        4. (New) If any point of two polygons is within 'tolerance' distance, they are merged

    For polygons with the same ID, this creates unified geometries using the union operation.

    OPTIMIZATION: Group by ID and use unary_union from Shapely for each group, clustering by proximity.
    Parameter:
        tolerance (float): maximum distance to merge clusters (default=1.5)
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
    # Step 1: For each ID, group polygons that are close to each other (within tolerance, or intersecting, or containing one another)
    # and merge them into a single polygon to create a more compact region. This reduces redundancy and simplifies the map.
    # (Polygons in the same group will be unified if they are adjacent, overlapping, or have points within the tolerance distance.)
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
            filtered_points = remove_interior_points(all_points, type, poly_id)
            # Sort points to form a contour (angle with respect to the centroid)
            if filtered_points:
                if len(filtered_points) > 2:
                    ordered_points = sort_points(filtered_points)
                else:
                    ordered_points = filtered_points
                simplified_vectorial_map.append({'id': poly_id, 'points': ordered_points})
    return simplified_vectorial_map

# NOt USED
# THis version tries to improve the existing verion to detect if a point is interior ina  polygon.
# Experimetnl feature.
def is_interior_point(point, points_set, type, poly_id, sim_state, dist=1):
    """
    Determines if a point is interior:
    1. First, uses the classic criterion (neighbors above, below, left, right in points_set).
    2. Then, adds the check that the 4 cardinal neighbors at distance 'dist' have the same id (as previously defined).
    Args:
        point (tuple): Coordinates (x, y) of the point to evaluate.
        points_set (set): Set of points for the classic criterion.
        type (str): Layer type ('fire', 'humidity', 'vegetation', 'wind').
        poly_id: Polygon ID (optional, if known, avoids search).
        sim_state (SimulationState): The simulation state object
        dist (float): Distance for cardinal neighbors.
    Returns:
        bool: True if the point is interior, False otherwise.
    """
    # 1. Criterio clásico
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
            break
    if not (above and below and left and right):
        return False
    # 2. Comprobación de vecinos cardinales con el mismo id
    directions = [(dist, 0), (-dist, 0), (0, dist), (0, -dist)]
    for dx, dy in directions:
        neighbor = (x + dx, y + dy)
        neighbor_id = get_point_id_from_global_structure_temp(neighbor, type, sim_state)
        if neighbor_id is None or float(neighbor_id) != float(poly_id):
            return False
    return True

def remove_interior_points(points, type, poly_id):
    """
    Removes interior and duplicate points from a list of points.
    A point is considered interior if there are points above, below, left, and right of it.
    Args:
        points (list of tuple): A list of points where each point is represented as a tuple (x, y).
        type (str): Layer type ('fire', 'humidity', 'vegetation', 'wind').
        poly_id: Polygon ID for neighbor checks.
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
    # filtered_points = [point for point in unique_points if not is_interior_point(point, points_set, type, poly_id)]
    # Si el filtrado elimina todos los puntos, devolver los originales únicos
    if not filtered_points:
        return unique_points
    return filtered_points

#m:n-CAk on R functions
def evolution_function_on_R(point,fire, vegetation, humidity, wind, new_state, new_vegetation, new_humidity, new_wind, sim_state):
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
        sim_state (SimulationState): The simulation state object
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

    #we access to the wind layer that is on Z to obtain the wind speed at this point.
    #This is a point, therefore we can use the combination function on Z.
    #We must access information contained on other layers, therefore we will use combination function
    #In this case, the points we will use are georeferenced by the same coordinates, therefore the combination functions
    #is just returning the point.

    #we can access to the wind layer to obtain the wind speed at this point.
    #Now the layer contains no wind, therefore, no effect.
    #However, one can define a layer with a wind field, static, to see its effect over the fire propagation.

    #Mira si el punt te alguna component deciman
    if (isinstance(i, float) and not i.is_integer()) or (isinstance(j, float) and not j.is_integer()):
        # Aquí entra si i o j tienen parte decimal
        a = 1

    wind_speed = combination_function_on_R(nc,new_wind,"wind", sim_state)

    cell_fire = combination_function_on_R(nc,fire,"fire", sim_state) 

    #Getting the vicinity, it can be defined also over the nucleous.
    vc = get_vc_fire_R(point, max_dim, wind_speed, cell_fire)
    #debugging information
    if DEBUG:
        if i == D_i and j == D_j and cell_fire == BURNING:
            print(f"DEBUG - Fire state at point {point} is {cell_fire}")
    #End debugging information
    cell_vegetation = combination_function_on_R(nc,vegetation, "vegetation", sim_state)
    #Start Debugging information
    #if cell_vegetation == 20:
    #    print(f"DEBUG - Vegetation at point {point} is {cell_vegetation}")
    # End Debugging information
    if cell_fire == BURNING:
        if cell_vegetation <= 0:
            points = []
            points.append((i, j))
            if DEBUG:
                if(i == D_i and j == D_j):
                    print(f"DEBUG - Point {point} is burning and has no vegetation, setting to BURNED")
            add_vectorial_map({'id': BURNED, 'points': points},new_state, "fire", sim_state)
            new_LP.append([i, j])
            #new_LP.extend([elems for elems in get_vc_R(point, max_dim, wind_speed)])
            new_LP.extend([[elems[0], elems[1]] for elems in get_vc_fire_R(point, max_dim, wind_speed)])     
        else:
            points = []
            cell_vegetation -= 1
            points.append((i, j))
            add_vectorial_map({'id': cell_vegetation, 'points': points},new_vegetation, "vegetation", sim_state)
            new_LP.append([i, j])
            # new_LP.extend([elems for elems in get_vc_R(point, max_dim, wind_speed)])
    elif cell_fire == UNBURNED:
        cell_humidity =  combination_function_on_R(nc,humidity, "humidity", sim_state)
        new_cell_humidity = cell_humidity
        for point_vc in vc:
            #We must acces information contained on other layers, therefore we will use combination funcion
            #In this case, the points we will use are georeferenciated by the same coordinates, therefore the combination functions
            #is just returning the point.
            vicinity_cell_state = combination_function_on_R(point_vc,fire, "fire", sim_state)
            if vicinity_cell_state  == BURNING:
                if cell_humidity > 0:
                    new_cell_humidity -= 1
                    points = []
                    points.append((i, j))
                    add_vectorial_map({'id': max(new_cell_humidity,0), 'points': points},new_humidity, "humidity", sim_state)
                    new_LP.append([i, j])
                    # No añadir vicinity cuando solo se reduce humedad
                elif cell_vegetation > 0:
                    points = []
                    points.append((i, j))
                    add_vectorial_map({'id': BURNING, 'points': points},new_state, "fire", sim_state)
                    new_LP.append([i, j])
                    #new_LP.extend([elems for elems in get_vc_R(point, max_dim, wind_speed)])
                    new_LP.extend([[elems[0], elems[1]] for elems in get_vc_fire_R(point, max_dim, wind_speed, cell_fire)])
              
    if wind_as_main_layer:
        # Evolution function for wind layer
        cell_wind = combination_function_on_R(nc, wind, "wind", sim_state)
        # Getting the vicinity.
        vc = get_vc_wind(point, max_dim)
        # cell_fire = combination_function_on_Z(nc,fire)

        if cell_wind == CALM:
            for point_vc in vc:
                vicinity_cell_wind = combination_function_on_R(point_vc,wind,"wind", sim_state)
                if vicinity_cell_wind == WIND:
                    points = []
                    points.append((i, j))
                    add_vectorial_map({'id': WIND, 'points': points},new_wind, "wind", sim_state)
                    new_LP.append([i, j])
                    new_LP.extend([[elems[0], elems[1]] for elems in get_vc_wind(point, max_dim)])
                    break
        else: #wind is not CALM
            if cell_fire == BURNING:
                    add_vectorial_map({'id': cell_wind+1, 'points': points},new_wind, "wind", sim_state)

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

    global STEEPS, plot_polygons
    
    folder_path = './'
    size = (100, 100)

    # Reading vegetation and humidity layers
    fileFire = 'fire.vec'
    polygonsFire = read_idrisi_vector_file(fileFire)
    
    fileVegetation = 'vegetation.vec'
    polygonsVegetation = read_idrisi_vector_file(fileVegetation)

    fileHumidity = 'humidity.vec'
    polygonsHumidity = read_idrisi_vector_file(fileHumidity)

    fileWind = 'wind.vec'
    polygonsWind = read_idrisi_vector_file(fileWind)
    
    # Create simulation state object to replace global structures
    sim_state = SimulationState()
    
    # Initialize simulation state with initial polygon data
    '''
    for polygon in polygonsFire:
        polygon_id = str(polygon['id'])
        points = polygon['points']
        if polygon_id not in sim_state.fire_points_by_id:
            sim_state.fire_points_by_id[polygon_id] = []
        sim_state.fire_points_by_id[polygon_id].extend(points)
    
    for polygon in polygonsVegetation:
        polygon_id = str(polygon['id'])
        points = polygon['points']
        if polygon_id not in sim_state.vegetation_points_by_id:
            sim_state.vegetation_points_by_id[polygon_id] = []
        sim_state.vegetation_points_by_id[polygon_id].extend(points)
    
    for polygon in polygonsHumidity:
        polygon_id = str(polygon['id'])
        points = polygon['points']
        if polygon_id not in sim_state.humidity_points_by_id:
            sim_state.humidity_points_by_id[polygon_id] = []
        sim_state.humidity_points_by_id[polygon_id].extend(points)
    
    for polygon in polygonsWind:
        polygon_id = str(polygon['id'])
        points = polygon['points']
        if polygon_id not in sim_state.wind_points_by_id:
            sim_state.wind_points_by_id[polygon_id] = []
        sim_state.wind_points_by_id[polygon_id].extend(points)
        '''
    
    # Create raster versions for compatibility
    create_idrisi_raster(polygonsFire, 'fire')
    create_idrisi_raster(polygonsVegetation, 'vegetation')
    create_idrisi_raster(polygonsHumidity, 'humidity')
    create_idrisi_raster(polygonsWind, 'wind')

    # Wind layer (raster version)
    # Note: The wind layer is currently not used in this vector-based simulation. If you wish to use a raster file for wind,
    # simply load the appropriate file here. However, if switching to a raster wind layer, you must also update the vicinity
    # and nucleus functions to handle the discrete grid case (Z) instead of vector polygons (R).
    wind_doc_path = os.path.join(folder_path, 'wind.doc')
    wind_img_path = os.path.join(folder_path, 'wind.img')
    # If you want to disable wind for validation purposes when working with raster files, uncomment the following line.
    wind_img_path = os.path.join(folder_path, 'no_wind.img')

    # In the example presented in the paper, wind is modeled as a vectorial (continuous) layer. 
    # This is where the wind data is loaded for use in the simulation.
    wind_data = read_idrisi_raster_file(wind_img_path)

    # Reading the wildfire starting point
    max_dim = [100, 100]
    LP = []

    for polygon in polygonsFire:
        polygon_id = polygon['id']
        points = polygon['points']
        if polygon_id == BURNING:
            for (x, y) in points:
                ini_point = (x, y)
                LP.append(ini_point)
                LP.extend([point for point in get_vc_fire_R(ini_point, max_dim)])

    if wind_as_main_layer:
        # Reading the wind starting point

        for polygon in polygonsWind:
            polygon_id = polygon['id']
            points = polygon['points']
            if polygon_id == WIND:
                for (x, y) in points:
                    ini_point = (x, y)
                    if ini_point not in LP:
                        LP.append(ini_point)
                        LP.extend([point for point in get_vc_wind(ini_point, max_dim)])

        '''
        initial_point_wind = [50,50]
        i_w, j_w = initial_point_wind
        wind_data[i_w, j_w]=WIND
        # Adding the initial point we change.
        LP.append(initial_point_wind)
        
        #Getting the vicinity for wind main layer
        vc = get_vc_wind_Z(initial_point_wind, max_dim)
        # Also adding the neighborhoods of this point, avoid duplicates
        for point in vc:
            if point not in LP:
                LP.append(point)
        '''

    fire_evolution = [polygonsFire]
    vegetation_evolution = [polygonsVegetation]
    humidity_evolution = [polygonsHumidity]
    # wind_evolution = [wind_data]
    wind_evolution = [polygonsWind]

    n_steps = STEEPS

    for _ in range(n_steps):
        LP_rep = []
        LP_new = []
        new_state = fire_evolution[-1].copy()
        new_vegetation = vegetation_evolution[-1].copy()
        new_humidity = humidity_evolution[-1].copy()
        new_wind = wind_evolution[-1].copy()

        for point in LP:
            LP_new, new_state, new_vegetation, new_humidity, new_wind = evolution_function_on_R(point, fire_evolution[0], vegetation_evolution[0], humidity_evolution[0], wind_evolution[0], new_state, new_vegetation, new_humidity, new_wind, sim_state)
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]

        if plot_polygons:
            fire_evolution.append(simplify_vectorial_map(new_state, "fire"))
            vegetation_evolution.append(simplify_vectorial_map(new_vegetation, "vegetation"))
            humidity_evolution.append(simplify_vectorial_map(new_humidity, "humidity"))
            wind_evolution.append(simplify_vectorial_map(new_wind, "wind"))
        else:
            fire_evolution.append(new_state)
            vegetation_evolution.append(new_vegetation)
            humidity_evolution.append(new_humidity)
            wind_evolution.append(new_wind)


        # Finalize the step by copying working structures to final
        sim_state.finalize_step()

        print(f"Step {_+1}/{n_steps} completed. Fire evolution size: {len(fire_evolution[-1])}, Vegetation size: {len(vegetation_evolution[-1])}, Humidity size: {len(humidity_evolution[-1])}, Wind size: {len(wind_evolution[-1])}")

    return fire_evolution, vegetation_evolution, humidity_evolution, wind_evolution

##############################################################
#m:n-CAk main functions
##############################################################

def get_vc_fire_Z(point, max_dim, wind_speed=0, cell_state=BURNING):
    """
    Vicinity function. Get the valid coordinates (Von Neumann neighbourhood) adjacent to a given point within a specified maximum dimension.
    The same for Z^2 and R^2 in this example.
    Args:
        point (tuple): A tuple (i, j) representing the coordinates of the point.
        max_dim (tuple): A tuple (max_i, max_j) representing the maximum dimensions of the grid.
    Returns:
        list: A list of tuples representing the valid adjacent coordinates.
    """
    global WIND_MULT
    vc= []
    i, j = point
    max_i, max_j = max_dim

    # Von Neumann neighborhood
    if i > 0: vc.append((i-1,j))
    if i < max_i - 1: vc.append((i+1, j))
    if j > 0:  vc.append((i, j-1))
    if j < max_j - 1: vc.append((i, j+1))

    #consider the case of wind
    wind_speed = wind_speed*WIND_MULT  # Assuming wind_speed is a multiplier for the number of steps to consider
    # Add the wind effect by extending the vicinity in the direction of the wind
    if cell_state==BURNING:
        j = j+1
        while wind_speed > 0: 
            j += 1
            if 0 <= j < max_j: vc.append((i, j))
            wind_speed -= 1
    else:
        j = j-1
        while wind_speed > 0: 
            j -= 1
            if 0 <= j < max_j: vc.append((i, j))
            wind_speed -= 1

    return vc

def get_vc_wind(point, max_dim):
    """
    Vicinity function. Get the valid coordinates (Moore neighbourhood, distance 2) adjacent to a given point within a specified maximum dimension.
    Args:
        point (tuple): A tuple (i, j) representing the coordinates of the point.
        max_dim (tuple): A tuple (max_i, max_j) representing the maximum dimensions of the grid.
        wind_speed (float): The speed of the wind affecting the vicinity.
        cell_state (int): The state of the cell (e.g., BURNING).
    Returns:
        list: A list of tuples representing the valid adjacent coordinates (Moore, dist=2).
    """
    # Parámetro local para la distancia Moore
    moore_distance = 3
    vc = []
    i, j = point
    max_i, max_j = max_dim
    # Moore neighborhood, configurable distance
    for di in range(-moore_distance, moore_distance+1):
        for dj in range(-moore_distance, moore_distance+1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            # Solo añadir si ambos índices están dentro de rango y son no negativos
            if (0 <= ni < max_i) and (0 <= nj < max_j):
                vc.append((ni, nj))
    # Filtrar por seguridad cualquier vecino con índices negativos o fuera de rango
    vc = [(ni, nj) for (ni, nj) in vc if 0 <= ni < max_i and 0 <= nj < max_j]
    return vc


def get_vc_fire_R(point, max_dim, wind_speed=0, cell_state=BURNING):
    """
    Vicinity function. Get the valid coordinates (Von Neumann neighbourhood) adjacent to a given point within a specified maximum dimension.
    The same for Z^2 and R^2 in this example.
    Args:
        point (tuple): A tuple (i, j) representing the coordinates of the point.
        max_dim (tuple): A tuple (max_i, max_j) representing the maximum dimensions of the grid.
    Returns:
        list: A list of tuples representing the valid adjacent coordinates.
    """
    global WIND_MULT
    vc= []
    i, j = point
    max_i, max_j = max_dim

    # Von Neumann neighborhood
    if i > 0: vc.append((i-1,j))
    if i < max_i - 1: vc.append((i+1, j))
    if j > 0:  vc.append((i, j-1))
    if j < max_j - 1: vc.append((i, j+1))

    #consider the case of wind
    wind_speed = wind_speed*WIND_MULT  # Assuming wind_speed is a multiplier for the number of steps to consider
    # Add the wind effect by extending the vicinity in the direction of the wind
    # we extend j because on numpy the first coordinate is the y coordinate and the second is the x coordinate.
    if cell_state==BURNING:
        if wind_speed>0:
            i=i+wind_speed
            if i < max_i - 1: vc.append((i, j))
    else:
        if wind_speed>0:
            i=i-wind_speed
            if i < max_i - 1: vc.append((i, j))
    return vc

def get_nc(point):
    """
    Nucleus function. Returns the input point without any modifications as a nucleus.
    The same for Z^2 and R^2 in this example.
    
    Args:
        point (any): The input point to be returned.

    Returns:
        any: The same input point as nucleus.
    """
    return point

def set_nc(point, layer, value):
    """
    Nucleus function. Sets the value for the nucleus.
    The same for Z^2 and R^2 in this example.

    Args:
        point (any): The input point to be returned.

    Returns:
        any: The same input point as nucleus.
    """
    x, y = point
    layer[x, y] = value
    return point

def combination_function_on_R(point, layer, type=None, sim_state=None):
    """
    Retrieves the value from the specified layer at the given point coordinates.
    All the layers in this example share the same coordinates, so the E function is the identity function.
    To simplify the example implementation we define just a single combination_function that uses as a parameter the layer.
    Determines the polygon ID for a given point within a specified layer.
    
    Args:
        point (tuple): A tuple representing the coordinates of the point (e.g., (x, y)).
        layer (object): The layer object containing polygon data.
        type (str): The type of layer ("fire", "humidity", "vegetation", "wind").
        sim_state (SimulationState): The simulation state object.
    Returns:
        int: The ID of the polygon that contains the point.
    """
    return int(find_polygon_id(point, layer, type, sim_state))

def combination_function_on_Z(point, layer):
    """
    Retrieves the value from the specified layer at the given point coordinates.
    All the layers in this example share the same coordinates, so the E function is the identity function.
    To simplify the example implementation we define just a single combination_function that uses as a parameter the layer.

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
        # Return the maximum value of the layer if out of bounds
        return 0

    return layer[i,j]
    # return point

# NOT USED
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
# Report functions
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
# Main
##############################################################

if __name__ == "__main__":

    domain = 'Z'

    domain = input("Please, select the domain to work on (Z or R): ")
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

        print_burning_cells_report(windEvolution, domain)
        
        # Print simulation state summary (commented out - would need sim_state object)
        # print_global_structures_summary(sim_state)
        
        # Create verification map for debugging
        create_verification_map(fireEvolution, domain)


        results_window(domain, fireEvolution, vegetationEvolution, humidityEvolution, windEvolution)

# =============================================================
# NOT USED
# Generator of vectorial map from FIRE_POINTS_BY_ID_FINAL
# =============================================================
def generate_vectorial_map_from_fire_points_by_id_final(sim_state, dist=1):
    """
    Generates a vectorial map (list of dicts with 'id' and 'points') from fire points by id final.
    For each ID, takes the points and generates the outermost polygon that encompasses those points,
    ensuring that consecutive points are at distance <= dist to avoid encompassing non-connected concave parts.
    Args:
        sim_state (SimulationState): The simulation state object
        dist (float): maximum distance between consecutive points on the contour (default=1)
    Returns:
        vectorial_map (list): list of dicts {'id': id, 'points': [p1, p2, ...]}
    """
    import numpy as np
    fire_points_by_id_final = sim_state.get_points_by_id_final("fire")
    vectorial_map = []
    for poly_id, points in fire_points_by_id_final.items():
        # Remove duplicates
        unique_points = []
        seen = set()
        for pt in points:
            if pt not in seen:
                unique_points.append(pt)
                seen.add(pt)
        if len(unique_points) < 3:
            if unique_points:
                vectorial_map.append({'id': poly_id, 'points': unique_points})
            continue
        # Remove interior points
        points_set = set(unique_points)
        filtered_points = [pt for pt in unique_points if not is_interior_point(pt, points_set, 'state', poly_id, sim_state, dist=dist)]
        if not filtered_points:
            filtered_points = unique_points
        # Sort points to form the external contour, ensuring connectivity
        def nearest_neighbor_chain(pts, max_dist=dist):
            pts = list(pts)
            if len(pts) < 3:
                return pts
            used = set()
            chain = [pts[0]]
            used.add(pts[0])
            for _ in range(1, len(pts)):
                last = chain[-1]
                # Find the closest unused point at distance <= max_dist
                candidates = [(p, np.hypot(p[0]-last[0], p[1]-last[1])) for p in pts if p not in used]
                candidates = [c for c in candidates if c[1] <= max_dist+1e-8]
                if not candidates:
                    # If there are no candidates, close the cycle with the closest one
                    left = [p for p in pts if p not in used]
                    if left:
                        next_pt = min(left, key=lambda p: np.hypot(p[0]-last[0], p[1]-last[1]))
                        chain.append(next_pt)
                        used.add(next_pt)
                    else:
                        break
                else:
                    next_pt = min(candidates, key=lambda c: c[1])[0]
                    chain.append(next_pt)
                    used.add(next_pt)
            return chain
        ordered_points = nearest_neighbor_chain(filtered_points, max_dist=dist)
        if len(ordered_points) < 3:
            vectorial_map.append({'id': poly_id, 'points': ordered_points})
        else:
            vectorial_map.append({'id': poly_id, 'points': ordered_points})
    return vectorial_map

