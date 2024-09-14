##############################################################
# Wildfire_on_T.py
# Created on: 2021-06-30
# Author: Pau Fonseca i Casas
# Copyright: Pau Fonseca i Casas
# Description: This script simulates the spread of a wildfire using a m:n-CAk cellular automaton model over R^2.
##############################################################

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

def read_idrisi_vector_file(file_path):
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

def reshape(data, shape):



    """



    Convierteix el vector columna en una matriu



    """


    return data.reshape(shape)

def update(frame):
    im.set_array(np.flipud(fireEvolution[frame]))
    ax.set_title(f'Step {frame}')
    return im, ax

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

def animateLayers(layersArray, interval=500, radius=1, color='green', title='No title'):
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
        plotVectorial(ax, layer, i, radius=radius, color=color, title=title)
        return []

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(layersArray), interval=interval, blit=False, repeat=True)
    plt.show()

def resultsWindow():
    root = tk.Tk()
    root.title("Select Action")

    button_frame = ttk.Frame(root)
    button_frame.pack(pady=20)

    button1 = ttk.Button(button_frame, text="Vegetation", command=lambda: animateLayers(vegetationEvolution, title='Vegetation'))
    button1.pack(side=tk.LEFT, padx=10)

    button2 = ttk.Button(button_frame, text="Humidity", command=lambda: animateLayers(humidityEvolution, title='Humidity'))
    button2.pack(side=tk.LEFT, padx=10)

    button3 = ttk.Button(button_frame, text="Fire", command=lambda: animateLayers(fireEvolution, title='Fire'))
    button3.pack(side=tk.LEFT, padx=10)

    root.mainloop()

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
            if len(polygon['points']) == 1:
                # Si el polígono es un solo punto, el área es 0
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
                simplified_vectorial_map.append({'id': poly_id, 'points': points_ext})

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

def create_idrisi_raster(polygons, output_filename):
    """
    Crea un archivo raster en formato IDRISI con dimensiones de 100x100 puntos,
    utilizando la función find_polygon_id para determinar los IDs de los polígonos
    que contienen cada punto.

    Args:
        polygons (list): Una lista de diccionarios, donde cada diccionario representa un polígono con las claves:
            - 'id' (any): El identificador del polígono.
            - 'points' (list): Una lista de tuplas que representan los vértices del polígono.
        output_filename (str): El nombre base del archivo de salida (sin extensión).
    """
    # Dimensiones del raster
    width, height = 100, 100

    # Crear una matriz de 100x100 puntos
    raster = np.zeros((height, width), dtype=int)

    # Iterar sobre cada punto en la matriz
    for i in range(height):
        for j in range(width):
            point = (i, j)
            polygon_id = find_polygon_id(point, polygons)
            if polygon_id is not None:
                raster[i, j] = polygon_id

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

##############################################################
#m:n-CAk main functions
##############################################################
#Vicinity function in R^2 Moore neighbourhood.
def get_vc(point, max_dim):
    vc= []
    i, j = point
    max_i, max_j = max_dim

    if i > 0: vc.append((i-1,j))
    if i < max_i - 1: vc.append((i+1, j))
    if j > 0:  vc.append((i, j-1))
    if j < max_j - 1: vc.append((i, j+1))

    return vc

#nucleous function, in this case returns the same point.

def get_nc(point):
    return point

#combination function
def combination_function(point):
    return point

#Evolution function, propagation of the wildfire.

def Evolution_function3(point,fire, vegetation, humidity, new_state, new_vegetation, new_humidity):
    new_LP = []
    nc = []
    vc = []
    max_dim = [100,100]

    #Getting the nucleous.
    nc = get_nc(point)
    i, j = point

    if i == 70 and j == 70:
        a=1
    #Getting the vicinity
    vc = get_vc(point, max_dim)

    cell_state = find_polygon_id(point,fire) 
    if cell_state == BURNING:  # Si la cel·la està burning
        value_veg = find_polygon_id(point,vegetation)
        if value_veg == 0:
            #new_state[i, j] = BURNED
            points = []
            points.append((i, j))
            #new_state.append({'id': BURNED, 'points': points})
            addVectorialMap({'id': BURNED, 'points': points},new_state)
        else:
            points = []
            value_veg -= 1
            points.append((i, j))
            #new_vegetation.append({'id': value_veg, 'points': points})
            addVectorialMap({'id': value_veg, 'points': points},new_vegetation)
            #The point must be considered on the nex iterations.
            new_LP.append((i,j))
            new_LP.extend([elems for elems in get_vc(point, max_dim)])
    elif cell_state == UNBURNED:
        for point_vc in vc:
            #We muist acces information contained on other layers, therefore we will use combination funcion
            #In this case, the points we will use are georeferences by the same coordinates, therefore the combination functions
            #is just returning the point.
            i_vc, j_vc = combination_function(point_vc) #ATENCIO cal retornar un punt, no  puc assegurar en el cas general que el punt es el mateix!!!!!!!!!!
            cell_state = find_polygon_id(point_vc,fire)
            cell_humidity =  find_polygon_id(point,humidity)
            cell_vegetation =  find_polygon_id(point,vegetation) 
            if cell_state  == BURNING:
                if cell_humidity > 0:
                    cell_humidity -= 1
                    points = []
                    points.append((i, j))
                    #new_humidity.append({'id': cell_humidity, 'points': points})
                    addVectorialMap({'id': cell_humidity, 'points': points},new_humidity)
                    #new_humidity[i, j] -= 1
                elif cell_vegetation > 0:
                    points = []
                    points.append((i, j))
                    #new_state.append({'id': BURNING, 'points': points})
                    addVectorialMap({'id': BURNING, 'points': points},new_state)
                    #new_state[i, j] = BURNING
                    new_LP.append((i_vc, j_vc))
                    new_LP.extend([elems for elems in get_vc(point_vc, max_dim)])

    return new_LP,new_state, new_vegetation, new_humidity

if __name__ == "__main__":
    folder_path = './'
    # Auxiliary functions to obtain the information of the layers from files (IDRISI 32 format).
    #defining the size for the layers, the same for all to simplify
    size= (100,100)

    #Reading vegetation and humidity layers.
    fileVegetation = 'vegetation.vec'
    polygonsVegetation = read_idrisi_vector_file(fileVegetation)
    fileHumidity = 'humidity.vec'
    polygonsHumidity = read_idrisi_vector_file(fileHumidity)

    create_idrisi_raster(polygonsVegetation,'vegetation_map2')
    create_idrisi_raster(polygonsHumidity,'humidity_map2')

    # Definition of the function E, the states for the layer fire. For layers vegetation and humidity
    # the interpretation is the identity function.
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    
    # Reading the wildfire starting point
    fileFire = 'fire.vec'
    polygonsFire = read_idrisi_vector_file(fileFire)

    #initial_fire = np.zeros(size)
    max_dim = [100,100]
    LP = []
    
    for polygon in polygonsFire:
        polygon_id = polygon['id']
        points = polygon['points']
        for (x, y) in points:
            ini_point = (x, y)
            #Adding the initial point we change.
            LP.append(ini_point)
            #Also adding the neighbourhoods of this point.
            LP.extend([point for point in get_vc(ini_point, max_dim)])
    
    # Variable that will contain all the states we define on the execution of the model.
    fireEvolution = [polygonsFire]
    vegetationEvolution = [polygonsVegetation]
    humidityEvolution = [polygonsHumidity]

    # Number of steeps to execute the evolution function
    n_steps = 100

    for _ in range(n_steps):
        LP_rep = []
        #LP_rep, new_state, new_vegetation, new_humidity = Evolution_function2(LP,fireEvolution[-1], vegetationEvolution[-1], humidityEvolution[-1])
        LP_new = []
        new_state = fireEvolution[-1].copy()
        new_vegetation=vegetationEvolution[-1].copy()
        new_humidity=humidityEvolution[-1].copy()

        for point in LP:
            LP_new, new_state, new_vegetation, new_humidity = Evolution_function3(point,fireEvolution[-1], vegetationEvolution[-1], humidityEvolution[-1], new_state, new_vegetation, new_humidity)
            [LP_rep.append(elemento) for elemento in LP_new if elemento not in LP_rep]

        LP = []
        [LP.append(elemento) for elemento in LP_rep if elemento not in LP]

        new_state = simplifyVectorialMap(new_state)
        fireEvolution.append(new_state)
        vegetationEvolution.append(new_vegetation)
        humidityEvolution.append(new_humidity)


    resultsWindow()
