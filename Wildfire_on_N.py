import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import os

import numpy as np

import matplotlib.pyplot as plt

##############################################################

#Auxiliary functions to obtain data and to represent the data
##############################################################


def read_doc_file(doc_path):



    metadata = {}



    with open(doc_path, 'r') as doc_file:



        for line in doc_file:



            key, value = line.strip().split(': ')



            metadata[key.strip()] = value.strip()
    return metadata


def read_img_file(img_path):



    data = np.loadtxt(img_path).astype(int)
    return data


def reshape(data, shape):



    """



    Convierteix el vector columna en una matriu



    """



    return data.reshape(shape)


def update(frame):

    im.set_array(np.flipud(fireEvolution[frame]))

    ax.set_title(f'Step {frame}')

    return im, ax

##############################################################

#m:n-CAk main functions
##############################################################


#Evolution function, propagation of the wildfire.

def Evolution_function(state, vegetation, humidity):



    new_state = state.copy()



    #aixo va fora de la funció d'evolució.


    for i in range(state.shape[0]):


        for j in range(state.shape[1]):



            #3Això FORMA PART de la funció d'evolució


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


#Vicinity function in N^2 Moore neighbourhood.

def get_vc(point, max_dim):

    vc= []

    i, j = point

    max_i, max_j = max_dim


    if i > 0: vc.append([i-1,j])

    if i < max_i - 1: vc.append([i+1, j])

    if j > 0:  vc.append([i, j-1])

    if j < max_j - 1: vc.append([i, j+1])

    return vc


#nucleous function, in this case returns the same point.

def get_nc(point):
    return point


#combination function

def combination_function(point):
    return point


#Evolution function, propagation of the wildfire.


def Evolution_function2(LP,state, vegetation, humidity):


    new_state = state.copy()

    new_vegetation=vegetation.copy()

    new_humidity=humidity.copy()

    new_LP = []

    nc = []

    vc = []

    max_dim = [100,100]


    #aixo hauria d'anar fora de la funció d'evolució.


    for point in LP:


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


def Evolution_function3(point,state, vegetation, humidity, new_state, new_vegetation, new_humidity):
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


if __name__ == "__main__":

    folder_path = './'
    # Auxiliary functions to obtain the information of the layers from files (IDRISI 32 format).
    #vegetation layer.
    vegetation_map_doc_path = os.path.join(folder_path, 'vegetation_map.doc')
    vegetation_map_img_path = os.path.join(folder_path, 'vegetation_map.img')

    #humidity layer
    humidity_doc_path = os.path.join(folder_path, 'humidity_map.doc')
    humidity_img_path = os.path.join(folder_path, 'humidity_map.img')


    # Reading the information of the layers
    #vegetatrion layer
    vegetation_data = read_img_file(vegetation_map_img_path)

    #humidity layer
    humidity_data = read_img_file(humidity_img_path)


    #defining the size for the layers, the same for all.
    size= (100,100)

    # Auxiliary functions to convert the vector in a matrix of data for both layers.
    humidity_data = reshape(humidity_data, size)
    vegetation_data = reshape(vegetation_data, size)

    # Definition of the function E, the interpretation for the layer state, the main layer.
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    
    # Modofying the initial conditons to start the wildfire
    initial_fire = np.zeros(size)
    ini_point = [70, 70]
    max_dim = [100,100]
    i, j = ini_point
    initial_fire[i, j] = BURNING
    LP = []


    #Adding the initial point we change.
    LP.append(ini_point)

    #Also adding the neighbourhoods of this point.
    LP.extend([point for point in get_vc(ini_point, max_dim)])
    

    # Variable that will contain all the states we define on the execution of the model.
    fireEvolution = [initial_fire]
    vegetationEvolution = [vegetation_data]
    humidityEvolution = [humidity_data]

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
        fireEvolution.append(new_state)
        vegetationEvolution.append(new_vegetation)
        humidityEvolution.append(new_humidity)

    fig, ax = plt.subplots()
    cmap = plt.cm.colors.ListedColormap(['green', 'red', 'black'])  
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(np.flipud(fireEvolution[0]), cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
    #im = ax.imshow(np.flipud(vegetationEvolution[0]), cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
    ani = FuncAnimation(fig, update, frames=n_steps, interval=100, repeat=True)
    plt.show(block=True)