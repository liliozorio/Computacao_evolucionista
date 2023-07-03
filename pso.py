import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer
import numpy as np
from itertools import combinations
import random


def plot_polygon_with_inner_polygons(outer_coords, inner_coords_list, archive):
    fig, ax = plt.subplots()
    
    outer_polygon = patches.Polygon(outer_coords, linewidth=1, edgecolor='black', facecolor='gray')
    ax.add_patch(outer_polygon)
    
    for inner_coords in inner_coords_list:
        inner_polygon = patches.Polygon(inner_coords, linewidth=1, edgecolor='black', facecolor='yellow')
        ax.add_patch(inner_polygon)
    
    x_coords = [coord[0] for coord in outer_coords]
    y_coords = [coord[1] for coord in outer_coords]
    ax.set_xlim(min(x_coords) - 1, max(x_coords) + 1)
    ax.set_ylim(min(y_coords) - 1, max(y_coords) + 1)
    
    ax.set_aspect('equal', 'box')
    
    plt.savefig(archive)

def max_distance(points):
    point = np.array(points, float)

    current_max = 0
    for a, b in combinations(np.array(points), 2):
        current_distance = np.linalg.norm(a-b)
        if current_distance > current_max:
            current_max = current_distance
    return current_max

def f1(arrs, z):
    polyGrande = Polygon([(2.5,4.5),
                    (5.0,4.5),
                    (6.5,2.5),
                    (6.5, 0.5),
                    (6,0),
                    (3.5,0),
                    (1.5, 2.0),
                    (1.5, 3.5)
                    ])
    r_f = []

    polyMapping = mapping(polyGrande)
    max_distance_var = max_distance(polyMapping['coordinates'][0])

    for arr in arrs:
        i=0
        soma_g1 = 0
        while(i<len(arr)):

            polya = Polygon([(arr[i]+0.5,arr[i+1]+0.5),
                            (arr[i]+0.5, arr[i+1]-0.5),
                            (arr[i]-0.5, arr[i+1]-0.5),
                            (arr[i]-0.5, arr[i+1]+0.5)])
            if(not polyGrande.contains(polya)):
                distance = polyGrande.distance(polya)
                soma_g1 = soma_g1 + max_distance_var*z + distance
            j = 0

            soma_g2 = 0
            while(j<len(arr)):
                if(i!=j):
                    polyb = Polygon([(arr[j]+0.5,arr[j+1]+0.5),
                            (arr[j]+0.5, arr[j+1]-0.5),
                            (arr[j]-0.5, arr[j+1]-0.5),
                            (arr[j]-0.5, arr[j+1]+0.5)])
                    distance = polyb.distance(polya)
                    soma_g2 = soma_g2 + distance
                    if(polya.intersects(polyb)):
                        soma_g2 = soma_g2 + max_distance_var*z
                j=j+2

            soma_g1 = soma_g1 + soma_g2

            i=i+2
        r_f.append(soma_g1)
    return r_f

def f2(arrs, z):
    polyGrande = Polygon([(0,3),
                    (1,4.5),
                    (5,4.5),
                    (6, 3),
                    (5,0),
                    (3, 1.5),
                    (1,0)
                    ])
    r_f = []

    polyMapping = mapping(polyGrande)
    max_distance_var = max_distance(polyMapping['coordinates'][0])

    for arr in arrs:
        i=0
        soma_g1 = 0
        while(i<len(arr)):

            polya = Polygon([(arr[i]+0.5,arr[i+1]+0.5),
                            (arr[i]+0.5, arr[i+1]-0.5),
                            (arr[i]-0.5, arr[i+1]-0.5),
                            (arr[i]-0.5, arr[i+1]+0.5)])
            if(not polyGrande.contains(polya)):
                distance = polyGrande.distance(polya)
                soma_g1 = soma_g1 + max_distance_var*z + distance
            j = 0

            soma_g2 = 0
            while(j<len(arr)):
                if(i!=j):
                    polyb = Polygon([(arr[j]+0.5,arr[j+1]+0.5),
                            (arr[j]+0.5, arr[j+1]-0.5),
                            (arr[j]-0.5, arr[j+1]-0.5),
                            (arr[j]-0.5, arr[j+1]+0.5)])
                    distance = polyb.distance(polya)
                    soma_g2 = soma_g2 + distance
                    if(polya.intersects(polyb)):
                        soma_g2 = soma_g2 + max_distance_var*z
                j=j+2

            soma_g1 = soma_g1 + soma_g2

            i=i+2
        r_f.append(soma_g1)
    return r_f

def confere(arr, polyGrande):
    i=0
    while(i<len(arr)):
        polya = Polygon([(arr[i]+0.5,arr[i+1]+0.5),
                        (arr[i]+0.5, arr[i+1]-0.5),
                        (arr[i]-0.5, arr[i+1]-0.5),
                        (arr[i]-0.5, arr[i+1]+0.5)])
        if(not polyGrande.contains(polya)):
            return False
        j = 0
        while(j<len(arr)):
            if(i!=j):
                polyb = Polygon([(arr[j]+0.5,arr[j+1]+0.5),
                        (arr[j]+0.5, arr[j+1]-0.5),
                        (arr[j]-0.5, arr[j+1]-0.5),
                        (arr[j]-0.5, arr[j+1]+0.5)])
                if(polya.intersects(polyb)):
                    return False
            j=j+2
        i=i+2
    return True


outer_polygon_coords1 = [(2.5,4.5),(5.0,4.5),(6.5,2.5),(6.5, 0.5),(6,0),(3.5,0),(1.5, 2.0),(1.5, 3.5)]
outer_polygon_coords2 = [(0,3),(1,4.5),(5,4.5),(6, 3),(5,0),(3, 1.5),(1,0)]

polygon1 = Polygon([(2.5,4.5),
                    (5.0,4.5),
                    (6.5,2.5),
                    (6.5, 0.5),
                    (6,0),
                    (3.5,0),
                    (1.5, 2.0),
                    (1.5, 3.5)
                    ])

polygon2 = Polygon([(0,3),
                    (1,4.5),
                    (5,4.5),
                    (6, 3),
                    (5,0),
                    (3, 1.5),
                    (1,0)
                    ])

z = 0
answare = []
optimizer_before = ''
best_pos_before = ''
while(confere(answare, polygon2)):
    z = z + 1
    # Call instance of PSO
    if z>1:
        options = {'c1': 1.5 , 'c2': 1.5,'w':0.5}
        optimizer_before = optimizer
        best_pos_before = answare
        initial = []
        for p in all_particles:
            random_number_x = random.uniform(1.5,6.5)
            random_number_z = random.uniform(0,4.5)
            p = np.append(p, random_number_x)
            p = np.append(p, random_number_z)
            initial.append(p)
        initial = np.array(initial)

        optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=z*2, options=options, init_pos=initial)
    else:
        options = {'c1': 1.5 , 'c2': 1.5,'w':0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=z*2, options=options)

    best_cost, best_pos = optimizer.optimize(f2, iters=2000, z=z)
    answare = best_pos
    all_particles = optimizer.swarm.position

z = z-1
print(z)

plot_cost_history(optimizer_before.cost_history)
plt.savefig('history_cost.png')

plot_cost_history(optimizer.cost_history)
plt.savefig('history_cost_fail.png')

best_pos_array = np.array(best_pos_before)

inner_polygons = []

i = 0
while(i<len(best_pos_array)):
    polya = [(best_pos_array[i]+0.5,best_pos_array[i+1]+0.5),
                (best_pos_array[i]+0.5, best_pos_array[i+1]-0.5),
                (best_pos_array[i]-0.5, best_pos_array[i+1]-0.5),
                (best_pos_array[i]-0.5, best_pos_array[i+1]+0.5)]
    inner_polygons.append(polya)
    i=i+2

plot_polygon_with_inner_polygons(outer_polygon_coords1, inner_polygons, 'resultado.png')

best_pos_array = np.array(best_pos)

inner_polygons = []

i = 0
while(i<len(best_pos_array)):
    polya = [(best_pos_array[i]+0.5,best_pos_array[i+1]+0.5),
                (best_pos_array[i]+0.5, best_pos_array[i+1]-0.5),
                (best_pos_array[i]-0.5, best_pos_array[i+1]-0.5),
                (best_pos_array[i]-0.5, best_pos_array[i+1]+0.5)]
    inner_polygons.append(polya)
    i=i+2

plot_polygon_with_inner_polygons(outer_polygon_coords1, inner_polygons, 'resultadofalho.png')