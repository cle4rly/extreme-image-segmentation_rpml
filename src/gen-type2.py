import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import random

from mpl_toolkits import mplot3d


SUBVOLUME_NUMBER = 8               # m
STARTING_SUBVOLUME_NUMBER = 5     # n >> m
UNIT_CUBE_MARGIN = 0.1              # area to place random voronoi tessellation points


def get_random_points(number):
    points = list()
    for _ in range(number):
        p1 = random.random() * (1+2*UNIT_CUBE_MARGIN) - UNIT_CUBE_MARGIN
        p2 = random.random() * (1+2*UNIT_CUBE_MARGIN) - UNIT_CUBE_MARGIN
        p3 = random.random() * (1+2*UNIT_CUBE_MARGIN) - UNIT_CUBE_MARGIN
        points.append([p1, p2 ,p3])
    return points

def intersection(r1, r2):
    set1 = set(r1)-{-1}
    set2 = set(r2)-{-1}
    return set1.intersection(set2)

def get_neighbors(region, regions):
    neighbors = list()
    for r in regions:
        if len(intersection(r, region)) == 3:
            neighbors.append(r)
    return neighbors

def get_edges_of_vertex(v_index, edges):
    es = list()
    for e in edges:
        if v_index in e:
            es.append(e)
    return es

def get_regions_to_merge(ridge_v, regions):
    merge_rs = list()
    for r in regions:
        if set(ridge_v).issubset(set(r)):
            merge_rs.append(r)
    return merge_rs

r = get_random_points(STARTING_SUBVOLUME_NUMBER)
#print(list([list(x) for x in r]))
vor = Voronoi(r)

print(vor.ridge_vertices)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.plot3D([ran[0] for ran in r], [ran[1] for ran in r], [ran[2] for ran in r], 'o')

for rv in vor.ridge_vertices:
    xs = list()
    ys = list()
    zs = list()
    vs = list()
    for v in rv:
        vs.append(list(vor.vertices[v]))
        xs.append(vor.vertices[v][0])
        ys.append(vor.vertices[v][1])
        zs.append(vor.vertices[v][2])
        
    print(vs)
    ax.plot3D(xs,ys,zs)
plt.show()


# exclude infinite regions (-1) and sort ridge_vertices
vor.ridge_vertices = list(filter(lambda v: -1 not in v, vor.ridge_vertices))
for rv in vor.ridge_vertices:
    rv.sort()
for r in vor.regions:
    r.sort()

merged_regions = list()
for _ in range(STARTING_SUBVOLUME_NUMBER - SUBVOLUME_NUMBER):
    ridge_v = random.choice(vor.ridge_vertices)
    vor.ridge_vertices.remove(ridge_v)
    merged_regions.append(get_regions_to_merge(ridge_v, vor.regions))

def remove_unnecessary_edges(v_index, edges, vertices):
    e = get_edges_of_vertex(v_index, vor.ridge_vertices)
    if len(e) == 1:
        edges.remove(e[0])
        remove_unnecessary_edges(np.where(vertices == e[0][0]), edges, vertices)
        remove_unnecessary_edges(np.where(vertices == e[0][1]), edges, vertices)


#remove unnecessary ridges (single edge on a vertice)
for i in range(len(vor.vertices)):
    remove_unnecessary_edges(i, vor.ridge_vertices, vor.vertices)


