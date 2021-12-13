import re
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from PIL import Image
import random
from statistics import mean
from objects3d import Point


SUBVOLUME_NUMBER = 8               # m
STARTING_SUBVOLUME_NUMBER = 25     # n >> m
UNIT_CUBE_MARGIN = 0.1             # area to place random voronoi tessellation points


class Plain:
    def __init__(self, points: Tuple[Point, Point, Point]) -> None:
        v1 = [points[1].value[0] - points[0].value[0], points[1].value[1] - points[0].value[1], points[1].value[2] - points[0].value[2]]
        v2 = [points[2].value[0] - points[0].value[0], points[2].value[1] - points[0].value[1], points[2].value[2] - points[0].value[2]]
        crs = np.cross(v1,v2)
        n = crs / np.linalg.norm(crs)
        self.f = lambda x,y,z : n[0] * (x-points[0].value[0]) + n[1] * (y-points[0].value[1]) + n[2] * (z-points[0].value[2])
    
    def distance_to_point(self, point: Point) -> float:
        return self.f(point.value[0], point.value[1], point.value[2])

    def contains_pixel(self, pixel: Point, resolution) -> bool:
        return abs(self.distance_to_point(pixel)) < 1/(resolution*2)


def get_random_points(number):
    points = list()
    for _ in range(number):
        x = random.random() * (1+2*UNIT_CUBE_MARGIN) - UNIT_CUBE_MARGIN
        y = random.random() * (1+2*UNIT_CUBE_MARGIN) - UNIT_CUBE_MARGIN
        z = random.random() * (1+2*UNIT_CUBE_MARGIN) - UNIT_CUBE_MARGIN
        points.append([x, y, z])
    return points

def intersection(r1, r2):
    set1 = set(r1)-{-1}
    set2 = set(r2)-{-1}
    return set1.intersection(set2)

def get_neighbor_regions(region, regions):
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

# returns set of points on ridge edge
def construct_ridge_edge(p1:Point, p2:Point):
    if not p1.in_unit_cube() and not p2.in_unit_cube():
        return {}
    if not p1.in_unit_cube():
        p1,p2 = p2,p1

    fx = lambda r : p1.value[0] + r * (p2.value[0]-p1.value[0])
    fy = lambda r : p1.value[1] + r * (p2.value[1]-p1.value[1])   # r from 0 to 1
    fz = lambda r : p1.value[2] + r * (p2.value[2]-p1.value[2])

    step = 1/(500*np.linalg.norm(np.array(p1.value) - np.array(p2.value)))

    ridge = set()
    r=0
    while r<1:
        r += step
        p = Point([fx(r), fy(r), fz(r)])
        p.round_to_pixel(500)
        if not p.in_unit_cube():
            break
        ridge.add(p)
        for n in p.get_18connected_nbhood(500):
            if n.in_unit_cube():
                ridge.add(n)
    
    return ridge

# returns set of inner points of ridge area
def construct_ridge_area(ridge_vertices):
    xs = [rv.value[0] for rv in ridge_vertices]
    ys = [rv.value[1] for rv in ridge_vertices]
    zs = [rv.value[2] for rv in ridge_vertices]
    center = Point([mean(xs), mean(ys), mean(zs)])
    center.round_to_pixel(500)

    print(ridge_vertices)
    print(f"c {center}")

    edge_points = set()
    for i in range(len(ridge_vertices)):
        edge_points.update(construct_ridge_edge(ridge_vertices[i], ridge_vertices[(i+1) % len(ridge_vertices)]))
    #plt.plot([e.value[0] for e in edge_points],[e.value[1] for e in edge_points], 'o')


    plain = Plain((ridge_vertices[0], ridge_vertices[1], ridge_vertices[2]))
    pixel_to_expand = [center]
    current_pixel = None
    area = set()
    k=0
    while len(pixel_to_expand) and k<250000:
        #print(f"to exp {len(pixel_to_expand)}")
        #print()
        k+=1
        if current_pixel:
            area.add(current_pixel)
        current_pixel = pixel_to_expand.pop()
        #print(f"crp {current_pixel}")
        for n in current_pixel.get_26connected_nbhood(500):
            if n.in_unit_cube() and plain.contains_pixel(n,500) and not n in edge_points and not n in area:
                pixel_to_expand.append(n)
    print(k)

    print(len(area))
    area.update(set(edge_points))
    print(len(area))
    return area

def get_boundary_pixel(vor):
    pixel = set()
    for rv in vor.ridge_vertices:
        if -1 in rv:
            continue
        rv_abs = list()
        for v in rv:
            v = Point(list(vor.vertices[v]))
            v.round_to_pixel(500)
            rv_abs.append(v)
        pixel.update(construct_ridge_area(rv_abs))
    return pixel

# y-column, x-image, z-row
def to_image(image_number,boundary):
    array = np.zeros((500, 500))
    print(f"{image_number}")
    for p in boundary:
        if p.value[0] == image_number/500:
            array[int(p.value[1]*500)][int(p.value[2]*500)] = 255
    return np.asarray(array)

# not same as img obj (random noise)
def save_image_stack(location,boundary):
    for i in range(500):
        img = to_image(i,boundary)
        img = Image.fromarray(img)
        img = img.convert("L")
        img.save(f"{location}/image{i}.png")

def plot(vor):
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
        if -1 in rv:
            continue
        rv_abs = list(map(lambda x: list(vor.vertices[x]), rv))
        rv_abs.append(rv_abs[0])

        for i in range(len(rv_abs)-1):
            ridge_x = list()
            ridge_y = list()
            ridge_z = list()
            for c in construct_ridge_edge(rv_abs[i], rv_abs[i+1]):
                #print(c)
                ridge_x.append(c[0])
                ridge_y.append(c[1])
                ridge_z.append(c[2])

            ax.plot3D(ridge_x, ridge_y, ridge_z, color="black")
    plt.show()


r = get_random_points(STARTING_SUBVOLUME_NUMBER)
#print(list([list(x) for x in r]))
vor = Voronoi(r)


def remove_unnecessary_edges(v_index, edges, vertices):
    e = get_edges_of_vertex(v_index, vor.ridge_vertices)
    if len(e) == 1:
        edges.remove(e[0])
        remove_unnecessary_edges(np.where(vertices == e[0][0]), edges, vertices)
        remove_unnecessary_edges(np.where(vertices == e[0][1]), edges, vertices)

save_image_stack("data/voronoi2", get_boundary_pixel(vor))

points = [[0.928, 1.03, 0.434], [0.974, 0.854, 0.366], [0.964, 0.66, 0.208], [0.696, 0.75, -0.094], [0.524, 1.006, -0.14], [0.728, 1.32, 0.374]]
points = [Point(p) for p in points]
area = construct_ridge_area(points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


edge_points = set()
for i in range(len(points)):
    edge_points.update(construct_ridge_edge(points[i], points[(i+1) % len(points)]))
ax.plot3D([a.value[0] for a in area], [a.value[1] for a in area], [a.value[2] for a in area], 'o')
ax.plot3D([e.value[0] for e in edge_points],[e.value[1] for e in edge_points],[e.value[2] for e in edge_points], 'o')
#ax.plot3D([p.value[0] for p in points], [p.value[1] for p in points], [p.value[2] for p in points])

plt.show()

# center nicht in unit cube -> area raus
# edge scheint noch zu breit