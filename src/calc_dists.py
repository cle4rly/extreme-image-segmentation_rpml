import multiprocessing as mp
from math import sqrt

from matrix_parser import MatrixParser


PATH = "data/9"
RESOLUTION = 200
RES_VALUES = [v/RESOLUTION for v in range(RESOLUTION)]

CURVE_POINTS = list()


def get_curve_distance(point):
    dist = 1.73     # max dist in unit cube
    s = 3           # 1 for every dimension
    
    for cp in CURVE_POINTS:
        x = point[0] - cp[0]
        new_s = x*x
        if new_s >= s:
            continue

        y = point[1] - cp[1]
        new_s += y*y
        if new_s >= s:
            continue

        z = point[2] - cp[2]
        new_s += z*z
        if new_s == 0:
            return 0
        if new_s >= s:
            continue

        new_dist = round(sqrt(new_s),3)
        if new_dist < dist:
            dist = new_dist
            s = new_s

    return dist

def save_dist_files(location):
    matrix = list()
    for i in range(RESOLUTION):
        array = list()
        pool = mp.Pool(mp.cpu_count())
        x_value = i/RESOLUTION

        for y_value in RES_VALUES:
            pixel = [(x_value, y_value, z_value) for z_value in RES_VALUES]
            array.append(pool.map(get_curve_distance, pixel))

        print(f"{i}")
        pool.close()

        matrix.append(array)

    file = open(location, "w")
    file.write(str(matrix))
    file.close()


point_matrix = MatrixParser.read_curves(RESOLUTION, PATH + "/points")
CURVE_POINTS = list()
for arr in point_matrix:
    CURVE_POINTS.extend(arr)

print(CURVE_POINTS)
save_dist_files(PATH+"/dist_matrix")

