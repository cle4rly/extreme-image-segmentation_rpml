import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import random

from mpl_toolkits import mplot3d


SPLINES_NUMBER = int(input("how many splines should be generated? "))
DISTANCE = float(input("which distance should they have? "))
PIXEL_NUMBER = 500
MAX_POINTS = 6
MIN_POINTS = 4  #has to be at least 4 for cubic interpolation
MIN_POINTS_INCLUDED = 3
MIN_SPLINE_LENGTH = 20
MAX_ITERATIONS = 50


def in_unit_cube(x,y,z):
    return 0<=x<=1 and 0<=y<=1 and 0<=z<=1

def spline_in_unit_cube(xs, spline):
    for (x,y,z) in spline:
        if not in_unit_cube(x) or not in_unit_cube(y) or not in_unit_cube(z):
            return False
    return True

def round_to_pixel(x):
    rounding_up = 0
    if x % (1/PIXEL_NUMBER) >= 1 / (PIXEL_NUMBER*2):
        rounding_up = 1
    return round(x - x % (1/PIXEL_NUMBER) + rounding_up * (1/PIXEL_NUMBER), 10)

def point_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def too_close(spline1, spline2, distance):
    for p1 in zip(spline1[0], spline1[1], spline1[2]):
        for p2 in zip(spline2[0], spline2[1], spline2[2]):
            if point_distance(p1, p2) < distance:
                return True
    return False

def includes_enough_points(spline, points, number):
    n = 0
    for p in points:
        if p not in zip(spline[0],spline[1],spline[2]):
            n += 1
    return n >= number

def generate_new_spline():

    xs = list()
    ys = list()
    zs = list()

    point_num = random.randint(MIN_POINTS, MAX_POINTS)
    for i in range(point_num):
        xs.append(random.random())
        ys.append(random.random())
        zs.append(random.random())

    new_xs = np.arange(0,1,1/PIXEL_NUMBER)

    xs.sort()
    spline_y = CubicSpline(xs,ys)
    spline_z = CubicSpline(xs,zs)

    # use only part of spline inside unit cube, when spline intersects unit cube multiple times, choose one at random
    ranges = list()
    current_range = list()
    for (i,x,y,z) in zip(range(PIXEL_NUMBER), new_xs, spline_y(new_xs), spline_z(new_xs)):
        if not in_unit_cube(x,y,z):
            if len(current_range) == 0:
                continue
            else:
                ranges.append(current_range.copy())
                current_range = []
        else:
            current_range.append(i)

    if len(ranges) == 0:
        return ([],[[],[],[]])

    chosen_range = ranges[random.randint(0,len(ranges)-1)] if len(ranges)>1 else ranges[0]
    new_xs = new_xs[chosen_range[0]:chosen_range[-1]]

    #shuffle dimensions, so that not every spline starts on the lower x side
    order = [0,1,2]
    random.shuffle(order)

    spline = [list(new_xs), list(spline_y(new_xs)), list(spline_z(new_xs))]
    spline = [spline[order[0]], spline[order[1]], spline[order[2]]]

    points = [xs, ys, zs]
    points = [points[order[0]], points[order[1]], points[order[2]]]

    #round to pixel values
    rounded_spline = list()
    for values in spline:
        rounded_spline.append(list(map(round_to_pixel, values)))
    
    return (points, rounded_spline)


def generate_splines(number, distance):
    splines = list()
    iterations_without_match = 0

    while len(splines) < number:

        data = generate_new_spline()
        points = data[0]
        spline = data[1]
        too_close_flag = False

        if iterations_without_match > MAX_ITERATIONS:
            print(f"to many iterations, exited with {len(splines)} / {SPLINES_NUMBER} splines")
            break
        
        if not includes_enough_points(spline, points, MIN_POINTS_INCLUDED) or len(spline[0]) < MIN_SPLINE_LENGTH:
            iterations_without_match += 1
            continue

        for s in splines:
            if too_close(s, spline, distance):
                too_close_flag = True
                break
        
        if too_close_flag:
            iterations_without_match += 1
            continue

        splines.append(spline)
        print(f"{len(splines)} / {SPLINES_NUMBER} splines found")
    
    return splines

def analyse_data(splines):
    x_values = list()
    y_values = list()
    z_values = list()
    data_range = np.arange(0,1,1/PIXEL_NUMBER)

    for s in splines:
        x_values.extend(s[0])
        y_values.extend(s[1])
        z_values.extend(s[2])
    
    x_count = list()
    y_count = list()
    z_count = list()

    for i in data_range:
        x_count.append(x_values.count(i))
        y_count.append(y_values.count(i))
        z_count.append(z_values.count(i))

    print(f"average x value: {np.average(x_values)}")
    print(f"average y value: {np.average(y_values)}")
    print(f"average z value: {np.average(z_values)}")

    plt.plot(data_range, x_count)
    plt.plot(data_range, y_count)
    plt.plot(data_range, z_count)
    plt.show()


def plot_splines(data):
    colors = ['#3B3285', '#08A4B1', '#A6CB3F', '#E7D614', '#EEAE21', '#EF517F']
    colors.reverse()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    i = 0
    for spline in data:
        ax.plot3D(spline[0], spline[1], spline[2], colors[i])
        i = (i+1) % len(colors)

    ax.set_title('type 1')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()

def write_to_file(data, name):
    f = open(name, "a")
    f.truncate()
    #f.write(data)
    f.close()



def main():
    splines = generate_splines(SPLINES_NUMBER, DISTANCE)
    write_to_file(str(splines), "data/splines.txt")
    plot_splines(splines)
    #analyse_data(splines)


if __name__ == "__main__":
    main()
