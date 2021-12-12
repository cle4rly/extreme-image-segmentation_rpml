from os import sep
import random
from enum import Enum
from PIL.Image import new
import numpy as np
from numpy.core.numerictypes import obj2sctype
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import itertools


class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2


class Point:
    def __init__(self, value=None) -> None:
        self.value = value if value else [random.random(), random.random(), random.random()]

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self):
        return str(self)

    def __eq__(self, __o: object) -> bool:
        return self.value == __o.value

    def __hash__(self) -> int:
        return hash(repr(self))

    def copy(self):
        return Point(self.value.copy())

    def get_pixel_value(self, resolution):
        values = list()
        for v in self.value:
            rounding_up = 0
            if v % (1/resolution) >= 1 / (resolution*2):
                rounding_up = 1
            values.append(round(v - v % (1/resolution) +
                          rounding_up * (1/resolution), 10))
        return values

    def round_to_pixel(self, resolution):
        self.value = self.get_pixel_value(resolution)

    def in_unit_cube(self):
        for v in self.value:
            if 0 > v or 1 <= v:
                return False
        return True

    def get_shifted_points(self, offsets):
        shifted = list()
        for off in offsets:
            shifted.append(
                Point((self.value[0]+off[0], self.value[1]+off[1], self.value[2]+off[2])))
        return shifted

    def get_6connected_nbhood(self, resolution):
        nbh = list()
        for dim, dir in itertools.product(Dimension, {-1, 1}):
            new_p = self.copy()
            new_p.value[dim.value] += dir/resolution
            new_p.round_to_pixel(resolution)
            if new_p.in_unit_cube():
                nbh.append(new_p)
        return nbh

    def get_18connected_nbhood(self, resolution):
        nbh = self.get_6connected_nbhood(resolution)
        for dim in Dimension:
            for dir1, dir2 in itertools.product({-1, 1}, repeat=2):
                new_p = self.copy()
                new_p.value[dim.value] += dir1/resolution
                new_p.value[(dim.value+1) % 3] += dir2/resolution
                new_p.round_to_pixel(resolution)
                if new_p.in_unit_cube():
                    nbh.append(new_p)
        return nbh

    def get_26connected_nbhood(self, resolution):
        nbh = self.get_18connected_nbhood(resolution)
        for dir1, dir2, dir3 in itertools.product({-1, 1}, repeat=3):
            new_p = self.copy()
            new_p.value[0] += dir1/resolution
            new_p.value[1] += dir2/resolution
            new_p.value[2] += dir3/resolution
            new_p.round_to_pixel(resolution)
            if new_p.in_unit_cube():
                nbh.append(new_p)
        return nbh


class Curve:

    def __init__(self, resolution, min_points=3, max_points=6, min_values=10):
        self.resolution = resolution
        self.interpolation_points = list()
        self.increasing_dim = random.choice(list(Dimension))

        point_num = random.randint(min_points, max_points)
        for _ in range(point_num):
            self.interpolation_points.append(Point().value)

        self.interpolation_points.sort(
            key=lambda x: x[self.increasing_dim.value])
        self.min_values = min_values
        self.values = self.calc_all_values()
        self.values = self.calc_unit_cube_values(self.estimate_length())
        self.offset = None
        self.neighbor_pixels = None

    def estimate_length(self):
        length = 0
        for i in range(len(self.values)-2):
            length += np.linalg.norm(
                np.array(self.values[i].value) - np.array(self.values[i+1].value))
        return length

    # returns list of point objects on curve
    def calc_all_values(self, factor=1):
        start_arg = self.interpolation_points[0][self.increasing_dim.value]
        end_arg = self.interpolation_points[-1][self.increasing_dim.value]
        arg_points = [p[self.increasing_dim.value]
                      for p in self.interpolation_points]
        arguments = np.arange(start_arg, end_arg, 1/(self.resolution*factor))

        value_dim1 = (self.increasing_dim.value + 1) % 3
        value_dim2 = (self.increasing_dim.value + 2) % 3
        spline_value1 = CubicSpline(
            arg_points, [p[value_dim1] for p in self.interpolation_points])
        spline_value2 = CubicSpline(
            arg_points, [p[value_dim2] for p in self.interpolation_points])

        curve = list()
        for arg in arguments:
            new_p = Point()
            new_p.value[self.increasing_dim.value] = arg
            new_p.value[value_dim1] = float(spline_value1(arg))
            new_p.value[value_dim2] = float(spline_value2(arg))
            curve.append(new_p)

        return curve

    # use only part of spline inside unit cube, when spline intersects unit cube multiple times, choose one at random
    def calc_unit_cube_values(self, amount):
        ranges = list()
        current_range = list()
        curve = self.calc_all_values(amount)

        for i in range(len(curve)):
            curve[i].round_to_pixel(self.resolution)
            if not curve[i].in_unit_cube():
                if len(current_range) == 0:
                    continue
                else:
                    ranges.append(current_range.copy())
                    current_range = []
            else:
                current_range.append(i)
        ranges.append(current_range)

        ranges.sort(key=lambda x: len(x))
        return curve[ranges[-1][0]:ranges[-1][-1]]

    def too_short(self) -> bool:
        return len(self.values) < self.min_values

    def point_included(self, point) -> bool:
        return point.value in self.values

    def get_offsets(self, distance):
        if not self.offset:
            step = 1/self.resolution
            max_pixel_offset = round(distance/step)
            positive_offset = list()
            for x in range(max_pixel_offset+1):
                for y in range(max_pixel_offset+1):
                    for z in range(max_pixel_offset+1):
                        if np.linalg.norm(np.array([x*step, y*step, z*step])) <= distance:
                            positive_offset.append((x*step, y*step, z*step))
                        else:
                            break
            positive_offset.remove((0, 0, 0))
            values = [-1, 1]
            cubes = list(itertools.product(values, repeat=3))
            cubes.remove((1, 1, 1))
            offset = set(positive_offset.copy())
            for cube in cubes:
                for off in positive_offset:
                    offset.add((cube[0]*off[0], cube[1]
                               * off[1], cube[2]*off[2]))
            self.offset = [list(off) for off in list(offset)]
        return self.offset

    def get_neighbor_pixels(self, distance):
        offsets = self.get_offsets(distance)
        if not self.neighbor_pixels:
            neighbors = set()
            last_off = set()
            for p in self.values:
                current_off = set(p.get_shifted_points(offsets))
                for n in current_off-last_off:
                    n.round_to_pixel(self.resolution)
                    if n.in_unit_cube() and p.value != n.get_pixel_value(self.resolution):
                        neighbors.add(
                            tuple(n.get_pixel_value(self.resolution)))
                last_off = current_off.copy()
            self.neighbor_pixels = list(map(lambda x: list(x), neighbors))
        return self.neighbor_pixels

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        curve = self.values
        xs = [p.value[0] for p in curve]
        ys = [p.value[1] for p in curve]
        zs = [p.value[2] for p in curve]
        ax.plot3D(xs, ys, zs)

        px = [p[0] for p in self.interpolation_points]
        py = [p[1] for p in self.interpolation_points]
        pz = [p[2] for p in self.interpolation_points]
        ax.plot3D(px, py, pz, 'o')

        ax.set_title('type 1')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

# c = Curve(500)
# c.plot()

p = Point([0.1, 0.2, 0.3])
print(p.get_26connected_nbhood(500))

# close spaces in splines (more values? -> relative)
# calc dist random points as reference
