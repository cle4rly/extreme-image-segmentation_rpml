from os import sep
import random
from enum import Enum
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
    def __init__(self, x=None, y=None, z=None) -> None:
        self.x = x
        self.y = y
        self.z = z
        if x is None:
            self.x = random.random()
        if y is None:
            self.y = random.random()
        if z is None:
            self.z = random.random()

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"
    
    def __repr__(self):
        return str(self)

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y and self.z == __o.z
    
    def __hash__(self) -> int:
        return hash(repr(self))

    def get_value(self):
        return [self.x, self.y, self.z]
    
    def get_pixel_value(self, resolution):
        values = list()
        for v in [self.x, self.y, self.z]:
            rounding_up = 0
            if v % (1/resolution) >= 1 / (resolution*2):
                rounding_up = 1
            values.append(round(v - v % (1/resolution) + rounding_up * (1/resolution), 10))
        return values

    def round_to_pixel(self, resolution):
        values = self.get_pixel_value(resolution)
        self.x = values[0]
        self.y = values[1]
        self.z = values[2]

    def in_unit_cube(self):
        return 0<=self.x<1 and 0<=self.y<1 and 0<=self.z<1

    def get_shifted_points(self, offsets):
        shifted = list()
        for off in offsets:
            shifted.append(Point(self.x+off[0], self.y+off[1], self.z+off[2]))
        return shifted


class Curve:

    def __init__(self, resolution, min_points=3, max_points=6, min_values=10):
        self.resolution = resolution
        self.interpolation_points = list()
        self.increasing_dim = random.choice(list(Dimension))
    
        point_num = random.randint(min_points, max_points)
        for _ in range(point_num):
            self.interpolation_points.append(Point().get_value())

        self.interpolation_points.sort(key=lambda x:x[self.increasing_dim.value])
        self.min_values = min_values
        self.values = self.get_values()
        self.offset = None
        self.neighbour_pixels = None
        
    def get_values(self):
        step = 1/self.resolution
        start_arg = self.interpolation_points[0][self.increasing_dim.value]
        end_arg = self.interpolation_points[-1][self.increasing_dim.value]
        arg_points = [p[self.increasing_dim.value] for p in self.interpolation_points]
        arguments = np.arange(start_arg, end_arg, step/2)

        value_dim1 = (self.increasing_dim.value + 1) % 3
        value_dim2 = (self.increasing_dim.value + 2) % 3
        spline_value1 = CubicSpline(arg_points, [p[value_dim1] for p in self.interpolation_points])
        spline_value2 = CubicSpline(arg_points, [p[value_dim2] for p in self.interpolation_points])

        if self.increasing_dim == Dimension.X:
            curve = [Point(x, float(spline_value1(x)), float(spline_value2(x))) for x in arguments]
        elif self.increasing_dim == Dimension.Y:
            curve = [Point(float(spline_value2(y)), y, float(spline_value1(y))) for y in arguments]
        elif self.increasing_dim == Dimension.Z:
            curve = [Point(float(spline_value1(z)), float(spline_value2(z)), z) for z in arguments]

        # use only part of spline inside unit cube, when spline intersects unit cube multiple times, choose one at random
        ranges = list()
        current_range = list()
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
        return [p.get_value() for p in curve[ranges[-1][0]:ranges[-1][-1]]]

    def too_short(self) -> bool:
        return len(self.values) < self.min_values

    # more efficient possible
    def point_included(self, point) -> bool:
        return point.get_value() in self.values

    def get_offsets(self, distance):
        if not self.offset:
            step = 1/self.resolution
            max_pixel_offset = round(distance/step)
            positive_offset = list()
            for x in range(max_pixel_offset+1):
                for y in range(max_pixel_offset+1):
                    for z in range(max_pixel_offset+1):
                        if np.linalg.norm(np.array([x*step,y*step,z*step])) <= distance:
                            positive_offset.append((x*step,y*step,z*step))
                        else:
                            break
            positive_offset.remove((0,0,0))
            values = [-1,1]
            cubes = list(itertools.product(values, repeat=3))
            cubes.remove((1,1,1))
            offset = set(positive_offset.copy())
            for cube in cubes:
                for off in positive_offset:
                    offset.add((cube[0]*off[0], cube[1]*off[1], cube[2]*off[2]))
            self.offset = [list(off) for off in list(offset)]
        return self.offset

    def get_neighbour_pixels(self, distance):
        offsets = self.get_offsets(distance)
        if not self.neighbour_pixels:
            neighbours = set()
            last_off = set()
            for p in self.values:
                current_off = set(Point(p[0],p[1],p[2]).get_shifted_points(offsets))
                for n in current_off-last_off:
                    n.round_to_pixel(self.resolution)
                    if n.in_unit_cube() and p != n.get_pixel_value(self.resolution):
                        neighbours.add(tuple(n.get_pixel_value(self.resolution)))
                last_off = current_off.copy()
            neighbours = list(map(lambda x: list(x), neighbours))
            self.neighbour_pixels = neighbours
        return self.neighbour_pixels


    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        curve = self.values
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        zs = [p[2] for p in curve]
        ax.plot3D(xs, ys, zs)
        
        px = [p[0] for p in self.interpolation_points]
        py = [p[1] for p in self.interpolation_points]
        pz = [p[2] for p in self.interpolation_points]
        ax.plot3D(px,py,pz,'o')

        ax.set_title('type 1')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

# c = Curve(500)
# print(len(c.values))
# print(len(c.get_neighbour_pixels(0.007)))


# close spaces in splines (more values? -> relative)