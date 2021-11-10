from os import sep
import random
from enum import Enum
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


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

    def get_value(self):
        return [self.x, self.y, self.z]
    
    def in_unit_cube(self):
        return 0<=self.x<=1 and 0<=self.y<=1 and 0<=self.z<=1


class Curve:

    def __init__(self, min_points=3, max_points=6, min_values=10, resolution=500):
        self.interpolation_points = list()
        self.increasing_dim = random.choice(list(Dimension))
    
        point_num = random.randint(min_points, max_points)
        for _ in range(point_num):
            self.interpolation_points.append(Point().get_value())

        self.interpolation_points.sort(key=lambda x:x[self.increasing_dim.value])
        self.min_values = min_values
        self.values = self.get_values(resolution)
        
    def get_values(self, resolution):
        step = 1/resolution
        start_arg = self.interpolation_points[0][self.increasing_dim.value]
        end_arg = self.interpolation_points[-1][self.increasing_dim.value]
        arg_points = [p[self.increasing_dim.value] for p in self.interpolation_points]
        arguments = np.arange(start_arg, end_arg, step)

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
        return [c.get_value() for c in curve[ranges[-1][0]:ranges[-1][-1]]]

    def too_short(self) -> bool:
        return len(self.values) < self.min_values

    def point_included(self, point) -> bool:
        pass

    def plot(self, resolution):
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

