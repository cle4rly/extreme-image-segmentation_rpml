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


class Curve:

    def __init__(self, min_points=4, max_points=6):
        self.interpolation_points = list()
        self.increasing_dim = random.choice(list(Dimension))
    
        point_num = random.randint(min_points, max_points)
        arguments = list()
        while len(self.interpolation_points) < point_num:
            new_point = [random.random(), random.random(), random.random()]
            arg = new_point[self.increasing_dim.value]
            if arg not in arguments and arg != 0 and arg !=1:
                self.interpolation_points.append(new_point)
                arguments.append(arg)

        boundary_values = [0,1]
        for point in [self.interpolation_points[0], self.interpolation_points[-1]]:
            point[random.choice(list(Dimension)).value] = random.choice(boundary_values)
        
        self.interpolation_points.sort(key=lambda x:x[self.increasing_dim.value])
        
    def get_values(self, resolution=500):
        step = 1/resolution
        start_arg = self.interpolation_points[0][self.increasing_dim.value]
        end_arg = self.interpolation_points[-1][self.increasing_dim.value]
        arg_points = [p[self.increasing_dim.value] for p in self.interpolation_points]
        if start_arg < end_arg:
            arguments = np.arange(start_arg, end_arg, step)
        else:
            arguments = np.arange(end_arg, start_arg, step)
            arg_points.reverse()

        value_dim1 = (self.increasing_dim.value + 1) % 3
        value_dim2 = (self.increasing_dim.value + 2) % 3
        spline_value1 = CubicSpline(arg_points, [p[value_dim1] for p in self.interpolation_points])
        spline_value2 = CubicSpline(arg_points, [p[value_dim2] for p in self.interpolation_points])
        
        if self.increasing_dim == Dimension.X:
            return [[x, float(spline_value1(x)), float(spline_value2(x))] for x in arguments]
        elif self.increasing_dim == Dimension.Y:
            return [[float(spline_value2(y)), y, float(spline_value1(y))] for y in arguments]
        elif self.increasing_dim == Dimension.Z:
            return [[float(spline_value1(z)), float(spline_value2(z)), z] for z in arguments]
   
    def leaves_unit_cube(self) -> bool:
        for p in self.get_values():
            for d in Dimension:
                if p[d.value] > 1 or p[d.value] < 0:
                    return True
        return False

    def point_included(self, point) -> bool:
        pass

    def plot(self, resolution):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        xs = [p[0] for p in self.get_values(resolution)]
        ys = [p[1] for p in self.get_values(resolution)]
        zs = [p[2] for p in self.get_values(resolution)]
        ax.plot3D(xs, ys, zs)
        
        ax.set_title('type 1')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()



c = Curve()
print(f"increasing in {c.increasing_dim}")
for p in c.interpolation_points:
    print(p)
print(c.leaves_unit_cube())
for p in c.get_values(20):
    print(p)
c.plot(50)