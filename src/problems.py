
import math
import numpy as np
from curve import Curve, Point
import matplotlib.pyplot as plt
from PIL import Image

import time


class Problem:

    def __init__(self, object_number, resolution):
        self.object_number = object_number
        self.resolution = resolution

    def plot(self):
        pass


class Type1(Problem):

    def __init__(self, object_number, distance, resolution=500):
        super().__init__(object_number, resolution)
        self.distance = distance
        self.curves = list()

        while len(self.curves) < self.object_number:
            print(len(self.curves))
            new_curve = Curve(self.resolution)
            if self.validate(new_curve):
                self.curves.append(new_curve)

    def too_close(self, curve):
        for c in self.curves:
            for p1 in c.values:
                for p2 in curve.values:
                    if np.linalg.norm(np.array(p1) - np.array(p2)) < self.distance:
                        return True
        return False
    
    def validate(self, curve) -> bool:
        if curve.too_short() or self.too_close(curve):
            return False
        return True

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for c in self.curves:
            xs = [p[0] for p in c.values]
            ys = [p[1] for p in c.values]
            zs = [p[2] for p in c.values]
            ax.plot3D(xs, ys, zs)

        ax.set_title('type 1')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def on_curve(self, point) -> bool:
        for c in self.curves:
            if c.point_included(point):
                return True
        return False

    def curve_distance(self, point):
        d = math.sqrt(3)    # max dist in unit cube
        for c in self.curves:
            for p in c.values:
                new_d = np.linalg.norm(np.array(p) - np.array(point))
                if new_d < d:
                    d = new_d
                if d == 0:
                    return d
        return d

    # x-column, y-image, z-row
    def to_image(self, image_number):
        array = np.zeros((self.resolution, self.resolution))
        print(f"{image_number}")
        for c in self.curves:
            arg_dim = c.increasing_dim.value
            value_dim1 = (c.increasing_dim.value + 1) %3
            value_dim2 = (c.increasing_dim.value + 2) %3
            for p in c.values:
                if p[arg_dim] == image_number/self.resolution:
                    for n in c.get_neighbour_pixels(p, self.distance):
                        if n[arg_dim] == image_number/self.resolution:
                            array[int(n[value_dim1]*self.resolution)][int(n[value_dim2]*self.resolution)] = 100
                    array[int(p[value_dim1]*self.resolution)][int(p[value_dim2]*self.resolution)] = 255

        return Image.fromarray(np.asarray(array))

    def save_image_stack(self, location):
        for i in range(self.resolution):
            img = self.to_image(i)
            img = img.convert("L")
            img.save(f"{location}/image{i}.png")


class Type2(Problem):

    def __init__(self, object_number, resolution=500):
        super().__init__(object_number, resolution)




p1 = Type1(15, 0.05)
p1.plot()
print(p1.object_number)
p1.save_image_stack("data/curves3")
print("done")