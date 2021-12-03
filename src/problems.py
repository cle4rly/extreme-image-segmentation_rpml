
import math
import numpy as np
from numpy.core.records import array
from curve import Curve, Point
import matplotlib.pyplot as plt
from PIL import Image
from typing import List


class ImageObj:

    def __init__(self, resolution, values) -> None:
        self.resolution = resolution
        self.values: List[List[List[int]]] = values

    def get_intensity(self, point):
        return self.values[point[0]*self.resolution][point[1]*self.resolution][point[2]*self.resolution]    #change!!!


class Problem:

    def __init__(self, object_number, resolution, sigma, sigma_near):
        self.object_number = object_number
        self.resolution = resolution
        self.sigma = sigma
        self.sigma_near = sigma_near

    def plot(self):
        pass


class Type1(Problem):

    def __init__(self, object_number, distance, resolution=500, sigma=0.1, sigma_near=0.3):
        super().__init__(object_number, resolution, sigma, sigma_near)
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

    # y-column, x-image, z-row
    def to_image(self, image_number):
        array = np.zeros((self.resolution, self.resolution))
        gaussian = 100 + np.random.normal(0, self.sigma_near, (self.resolution, self.resolution)) * 255
        print(f"{image_number}")
        for c in self.curves:
            for n in c.get_neighbor_pixels(self.distance):
                if n[0] == image_number/self.resolution:
                    y = int(n[1]*self.resolution)
                    z = int(n[2]*self.resolution)
                    array[y][z] = gaussian[y][z]
            for p in c.values:
                if p[0] == image_number/self.resolution:
                    array[int(p[1]*self.resolution)][int(p[2]*self.resolution)] = 255

        return np.asarray(array)

    def add_noise(self, img):
        gaussian = abs(np.random.normal(0, self.sigma, (self.resolution, self.resolution))) * 255
        return img + gaussian

    def get_image_object(self):
        array = [0] * self.resolution
        for i in range(self.resolution):
            image = self.add_noise(self.to_image(i))
            array[i] = [list(img) for img in image]
        return ImageObj(self.resolution, array)

    # not same as img obj (random noise)
    def save_image_stack(self, location):
        for i in range(self.resolution):
            img = self.to_image(i)
            noisy_img = Image.fromarray(self.add_noise(img))
            noisy_img = noisy_img.convert("L")
            noisy_img.save(f"{location}/image{i}.png")


class Type2(Problem):

    def __init__(self, object_number, resolution=500):
        super().__init__(object_number, resolution)




# p1 = Type1(5, 0.005)
# p1.plot()
# print(p1.object_number)
# p1.save_image_stack("data/curves15")
# print("done")