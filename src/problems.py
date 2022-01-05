from os import times
import numpy as np
from numpy.core.fromnumeric import partition
from objects3d import Curve, Point, Plain
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from scipy.spatial import Voronoi
import time
import itertools


class ImageObj:

    def __init__(self, resolution, values) -> None:
        self.resolution = resolution
        self.values: List[List[List[int]]] = values

    def get_intensity(self, point: Point):
        return self.values[int(point.value[0]*self.resolution)][int(point.value[1]*self.resolution)][int(point.value[2]*self.resolution)]    #change!!!


class Problem:

    def __init__(self, object_number, resolution, sigma, sigma_near):
        self.object_number = object_number
        self.resolution = resolution
        self.sigma = sigma
        self.sigma_near = sigma_near

    def plot(self):
        pass


class Type1(Problem):

    def __init__(self, object_number, distance, resolution=500, sigma=0.3, sigma_near=0.3):
        super().__init__(object_number, resolution, sigma, sigma_near)
        self.distance = distance
        self.curves = list()
        self.matrix = np.zeros((self.resolution, self.resolution, self.resolution))

        while len(self.curves) < self.object_number:
            print(len(self.curves))
            new_curve = Curve(self.resolution)
            if self.validate(new_curve):
                self.curves.append(new_curve)

    def too_close(self, curve):
        for c in self.curves:
            for p1 in c.values:
                for p2 in curve.values:
                    if np.linalg.norm(np.array(p1.value) - np.array(p2.value)) < self.distance:
                        return True
        return False
    
    def validate(self, curve) -> bool:
        if curve.too_short() or self.too_close(curve):
            return False
        return True

    def point_on_curve(self, point) -> bool:
        for c in self.curves:
            if c.point_included(point):
                return True
        return False

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for c in self.curves:
            xs = [p.value[0] for p in c.values]
            ys = [p.value[1] for p in c.values]
            zs = [p.value[2] for p in c.values]
            ax.plot3D(xs, ys, zs)

        ax.set_title('type 1')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def get_curve_distance(self, point):
        next_points = [point]
        checked = set()
        while len(next_points):
            p = next_points.pop()
            print(len(next_points))
            if self.point_on_curve(p):
                return np.linalg.norm(np.array(p.value) - np.array(point.value))
            for n in p.get_6connected_nbhood(self.resolution):
                if n not in checked:
                    next_points.append(n)
            checked.add(p)

    # y-column, x-image, z-row
    def to_image(self, image_number):
        array = np.zeros((self.resolution, self.resolution))
        gaussian_near = 100 + np.random.normal(0, self.sigma_near, (self.resolution, self.resolution)) * 255
        gaussian_far = abs(np.random.normal(0, self.sigma, (self.resolution, self.resolution))) * 255
        values = [v/self.resolution for v in range(self.resolution)]
        values = [Point(v) for v in values]
        vecdist = np.vectorize(self.get_curve_distance)
        print(f"{image_number}")
        pixel = [p for p in itertools.product(values, repeat=2)]
        array = vecdist(pixel)
        return np.asarray(array)

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

    def save_image_stack2(self, location):
        i = 0
        for row in self.matrix:
            img = Image.fromarray(row.astype('uint8'))
            img = img.convert("L")
            img.save(f"{location}/image{i}.png")
            i += 1

    def save_cube(self):
        white = 255
        done = set()
        next_pixel = [[]]
        for curve in self.curves:
            next_pixel[0].extend(curve.values)
        next_pixel[0] = set(next_pixel[0])
        print(f"len of values {len(next_pixel[0])}")
        dist = 0
        while len(next_pixel):
            current_pixel_set = next_pixel.pop()
            next_pixel_set = set()
            print(f"dist {dist}")
            for pixel in current_pixel_set:
                x = int(round(pixel.value[0] * self.resolution))
                y = int(round(pixel.value[1] * self.resolution))
                z = int(round(pixel.value[2] * self.resolution))
                self.matrix[x][y][z] = round(white - dist)
                if white-dist < 0:
                    print(f"negative {white} - {dist}")
                    print(f"{x} {y} {z}")
                # if x == 29 or x == 58:
                #     print(f"seltsam {white} - {dist}")
                #     print(f"{x} {y} {z}")
                done.add(pixel)
                for n in pixel.get_18connected_nbhood(self.resolution):
                    nx = int(round(n.value[0] * self.resolution))
                    ny = int(round(n.value[1] * self.resolution))
                    nz = int(round(n.value[2] * self.resolution))
                    if not self.matrix[nx][ny][nz] and n not in current_pixel_set:
                        next_pixel_set.add(n)
            if next_pixel_set:
                next_pixel.append(next_pixel_set)
            dist += (white/self.resolution)*2
        print(self.matrix)


p1 = Type1(13, 0.01, 100)
t1 = time.time()
p1.save_cube()
t2 = time.time()
print(f"time {t2-t1}")
p1.save_image_stack2("data/test5")
p1.plot()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_zlim(0,1)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

# p = Type2(30)
# bp = p.get_boundary_pixel(ax)
# p.save_image_stack("data/voronoi5", bp)


# plt.show()



# schauen ob nicht null statt schon benutzt set
# nur äußerste speicehenr