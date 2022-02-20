import numpy as np
from numpy.core.fromnumeric import mean
from objects3d import Curve, Point
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import time
import multiprocessing as mp


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
        self.res_values = np.array([v/self.resolution for v in range(self.resolution)])

        while len(self.curves) < self.object_number:
            print(len(self.curves))
            new_curve = Curve(self.resolution)
            if self.validate(new_curve):
                self.curves.append(new_curve)
        
        file = open("data/values13", "w")

        self.curve_points = list()
        for curve in self.curves:
            file.write(str(curve.values)+"\n\n")
            self.curve_points.extend(curve.values)
        print(self.curve_points)
        print(len(self.curve_points))

        file.close()

        self.gaussian_near = 100 + np.random.normal(0, self.sigma_near, (self.resolution, self.resolution)) * 255
        self.gaussian_far = abs(np.random.normal(0, self.sigma, (self.resolution, self.resolution))) * 255

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

    def get_curve_distance2(self, point):
        print(f"poi {point}")
        next_points = [point]
        checked = set()
        while len(next_points):
            p = next_points.pop()
            if self.point_on_curve(p):
                return np.linalg.norm(np.array(p.value) - np.array(point.value))
            for n in p.get_6connected_nbhood(self.resolution):
                if n not in checked:
                    next_points.append(n)
            checked.add(p)

    def get_curve_distance3(self, point):
        dist = np.float(1.73)
        for val in self.curve_points:
            new_dist = round(np.linalg.norm(np.array(point) - np.array(val)),2)
            if new_dist == 0:
                return 0
            if new_dist < dist:
                dist = new_dist
        return dist
    
    def get_curve_distance(self, point):
        dist = 1.73
        for curve in self.curves:
            for val in curve.values:
                new_dist = round(np.linalg.norm(np.array(point.value) - np.array(val.value)),2)
                if new_dist < dist:
                    dist = new_dist
                if new_dist == 0:
                    return 0
        return dist

    # y-column, x-image, z-row
    def to_image(self, image_number):
        array = list()
        pool = mp.Pool(mp.cpu_count())
        x_value = image_number/self.resolution
        for y in range(self.resolution):
            pixel = [[x_value, y/self.resolution, z_value] for z_value in self.res_values]
            dists = pool.map(self.get_curve_distance3, pixel)
            array.append([round((255-(d*255))/2) for d in dists])

        print(f"{image_number}")
        pool.close()
        return Image.fromarray(np.uint8(array), 'L')

    def get_image_object(self):
        array = [0] * self.resolution
        for i in range(self.resolution):
            image = self.add_noise(self.to_image(i))
            array[i] = [list(img) for img in image]
        return ImageObj(self.resolution, array)

    # not same as img obj (random noise)
    def save_image_stack(self, location):
        times = list()
        for i in range(self.resolution):
            t1 = time.time()
            img = self.to_image(i)
            t2 = time.time()
            times.append(t2-t1)
            print(t2-t1)
            print(f"new mean: {mean(times)}")
            img.save(f"{location}/image{i}.png")



p1 = Type1(10, 0.02, 500)
#p1.save_image_stack("data/curves21")
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