import numpy as np
from objects3d import Curve, Point, Plain
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from scipy.spatial import Voronoi
from statistics import mean


class ImageObj:

    def __init__(self, resolution, values) -> None:
        self.resolution = resolution
        self.values: List[List[List[int]]] = values

    def get_intensity(self, point: Point):
        print(point)
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
                if p.value[0] == image_number/self.resolution:
                    array[int(p.value[1]*self.resolution)][int(p.value[2]*self.resolution)] = 255

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

    def __init__(self, object_number, resolution=500, sigma=0.1, sigma_near=0.3):
        super().__init__(object_number, resolution, sigma, sigma_near)
        points = list()
        for _ in range(object_number):
            points.append(Point())
        vor = Voronoi([p.value for p in points])
        self.ridge_vertices = self.get_absolute_ridge_vertices(vor.ridge_vertices, vor.vertices)
        self.project_vertices_to_unit_cube()

    def get_absolute_ridge_vertices(self, ridge_vertices, vertices):
        abs_ridge_vertices = list()
        for rv in ridge_vertices:
            if -1 in rv:
                continue
            abs_rv = list()
            for v in rv:
                v = Point(list(vertices[v]))
                v.round_to_pixel(self.resolution)
                abs_rv.append(v)
            abs_ridge_vertices.append(abs_rv)
        return abs_ridge_vertices

    def all_in_unit_cube(self, points: List[Point]):
        for p in points:
            if not p.in_unit_cube():
                return False
        return True
    
    def all_outside_unit_cube(self, points: List[Point]):
        for p in points:
            if p.in_unit_cube():
                return False
        return True

    #TODO overlaps with construct_ridge_edge
    def project_vertex_to_unit_cube(self, v0:Point, v1:Point):
        swapped = False
        if v0.in_unit_cube():
            v0,v1 = v1,v0
            swapped = True

        fx = lambda r : v0.value[0] + r * (v1.value[0]-v0.value[0])
        fy = lambda r : v0.value[1] + r * (v1.value[1]-v0.value[1])
        fz = lambda r : v0.value[2] + r * (v1.value[2]-v0.value[2])

        step = 1/(self.resolution*np.linalg.norm(np.array(v0.value) - np.array(v1.value)))

        r=0
        while r<1:
            r += step
            p = Point([fx(r), fy(r), fz(r)])
            p.round_to_pixel(self.resolution)
            if p.in_unit_cube():
                if swapped:
                    return v1,p
                return p,v1

    # TODO
    def project_vertices_to_unit_cube(self):
        outside = []
        for vertices in self.ridge_vertices:
            if self.all_in_unit_cube(vertices):
                continue
            if self.all_outside_unit_cube(vertices):
                outside.append(vertices)
            to_remove = []
            for i in range(len(vertices)):
                v0 = vertices[(i-1) % len(vertices)]
                v1 = vertices[i]
                v2 = vertices[(i+1) % len(vertices)]
                if self.all_in_unit_cube([v0, v1, v2]):
                    continue
                if self.all_outside_unit_cube([v0, v1, v2]):
                    to_remove.append(v1)
                if v0.in_unit_cube() != v1.in_unit_cube():
                    v0, v1 = self.project_vertex_to_unit_cube(v0, v1)
            for p in to_remove:
                vertices.remove(p)
        for vertices in outside:
            self.ridge_vertices.remove(vertices)

    # returns set of points on ridge edge
    def construct_ridge_edge(self, p1:Point, p2:Point):
        if not p1.in_unit_cube() and not p2.in_unit_cube():
            return {}

        if not p1.in_unit_cube():
            p1,p2 = p2,p1

        fx = lambda r : p1.value[0] + r * (p2.value[0]-p1.value[0])
        fy = lambda r : p1.value[1] + r * (p2.value[1]-p1.value[1])
        fz = lambda r : p1.value[2] + r * (p2.value[2]-p1.value[2])

        step = 1/(self.resolution*np.linalg.norm(np.array(p1.value) - np.array(p2.value)))

        ridge = set()
        r=0
        while r<1:
            r += step
            p = Point([fx(r), fy(r), fz(r)])
            p.round_to_pixel(self.resolution)
            if not p.in_unit_cube():
                break
            ridge.add(p)
            for n in p.get_18connected_nbhood(self.resolution):
                if n.in_unit_cube():
                    ridge.add(n)

        return ridge

    # returns set of inner points of ridge area
    def construct_ridge_area(self, ridge_vertices):
        xs = [rv.value[0] for rv in ridge_vertices]
        ys = [rv.value[1] for rv in ridge_vertices]
        zs = [rv.value[2] for rv in ridge_vertices]
        center = Point([mean(xs), mean(ys), mean(zs)])
        center.round_to_pixel(self.resolution)
        print(f"cent {center}")
        print(f"verts {ridge_vertices}")

        edge_points = set()
        for i in range(len(ridge_vertices)):
            edge_points.update(self.construct_ridge_edge(ridge_vertices[i], ridge_vertices[(i+1) % len(ridge_vertices)]))

        plain = Plain((ridge_vertices[0], ridge_vertices[1], ridge_vertices[2]))
        pixel_to_expand = [center]
        current_pixel = None
        area = set()
        i = 0
        while len(pixel_to_expand) and i<50000:
            i+=1
            if current_pixel:
                area.add(current_pixel)
            current_pixel = pixel_to_expand.pop()
            for n in current_pixel.get_26connected_nbhood(500):
                if n.in_unit_cube() and plain.contains_pixel(n,500) and not n in edge_points and not n in area:
                    pixel_to_expand.append(n)
        
        print(len(area))
        area.update(set(edge_points))
        return area
    
    def get_boundary_pixel(self,ax):
        pixel = set()
        print(f"len rvs = {len(self.ridge_vertices)}")
        i = 0
        for rv in self.ridge_vertices:
            print(f"completed {i} of {len(self.ridge_vertices)}")
            i+=1
            b = self.construct_ridge_area(rv)
            pixel.update(b)
            ax.plot3D([e.value[0] for e in b],[e.value[1] for e in b],[e.value[2] for e in b])
        return pixel

    # y-column, x-image, z-row
    def to_image(self, image_number, boundary):
        array = np.zeros((self.resolution, self.resolution))
        print(f"{image_number}")
        for p in boundary:
            if p.value[0] == image_number/self.resolution:
                array[int(p.value[1]*self.resolution)][int(p.value[2]*self.resolution)] = 255
        return np.asarray(array)

    # not same as img obj (random noise)
    def save_image_stack(self, location, boundary):
        for i in range(self.resolution):
            img = self.to_image(i,boundary)
            img = Image.fromarray(img)
            img = img.convert("L")
            img.save(f"{location}/image{i}.png")


# p1 = Type1(3, 0.01)
# p1.plot()
# print(p1.object_number)
# p1.save_image_stack("data/curves20")
# print("done")

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