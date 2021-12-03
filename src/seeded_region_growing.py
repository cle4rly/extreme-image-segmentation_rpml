import random
from curve import Point
from problems import ImageObj, Type1, Image


class Region:

    def __init__(self, seed_point, seed_intensity) -> None:
        self.medium_intensity = seed_intensity
        self.inner_pixel = set()
        self.inner_border_pixel = set()
        self.border_pixel = {seed_point}
    
    def get_new_neighbors(self, resolution):
        neighbors = set()
        for p in self.border_pixel:
            for n in p.get_6connected_nbhood(resolution):
                if n not in self.border_pixel and n not in self.inner_border_pixel:
                    neighbors.add(n)
        return neighbors

    def expand(self, new_border_pixel, new_intensity):
        self.inner_pixel.add(self.inner_border_pixel)
        self.inner_border_pixel.add(self.border_pixel)
        self.border_pixel = new_border_pixel

        # calc new medium
        n_old = len(self.inner_pixel) + len(self.inner_border_pixel) + len(self.border_pixel)
        n_new = len(new_border_pixel)
        n = n_old + n_new
        self.medium_intensity = self.medium_intensity * n_old/n + new_intensity * n_new/n


class SeededRegionGrowing:

    def __init__(self, img, seed_points_num=100, intensity_threshold=20) -> None:
        self.image: ImageObj = img
        self.seed_points_num = seed_points_num
        self.intensity_threshold = intensity_threshold
        self.seed_points = self.get_seed_points()
        print(self.seed_points)
        self.regions = [Region(sp, self.image.get_intensity(sp)) for sp in self.seed_points]

    # TODO: better method (e.g. max per cube)
    def get_seed_points(self):
        seeds = list()
        for _ in range(self.seed_points_num):
            x = random.randint(0, self.image.resolution-1)
            y = random.randint(0, self.image.resolution-1)
            rand_row = self.image.values[x][y]
            z = rand_row.index(max(rand_row))
            seeds.append(self.image.get_intensity([x,y,z]))
        return seeds

    def expand_regions(self, region_index):
        changed = True
        while changed:
            changed = False
            for r in self.regions[region_index]:
                new_pixel = set()
                new_intensities = list()
                for p in r.get_new_neighbors(self.img.resolution):
                    intensity = self.image.get_intensity(p)
                    if abs(intensity - r.medium_intensity) < self.intensity_threshold:
                        new_pixel.add(p)
                        new_intensities.append(intensity)
                if new_pixel:
                    new_medium = sum(new_pixel) / len(new_pixel)
                    r.expand(new_pixel, new_medium)
                    changed = True
            print(f"r: {self.regions}")


p1 = Type1(5, 0.005)
img = p1.get_image_object()
srg = SeededRegionGrowing(img)
print(srg.regions)      


# resolution aus pixel raus (liber nur index)
# regions mergen!!!