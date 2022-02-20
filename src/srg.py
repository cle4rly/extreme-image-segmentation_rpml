from os import times
import numpy as np
from objects3d import Region, Point
import time


PATH = "data"
RESOLUTION = 500

VALUE_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))

SEED_REGIONS_PER_DIM = 5    # devider of resolution
SEED_POINT_NUM = SEED_REGIONS_PER_DIM ** 3
THRESHOLD = 60
REGION_MATRIX = np.full((RESOLUTION, RESOLUTION, RESOLUTION), SEED_POINT_NUM)



# read matrix from file

file = open(PATH+"/value_matrix500", "r")
input = file.read()
slices = input.split("]], [[")
slices[0] = slices[0][3:]
slices[-1] = slices[-1][:-3]
for x in range(RESOLUTION):
    print(x)
    rows = slices[x].split("], [")
    for y in range(RESOLUTION):
        row = rows[y]
        row = row.split(",")
        for z in range(RESOLUTION):
            VALUE_MATRIX[x][y][z] = int(row[z])
            z += 1
        y += 1


t1 = time.time()

# find seed points

seeds = list()
for x_factor in range(SEED_REGIONS_PER_DIM):
    for y_factor in range(SEED_REGIONS_PER_DIM):
        for z_factor in range(SEED_REGIONS_PER_DIM):
            max = 0
            seed = (0,0,0)
            length = int(RESOLUTION/SEED_REGIONS_PER_DIM)
            for x in range(x_factor*length, (x_factor+1)*length):
                for y in range(y_factor*length, (y_factor+1)*length):
                    for z in range(z_factor*length, (z_factor+1)*length):
                        val = VALUE_MATRIX[x][y][z]
                        if val > max:
                            max = val
                            seed = (x,y,z)
            seeds.append(seed)

t2 = time.time()
print(seeds)


# region growing

regions = list()
stashed_regions = list()
n = 0
for seed in seeds:
    regions.append(Region(n, Point(list(seed)), VALUE_MATRIX[seed[0]][seed[1]][seed[2]]))
    REGION_MATRIX[seed[0]][seed[1]][seed[2]] = n
    stashed_regions.append(None)
    n += 1

t3 = time.time()

changed = True
while changed:
    changed = False
    not_growing_regions = list()
    print(f"regions: {len([r for r in regions if r])}")
    for r in regions:
        if not r:
            continue

        print(f"{len(r.border_pixel)} in {r.num}")
        new_pixel = set()
        new_intensities = list()

        merged = False
        for p in r.get_new_neighbors(RESOLUTION):
            if merged:
                break

            x = p.value[0]
            y = p.value[1]
            z = p.value[2]

            if REGION_MATRIX[x][y][z] == r.num:
                continue

            intensity = VALUE_MATRIX[x][y][z]

            if abs(intensity - r.medium_intensity) < THRESHOLD:
                new_pixel.add(p)
                old_r_num = REGION_MATRIX[x][y][z]
                if old_r_num != SEED_POINT_NUM and old_r_num > r.num:
                    #print(f"merge {r.num} with {old_r_num}")
                    old_r = regions[old_r_num] if regions[old_r_num] else stashed_regions[old_r_num]
                    old_r.merge(r)
                    not_growing_regions.append(r.num)
                    merged = True
                REGION_MATRIX[x][y][z] = r.num
                new_intensities.append(intensity)

        if new_pixel:
            new_medium = sum(new_intensities) / len(new_pixel)
            r.expand(new_pixel, new_medium)
            changed = True
        else:
            not_growing_regions.append(r.num)

    for i in not_growing_regions:
        stashed_regions[i] = regions[i].copy()
        regions[i] = None

t4 = time.time()

with open("data/region_matrix500", "w") as fp:
    fp.write(str(REGION_MATRIX))

print(f"finding seeds: {t2-t1} regions: {t4-t3}")
