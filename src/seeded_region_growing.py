import numpy as np
from objects3d import Region, Point
import time


PATH = "data/7"
RESOLUTION = 25

VALUE_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))

SEED_REGIONS_PER_DIM = 4
SEED_POINT_NUM = SEED_REGIONS_PER_DIM ** 3
THRESHOLD = 127.5
REGION_MATRIX = np.full((RESOLUTION, RESOLUTION, RESOLUTION), SEED_POINT_NUM)



# read matrix from file

file = open(PATH+"/value_matrix", "r")
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
            length = int(RESOLUTION/SEED_REGIONS_PER_DIM)
            seed = (x_factor*length, y_factor*length, z_factor*length)
            max = 0
            for x in range(x_factor*length, (x_factor+1)*length):
                for y in range(y_factor*length, (y_factor+1)*length):
                    for z in range(z_factor*length, (z_factor+1)*length):
                        val = VALUE_MATRIX[x][y][z]
                        if val > max:
                            max = val
                            seed = (x,y,z)
            seeds.append(seed)
"""

seeds = list()
while len(seeds)<SEED_POINT_NUM:
    s = (random.choice(range(RESOLUTION)), random.choice(range(RESOLUTION)), random.choice(range(RESOLUTION)))
    if s not in seeds:
        seeds.append(s)
"""

t2 = time.time()
print(seeds)

# region growing

regions = list()
n = 0
for seed in seeds:
    regions.append(Region(n, Point(list(seed)), VALUE_MATRIX[seed[0]][seed[1]][seed[2]]))
    REGION_MATRIX[seed[0]][seed[1]][seed[2]] = n
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

        print(f"{len(r.pixel)} in {r.num}")
        new_pixel = set()
        new_intensities = list()

        for p in r.get_new_neighbors(RESOLUTION):

            x = p.value[0]
            y = p.value[1]
            z = p.value[2]

            # pixel already belongs to another region
            if REGION_MATRIX[x][y][z] != SEED_POINT_NUM:
                continue

            intensity = VALUE_MATRIX[x][y][z]

            if abs(intensity - r.medium_intensity) < THRESHOLD and p not in new_pixel:
                new_pixel.add(p)
                new_intensities.append(intensity)
                REGION_MATRIX[x][y][z] = r.num

        if new_pixel:
            new_medium = sum(new_intensities) / len(new_pixel)
            r.expand(new_pixel, new_medium)
            changed = True
        else:
            not_growing_regions.append(r.num)

    for i in not_growing_regions:
        regions[i] = None



final_regions = dict()      # key: r num, value: r
for x in range(RESOLUTION):
    for y in range(RESOLUTION):
        for z in range(RESOLUTION):
            r = REGION_MATRIX[x][y][z]
            intensity = VALUE_MATRIX[x][y][z]
            if r not in final_regions.keys():
                final_regions[r] = Region(r, Point([x,y,z]), intensity)
            else:
                final_regions[r].expand({Point([x,y,z])}, intensity)


# merge neighbor regions
final_regions = list(final_regions.values())
changed = True
merged = set()
while changed:
    changed = False
    for i1 in range(len(final_regions)):
        if i1 in merged:
            continue
        r1 = final_regions[i1]
        for i2 in range(i1+1, len(final_regions)):
            if i2 in merged:
                continue
            r2 = final_regions[i2]
            if r1.is_neighbor(r2, RESOLUTION) and abs(r1.medium_intensity - r2.medium_intensity) < THRESHOLD:
                final_regions[i1].merge(final_regions[i2])
                merged.add(i2)
                changed = True
                print(f"merged {i2} in {i1}")

print(f"regions before merging: {len(final_regions)}")
final_regions = [final_regions[i] for i in range(len(final_regions)) if i not in merged]
print(f"regions after merging: {len(final_regions)}")

REGION_MATRIX = np.full((RESOLUTION, RESOLUTION, RESOLUTION), SEED_POINT_NUM)
for r in final_regions:
    for p in r.pixel:
        REGION_MATRIX[p.value[0]][p.value[1]][p.value[2]] = r.num


t4 = time.time()

with open(PATH + "/region_matrix", "w") as fp:
    fp.write(str(str(REGION_MATRIX.tolist())))

print(f"finding seeds: {t2-t1} regions: {t4-t3}")