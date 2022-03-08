import numpy as np
from objects3d import Region, Point
import time
import multiprocessing as mp


PATH = "data/5"
RESOLUTION = 75

VALUE_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION), dtype=int)

SEED_REGIONS_PER_DIM = 5    # devider of resolution
SEED_POINT_NUM = SEED_REGIONS_PER_DIM ** 3
THRESHOLD = 127.5
REGION_MATRIX = np.full((RESOLUTION, RESOLUTION, RESOLUTION), SEED_POINT_NUM)

ASSIGNMENT_COUNT = 0



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
print(f"seeds: {seeds}")


# region growing

regions = [(0,0)] * SEED_POINT_NUM    # tuples of pixel num and average value for each region at its index

n = -1
for seed in seeds:
    VALUE_MATRIX[seed[0]][seed[1]][seed[2]] = n
    n -= 1

t3 = time.time()

def check_pixel(position):
    x = position[0]
    y = position[1]
    z = position[2]
    value = VALUE_MATRIX[x][y][z]
    if value < 0:
        return False
    for n in Point([x,y,z]).get_6connected_nbhood2(RESOLUTION):
        n_val = VALUE_MATRIX[n.value[0]][n.value[1]][n.value[2]]
        if n_val < 0 and abs(value - regions[n_val][1]) < THRESHOLD:
            VALUE_MATRIX[x][y][z] = n_val
            new_count = regions[n_val][0] + 1
            new_avg = regions[n_val][1] * (new_count-1) / new_count + value * 1 / new_count
            regions[n_val] = (new_count, new_avg)
            return True
    return False

pool = mp.Pool(mp.cpu_count())
c = 0
u = 0

changed = True
assignments = 0
while changed:
    changed = False
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            for z in range(RESOLUTION):
                if check_pixel([x,y,z]):
                    changed = True
                    assignments += 1
            #pool.map(check_pixel, [[x,y,z] for z in range(RESOLUTION)])
    print(assignments)

pool.close()


t4 = time.time()

# assign 0 to pixel without region
VALUE_MATRIX[VALUE_MATRIX > 0] = 0

with open(PATH + "/region_matrix", "w") as fp:
    fp.write(str(VALUE_MATRIX.tolist()))

print(f"finding seeds: {t2-t1} regions: {t4-t3}")