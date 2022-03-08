import numpy as np
from PIL import Image



PATH = "data/7"
WHITE = 255
GRAY = 200
RESOLUTION = 25
REGIONS_PER_DIM = 4
REGION_NUM = REGIONS_PER_DIM ** 3
REGION_STEPS = int(round(WHITE/REGIONS_PER_DIM))

REGION_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))
REGION_MATRIX = REGION_MATRIX.tolist()

VALUE_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION, 3))

REGION_COLORS = np.zeros((REGION_NUM+1,3))
i = 0
for r in range(0, WHITE, REGION_STEPS):
    for g in range(0, WHITE, REGION_STEPS):
        for b in range(0, WHITE, REGION_STEPS):
            print(f"{r} {g} {b}")
            REGION_COLORS[i][0] = r
            REGION_COLORS[i][1] = g
            REGION_COLORS[i][2] = b
            i += 1
REGION_COLORS[REGION_NUM][0] = WHITE
REGION_COLORS[REGION_NUM][1] = WHITE
REGION_COLORS[REGION_NUM][2] = WHITE

#np.random.shuffle(REGION_COLORS)

black_index = 0
for i in range(len(REGION_COLORS)):
    if REGION_COLORS[i][0] == 0 and REGION_COLORS[i][1] == 0 and REGION_COLORS[i][2] == 0:
        black_index = i

REGION_COLORS[[-1,black_index]] = REGION_COLORS[[black_index,-1]] # always use black for 0 values
REGION_COLORS[0] = [GRAY, GRAY, GRAY]

print(REGION_COLORS)


# read region matrix from file

file = open(PATH + "/region_matrix", "r")
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
            REGION_MATRIX[x][y][z] = int(row[z])
            z += 1
        y += 1


# generate images

for x in range(RESOLUTION):
    print(x)
    for y in range(RESOLUTION):
        for z in range(RESOLUTION):
            VALUE_MATRIX[x][y][z] = REGION_COLORS[int(REGION_MATRIX[x][y][z])]
    img = Image.fromarray(np.array(VALUE_MATRIX[x]).astype('uint8'), "RGB")
    img.save(f"{PATH}/regions3/image{x}.png")

