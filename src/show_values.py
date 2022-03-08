import numpy as np
from PIL import Image


PATH = "data/5"
RESOLUTION = 75

VALUE_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION), dtype=int)




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






for i in range(RESOLUTION):
    img = Image.fromarray(np.uint8(VALUE_MATRIX[i]), 'L')
    img.save(f"{PATH}/vals/image{i}.png")