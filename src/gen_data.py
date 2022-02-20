import os
from PIL import Image
import numpy as np
import random
import json


PATH = "data/output-total/curves4"
RESOLUTION = 500
PROB_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))
VALUE_MATRIX = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))
VALUE_MATRIX = VALUE_MATRIX.tolist()


def save_image(location, num):
    img = Image.fromarray(np.uint8(VALUE_MATRIX[num]), 'L')
    img.save(f"{location}/image{num}.png")


print("reading dists from file")

max = 0
for x in range(RESOLUTION):
    print(x)
    file = open(PATH+"/dists"+str(x), "r")
    input = file.read()
    rows = input.split("], [")
    rows[0] = rows[0][2:]
    rows[-1] = rows[-1][:-2]
    for y in range(RESOLUTION):
        row = rows[y]
        row = row.split(",")
        for z in range(RESOLUTION):
            d = float(row[z])
            PROB_MATRIX[x][y][z] = d
            if d > max:
                max = d
            z += 1
        y += 1

print(f"max = {max}")


print("dists to probabilities")
# (1 - curve, 0 - max dist)

for x in range(RESOLUTION):
    print(x)
    for y in range(RESOLUTION):
        for z in range(RESOLUTION):
            PROB_MATRIX[x][y][z] = 1 - PROB_MATRIX[x][y][z]/max


# gen both noises

SIGMA = 30
WHITE = 255
MEAN1 = WHITE*3/10
MEAN2 = WHITE*7/10

err = 0

noise1 = list(np.random.normal(MEAN1,SIGMA,10000))
for i in range(len(noise1)):
    noise1[i] = round(noise1[i])
    if noise1[i] < 0:
        noise1[i] = 0
        err += 1
    if noise1[i] > WHITE:
        noise1[i] = WHITE
        err += 1

noise2 = list(np.random.normal(MEAN2,SIGMA,10000))
for i in range(len(noise2)):
    noise2[i] = round(noise2[i])
    if noise2[i] < 0:
        noise2[i] = 0
        err += 1
    if noise2[i] > WHITE:
        noise2[i] = WHITE
        err += 1

print(err)
print("choose values and save files")

for x in range(RESOLUTION):
    print(x)
    for y in range(RESOLUTION):
        for z in range(RESOLUTION):
            noises = [1,2]
            p = PROB_MATRIX[x][y][z]
            probs = [1-p, p]
            n = random.choices(noises, probs)
            if n == [1]:
                VALUE_MATRIX[x][y][z] = int(random.choice(noise1))
            else:
                VALUE_MATRIX[x][y][z] = int(random.choice(noise2))
    #save_image("data/test", x)



with open("data/value_matrix500", "w") as fp:
    fp.write(str(VALUE_MATRIX))
