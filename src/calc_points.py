from PIL import Image
import numpy as np

from matrix_parser import MatrixParser


PATH = "data/9"
RESOLUTION = 200


value_matrix = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION), dtype=int)
curves_points = MatrixParser.read_curves(RESOLUTION, PATH + "/points")


color = 255
for curve in curves_points:
    for point in curve:
        x = int(round(point[0]*RESOLUTION))
        y = int(round(point[1]*RESOLUTION))
        z = int(round(point[2]*RESOLUTION))
        value_matrix[x, y, z] = color
    color -= 10


# save images

for i in range(RESOLUTION):
    img = Image.fromarray(np.uint8(value_matrix[i]), 'L')
    img.save(PATH + f"/point_imgs/image{i}.png")

# save value matrix

file = open(PATH + "/point_matrix", 'w')
file.write(str(value_matrix.tolist()))
file.close()