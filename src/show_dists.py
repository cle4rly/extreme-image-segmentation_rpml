import numpy as np
from PIL import Image
from matrix_parser import MatrixParser


PATH = "data/9"
RESOLUTION = 200


matrix = MatrixParser.read_from_file(RESOLUTION, PATH + "/dist_matrix", dtype=float)
matrix = 255 - (matrix * 255) / np.amax(matrix)

for i in range(RESOLUTION):
    img = Image.fromarray(np.uint8(matrix[i]), 'L')
    img.save(f"{PATH}/dists/image{i}.png")