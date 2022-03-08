import math

from matrix_parser import MatrixParser


PATH = "data/8"
RESULT = PATH + "/region_matrix2"
    
RESOLUTION = 100
PAIR_NUM = math.comb(RESOLUTION**3, 2)

REGIONS_PER_DIM = 4
REGION_NUM = REGIONS_PER_DIM ** 3

indices = [(x,y,z) for x in range(RESOLUTION) for y in range(RESOLUTION) for z in range(RESOLUTION)]

result_matrix = MatrixParser.read_from_file(RESOLUTION, RESULT)

unclassified = 0
for i in indices:
    if result_matrix[i] == REGION_NUM:
        unclassified += 1

print(unclassified)

