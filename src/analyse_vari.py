import math

from matrix_parser import MatrixParser


PATH = "data/7"
TRUTH = PATH + "/point_matrix"
RESULT = PATH + "/random_matrix"

RESOLUTION = 25
PAIR_NUM = math.comb(RESOLUTION**3, 2)

indices = [(x,y,z) for x in range(RESOLUTION) for y in range(RESOLUTION) for z in range(RESOLUTION)]

truth_matrix = MatrixParser.read_from_file(RESOLUTION, TRUTH)
result_matrix = MatrixParser.read_from_file(RESOLUTION, RESULT)

n = RESOLUTION**3
xs = dict()
ys = dict()

def fill_dict(dict, matrix):
    for i in indices:
        key = matrix[i]
        if key not in dict.keys():
            dict[key] = list()
        dict[key].append(i)

fill_dict(xs, truth_matrix)
fill_dict(ys, result_matrix)

variation = 0

for i in xs.keys():
    p = len(xs[i]) / n
    for j in ys.keys():
        q = len(ys[j]) / n
        r = len(set(xs[i]) & set(ys[j])) / n
        if r != 0:
            variation += r * (math.log(round(r/p, 6), 2) + math.log(round(r/q, 6), 2))

print(variation * (-1))