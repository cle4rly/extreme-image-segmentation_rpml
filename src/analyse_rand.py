import math
import time
import multiprocessing as mp

from matrix_parser import MatrixParser


PATH = "data/8"
TRUTH = PATH + "/point_matrix"
RESULT = PATH + "/region_matrix2"
POINTS = PATH + "/points"

RESOLUTION = 100
PAIR_NUM = math.comb(RESOLUTION**3, 2)

indices = [(x,y,z) for x in range(RESOLUTION) for y in range(RESOLUTION) for z in range(RESOLUTION)]

truth_matrix = MatrixParser.read_from_file(RESOLUTION, TRUTH)
result_matrix = MatrixParser.read_from_file(RESOLUTION, RESULT)
points = MatrixParser.read_points(RESOLUTION, POINTS)


def compare(i):
    i1 = i[0]
    i2 = i[1]
    tp = tn = fp = fn = 0

    same_result = result_matrix[i1] == result_matrix[i2]
    same_truth = truth_matrix[i1] == truth_matrix[i2]

    if same_result and same_truth:
        tp += 1
    elif not same_result and not same_truth:
        tn += 1
    elif same_result and not same_truth:
        fp += 1
    elif not same_result and same_truth:
        fn += 1
    
    return (tp, tn, fp, fn)

def compare_curve_points(i):
    return compare((points[i[0]], points[i[1]]))

def compare_points(i):
    return compare((indices[i[0]], indices[i[1]]))


def comp_all_points():
    tp = tn = fp = fn = 0

    t1 = time.time()
    pool = mp.Pool(mp.cpu_count())

    c=0
    for i1 in range(RESOLUTION**3):
        res = list()
        c +=1
        print(c/RESOLUTION**3)
        array = [(i1,i2) for i2 in range(i1+1, RESOLUTION**3)]
        res.append(pool.map(compare_points, array))

        for re in res:
            for r in re:
                tp += r[0]
                tn += r[1]
                fp += r[2]
                fn += r[3]
        
    pool.close()
    t2 = time.time()
    print(t2-t1)
    return (tp, tn, fp, fn)

def comp_curve_points():
    tp = tn = fp = fn = 0

    t1 = time.time()
    pool = mp.Pool(mp.cpu_count())
    l = 0
    for i1 in range(len(points)):
        res = list()
        array = [(i1, i2) for i2 in range(i1+1, len(points))]
        res.append(pool.map(compare_curve_points, array))
        l += len(array)

        for re in res:
            for r in re:
                tp += r[0]
                tn += r[1]
                fp += r[2]
                fn += r[3]

    pool.close()
    t2 = time.time()
    print(t2-t1)
    return (tp, tn, fp, fn)


print(PAIR_NUM)
res = comp_curve_points()
res = (str(res[0]), str(res[1]), str(res[2]), str(res[3]))
print()
print("\n".join(res))
