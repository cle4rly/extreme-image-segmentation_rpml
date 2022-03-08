import numpy as np


class MatrixParser:

    def __init__(self) -> None:
        pass

    def read_from_file(resolution, file_name, dtype=int):
        matrix = np.zeros((resolution, resolution, resolution), dtype)
        file = open(file_name, "r")
        input = file.read()
        slices = input.split("]], [[")
        slices[0] = slices[0][3:]
        slices[-1] = slices[-1][:-3]
        for x in range(resolution):
            rows = slices[x].split("], [")
            for y in range(resolution):
                row = rows[y]
                row = row.split(",")
                for z in range(resolution):
                    matrix[x][y][z] = dtype(row[z])
                    z += 1
                y += 1
        return matrix

    def read_curves(resolution, file_name):
        array = list()
        file = open(file_name, "r")
        input = file.read()
        curves = input.split("\n\n")[:-1]
        for curve in curves:
            curve = curve[2:-2]
            curve = curve.split("], [")
            curve_points = list()
            for p in curve:
                p = p.split(",")
                curve_points.append((float(p[0]), float(p[1]), float(p[2])))
            array.append(curve_points)
        return array

    def read_points(resolution, file_name):
        curve_points = list()
        file = open(file_name, "r")
        input = file.read()
        curves = input.split("\n\n")[:-1]
        for curve in curves:
            curve = curve[2:-2]
            curve = curve.split("], [")
            for p in curve:
                p = p.split(",")
                curve_points.append((round(float(p[0])*resolution), round(float(p[1])*resolution), round(float(p[2])*resolution)))
        return curve_points

    def save_to_file(matrix):
        pass