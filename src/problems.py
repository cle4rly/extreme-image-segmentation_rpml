
from _typeshed import Self

import math


class Problem:

    def __init__(self, object_number):
        self.object_number = object_number

    def initialize(self):
        pass

    def plot(self):
        pass


class Type1(Problem):

    def __init__(self, object_number, distance):
        super().__init__(object_number)
        self.distance = distance
        self.curves = list()

    @staticmethod
    def point_distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def too_close(self, curve):
        for c in self.curves:
            for p1 in c.get_values():
                for p2 in curve.get_values():
                    if Type1.point_distance(p1, p2) < self.distance:
                        return True
        return False
        