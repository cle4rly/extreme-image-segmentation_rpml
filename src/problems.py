
import math
import numpy as np
from curve import Curve
import matplotlib.pyplot as plt


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

        c=0
        while len(self.curves) < self.object_number:
            print(f"{c} {len(self.curves)}")
            c+=1
            new_curve = Curve()
            if self.validate(new_curve):
                self.curves.append(new_curve)

    def too_close(self, curve):
        for c in self.curves:
            for p1 in c.values:
                for p2 in curve.values:
                    if np.linalg.norm(np.array(p1) - np.array(p2)) < self.distance:
                        return True
        return False
    
    def validate(self, curve) -> bool:
        if curve.too_short() or self.too_close(curve):
            return False
        return True

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for c in self.curves:
            xs = [p[0] for p in c.values]
            ys = [p[1] for p in c.values]
            zs = [p[2] for p in c.values]
            ax.plot3D(xs, ys, zs)

        ax.set_title('type 1')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()


class Type2(Problem):

    def __init__(self, object_number):
        super().__init__(object_number)




p1 = Type1(15, 0.05)
p1.plot()
print(p1.object_number)
