from typing import Callable

from matplotlib import pyplot as plt


def knot_function(i: int, k: int, knots: list[float]) -> Callable:
    if k == 0:
        return lambda t: int(knots[i] <= t < knots[i + 1])

    return lambda t: \
        (t - knots[i]) / (knots[i + k] - knots[i]) \
        * knot_function(i, k - 1, knots)(t) \
        + (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) \
        * knot_function(i + 1, k - 1, knots)(t)  # fmt: skip


t = 3.23

knot_vector = [*range(7)]
k = 2
c = [5, -5, 3, 2]


bases = [knot_function(n, k, knot_vector) for n in range(len(knot_vector) - k - 1)]
x = [i / 200 for i in range(1200)] 
y = [sum(i * j(t) for i, j in zip(c, bases)) for t in x] + c
x += [len(knot_vector) / len(c) * i for i in range(len(c))]

plt.scatter(x, y)
plt.show()
