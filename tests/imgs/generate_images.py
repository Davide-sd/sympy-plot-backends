"""
Each release of Matplotlib is inevitably going to change something, like the
ticks style on the axis, the grid styling...

Also inevitably, as the module progresses in development, tests might requires
new Matplotlib features. Upgrading the dependency is not enough to make tests
pass, because of the style changes. For example, ``plot_implicit`` is going
to compare its results with pre-generated outputs, which must be regenerated
everytime an updated version of Matplotlib is required.

This is a critical step: the following script creates the outputs, but they
must be manually checked with what they are going to replace in order to
assure that the outputs are correct!!!!
"""

from sympy import *
from spb import *

def save_plot(expr, range_x, range_y, filename, size):
    p = plot_implicit(expr, range_x, range_y,
        size=size, adaptive=True, grid=False, show=False,
        use_latex=False, backend=MB)
    p.save(filename)

var("x, y")
r1 = (x - 1) ** 2 + y ** 2 < 2
r2 = (x + 1) ** 2 + y ** 2 < 2
save_plot(r1 & r2, (x, -5, 5), (y, -5, 5), "test_region_and.png", (8, 6))
save_plot(r1 | r2, (x, -5, 5), (y, -5, 5), "test_region_or.png", (8, 6))
save_plot(~r1, (x, -5, 5), (y, -5, 5), "test_region_not.png", (8, 6))
save_plot(r1 ^ r2, (x, -5, 5), (y, -5, 5), "test_region_xor.png", (8, 6))
        
save_plot(Eq(y, cos(x)), (x, -5, 5), (y, -5, 5), "pi_01.png", (5, 4))
save_plot(Eq(y ** 2, x ** 3 - x), (x, -5, 5), (y, -5, 5), "pi_02.png", (5, 4))
save_plot(y > 1 / x, (x, -5, 5), (y, -5, 5), "pi_03.png", (5, 4))
save_plot(y < 1 / tan(x), (x, -5, 5), (y, -5, 5), "pi_04.png", (5, 4))
save_plot(y >= 2 * sin(x) * cos(x), (x, -5, 5), (y, -5, 5), "pi_05.png", (5, 4))
save_plot(y <= x ** 2, (x, -5, 5), (y, -5, 5), "pi_06.png", (5, 4))
save_plot(y > x, (x, -5, 5), (y, -5, 5), "pi_07.png", (5, 4))
save_plot(And(y > exp(x), y > x + 2), (x, -5, 5), (y, -5, 5), "pi_08.png", (5, 4))
save_plot(Or(y > x, y > -x), (x, -5, 5), (y, -5, 5), "pi_09.png", (5, 4))
save_plot(x ** 2 - 1, (x, -5, 5), (y, -5, 5), "pi_10.png", (5, 4))
save_plot(y > cos(x), (x, -5, 5), (y, -5, 5), "pi_11.png", (5, 4))
save_plot(y < cos(x), (x, -5, 5), (y, -5, 5), "pi_12.png", (5, 4))
save_plot(And(y > cos(x), Or(y > x, Eq(y, x))), (x, -5, 5), (y, -5, 5), "pi_13.png", (5, 4))
save_plot(y - cos(pi / x), (x, -5, 5), (y, -5, 5), "pi_14.png", (5, 4))
# NOTE: this should fallback to adaptive=False
save_plot(Eq(y, re(cos(x) + I * sin(x))), (x, -5, 5), (y, -5, 5), "pi_15.png", (5, 4))
