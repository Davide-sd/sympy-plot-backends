from spb.doc_utils import _modify_code, _modify_iplot_code

def test_modify_code_1():
    code = """
from sympy import symbols, sin, cos, pi
from spb import plot3d, PB
import numpy as np
x, y = symbols("x, y")
expr = (cos(x) + sin(x) * sin(y) - sin(x) * cos(y))**2
plot3d(
    expr, (x, 0, pi), (y, 0, 2 * pi), backend=PB, use_cm=True,
    tx=np.rad2deg, ty=np.rad2deg, wireframe=True, wf_n1=20, wf_n2=20,
    xlabel="x [deg]", ylabel="y [deg]",
    aspect=dict(x=1.5, y=1.5, z=0.5))
"""
    s = _modify_code(code).split("\n")
    assert "myplot.fig" in s[-1]
    assert "myplot = plot3d" in s[-2]


def test_modify_code_2():
    code = """
from sympy import *
from spb import *
a, b = S(3) / 2, 1
t = symbols("t")
c = 2 * sqrt(a * b)
r = a + b
sphere = plot3d_revolution(
    (r * cos(t), r * sin(t)), (t, 0, pi),
    n=50, parallel_axis="x",
    backend=PB,
    show_curve=False, show=False,
    rendering_kw={})
line = plot3d_parametric_line(
    a * cos(t) + b * cos(3 * t),
    a * sin(t) - b * sin(3 * t),
    c * sin(2 * t), (t, 0, 2*pi),
    {},
    backend=PB, show=False)
(line + sphere).show()
"""
    new_code = _modify_code(code)
    s = new_code.split("\n")
    assert ").show()" not in new_code
    assert s[-2] == "myplot = line + sphere"
    assert s[-1] == "myplot.fig"


def test_modify_iplot_code_do_not_modify():
    code = """
from sympy import *
from spb import *
import param
n, u, v = symbols("n, u, v")
x = v * cos(u)
y = v * sin(u)
z = sin(n * u)
plot3d_parametric_surface(
    (x, y, z, (u, 0, 2*pi), (v, -1, 0)),
    params = {
        n: param.Integer(2, label="n")
    },
    backend=KB,
    use_cm=True,
    title=r"Plücker's \, conoid",
    wireframe=True,
    wf_rendering_kw={"width": 0.004},
    wf_n1=75, wf_n2=6)
"""
    s = _modify_iplot_code(code)
    assert "show" not in s
    assert "servable" not in s
    assert "params" in s


def test_modify_iplot_code():
    code = """
from sympy import *
from spb import *
import param
from bokeh.models.formatters import PrintfTickFormatter
formatter = PrintfTickFormatter(format='%.4f')

p1, p2, t, r, c = symbols("p1, p2, t, r, c")
phi = - (r * t + p1 * sin(c * r * t) + p2 * sin(2 * c * r * t))
phip = phi.diff(t)
r1 = phip / (1 + phip)

plot_polar(
    (r1, (t, 0, 2*pi)),
    params = {
        p1: (0.035, -0.035, 0.035, 50, formatter),
        p2: (0.005, -0.02, 0.02, 50, formatter),
        # integer parameter created with param
        r: param.Integer(2, softbounds=(2, 5), label="r"),
        # integer parameter created with usual syntax
        c: (3, 1, 5, 4)
    },
    use_latex = False,
    backend = BB,
    aspect = "equal",
    n = 5000,
    layout = "sbl",
    ncols = 1,
    servable = True,
    name = "Non Circular Planetary Drive - Ring Profile")
"""
    assert "show" not in code
    assert "servable = True" in code

    new_code = _modify_iplot_code(code)
    s = new_code.split("\n")
    assert "show=False" in new_code
    assert "servable=False" in new_code
    assert "panelplot = plot_polar" in s[-2]
    assert "create_template" in s[-1]
