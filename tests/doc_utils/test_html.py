import pytest
import sys
from spb.doc_utils.html import _modify_code, _modify_iplot_code

# NOTE: those functions requires Python >= 3.9 because they use the
# ast module, in particular the unparse function, which is not available
# on previous version.
# However, this is not a problem for final users, as those functions are
# only used to generate the documentation on readthedocs.


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires python3.9 or higher"
)
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


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires python3.9 or higher"
)
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


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires python3.9 or higher"
)
def test_modify_iplot_code_do_not_modify():
    # Many plots will be able to be screenshotted without any modifications
    code = """
from sympy import *
from spb import *
x, a, b, c, n = symbols("x, a, b, c, n")
plot(
   (cos(a * x + b) * exp(-c * x), "oscillator"),
   (exp(-c * x), "upper limit", {"linestyle": ":"}),
   (-exp(-c * x), "lower limit", {"linestyle": ":"}),
   prange(x, 0, n * pi),
   params={
       a: (1, 0, 10),
       b: (0, 0, 2 * pi),
       c: (0.25, 0, 1),
       n: (2, 0, 4)
   },
   ylim=(-1.25, 1.25), imodule="panel")
"""
    s = _modify_iplot_code(code)
    assert "show" not in s
    assert "servable" not in s
    assert "params" in s
    assert "imodule='panel'" in s
    assert "panelplot" not in s


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires python3.9 or higher"
)
def test_modify_iplot_code_KB():
    # Verify that interactive plots with backend=KB returns a panel object
    # containing only widgets
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
    assert "show" in s
    assert "servable" not in s
    assert "params" in s
    assert "imodule='panel'" in s
    assert "panelplot" in s
    assert "layout_controls" in s


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires python3.9 or higher"
)
def test_modify_iplot_code():
    # Verify that if servable=True, then the modified code contains the
    # variable panelplot and that it returns the template in order to
    # construct the webpage for the screenshot
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
    assert "imodule='panel'" in new_code
    assert "template={" in new_code
    assert "panelplot = plot_polar" in s[-2]
    assert "create_template" in s[-1]


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires python3.9 or higher"
)
def test_modify_iplot_code_KB_sums():
    # verify that sums of interactive plots with K3DBackends get preprocessed
    # correctly.

    code = """from sympy import *
from spb import *
import k3d
a, b, s, e, t = symbols("a, b, s, e, t")
c = 2 * sqrt(a * b)
r = a + b
params = {
   a: (1.5, 0, 2),
   b: (1, 0, 2),
   s: (0, 0, 2),
   e: (2, 0, 2)
}
sphere = plot3d_revolution(
   (r * cos(t), r * sin(t)), (t, 0, pi),
   params=params, n=50, parallel_axis="x",
   backend=KB,
   show_curve=False, show=False,imodule="ipywidgets",
   rendering_kw={"color":0x353535})
line = plot3d_parametric_line(
   a * cos(t) + b * cos(3 * t),
   a * sin(t) - b * sin(3 * t),
   c * sin(2 * t), prange(t, s*pi, e*pi),
   {"color_map": k3d.matplotlib_color_maps.Summer}, params=params,
   backend=KB, show=False)
(line + sphere).show()"""

    new_code = _modify_iplot_code(code)
    assert "ipywidgets" not in new_code
    assert "imodule='ipywidgets'" not in new_code
    assert "imodule='panel'" in new_code
    assert "panelplot = line + sphere" in new_code
    assert "layout_controls" in new_code


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires python3.9 or higher"
)
def test_modify_graphics_plotly():
    # verify that doc_utils adds `show=false` and `myplot=` to code block
    # containing `graphics(...)`

    code = """
from sympy import symbols, sin, cos, pi
from spb import *
import numpy as np
x, y = symbols("x, y")
expr = (cos(x) + sin(x) * sin(y) - sin(x) * cos(y))**2
graphics(
    surface(expr, (x, 0, pi), (y, 0, 2 * pi), use_cm=True,
        tx=np.rad2deg, ty=np.rad2deg,
        wireframe=True, wf_n1=20, wf_n2=20),
    backend=PB, xlabel="x [deg]", ylabel="y [deg]",
    aspect=dict(x=1.5, y=1.5, z=0.5))"""

    new_code = _modify_code(code)

    assert "show=False" in new_code
    assert "myplot = graphics" in new_code


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="requires python3.9 or higher"
)
def test_modify_iplot_code_graphics():
    # Verify that if `graphics(..., servable=True), then the modified code
    # contains the variable panelplot and that it returns the template in
    # order to construct the webpage for the screenshot
    code = """
from sympy import *
from spb import *
import param
a, b, c, d, e, f, theta, tp = symbols("a:f theta tp")
def func(n):
    t1 = (c + sin(a * theta + d))
    t2 = ((b + sin(b * theta + e)) - (c + sin(a * theta + d)))
    t3 = (f + sin(a * theta + n / pi))
    return t1 + t2 * t3 / 2
params = {
    a: param.Integer(6, label="a"),
    b: param.Integer(12, label="b"),
    c: param.Integer(18, label="c"),
    d: (4.7, 0, 2*pi),
    e: (1.8, 0, 2*pi),
    f: (3, 0, 5),
    tp: (2, 0, 2)
}
series = []
for n in range(20):
    series += line_polar(
        func(n), prange(theta, 0, tp*pi), params=params,
        rendering_kw={"line_color": "black", "line_width": 0.5})
graphics(
    *series,
    aspect="equal",
    layout = "sbl",
    ncols = 1,
    title="Guilloché Pattern Explorer",
    backend=BB,
    legend=False,
    use_latex=False,
    servable=True,
    imodule="panel"
)
"""
    assert "show" not in code
    assert "servable=True" in code

    new_code = _modify_iplot_code(code)
    s = new_code.split("\n")
    assert "show=False" in new_code
    assert "servable=False" in new_code
    assert "imodule='panel'" in new_code
    assert "template={" in new_code
    assert "panelplot = graphics" in s[-2]
    assert "create_template" in s[-1]


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="requires python3.9 or higher"
)
def test_modify_iplot_code_graphics_series_ui_widgets():
    # Verify that if `graphics(..., app=True), then the modified code
    # contains the variable panelplot and that it returns the template in
    # order to construct the webpage for the screenshot
    code = """
from sympy import *
from spb import *
z = symbols("z")

graphics(
    domain_coloring(sin(z), (z, -2-2j, 2+2j), coloring="b"),
    backend=MB,
    grid=False,
    layout="sbl",
    ncols=1,
    template={"sidebar_width": "30%"},
    app=True,
)
"""
    assert "show" not in code
    assert "servable" not in code

    new_code = _modify_iplot_code(code)
    s = new_code.split("\n")
    assert "show=False" in new_code
    assert "servable" not in new_code
    assert "imodule='panel'" in new_code
    assert "template={" in new_code
    assert "panelplot = graphics" in s[3]
    assert "for k, card in panelplot._additional_widgets.items():" in s[-3]
    assert "card.collapsed = False" in s[-2]
    assert "create_template" in s[-1]
