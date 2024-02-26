import pytest
from spb.graphics import (
    vector_field_2d, vector_field_3d, surface_parametric, arrow_2d
)
from spb.series import (
    ContourSeries, Vector2DSeries, Vector3DSeries, SliceVector3DSeries,
    Arrow2DSeries
)
from sympy import (
    symbols, cos, sin, pi, sqrt, Matrix, tan, Tuple
)
from sympy.abc import j, k, l
from sympy.vector import CoordSys3D, gradient
from sympy.physics.mechanics import ReferenceFrame

x, y, z, p1, p2 = symbols("x y z p1 p2")
C = CoordSys3D("C")
v1 = sin(x - y) * C.i + cos(x + y) * C.j
v2 = [sin(x - y), cos(x + y)]
v3 = Matrix(v2)
R = ReferenceFrame("R")
v4 = sin(x - y) * R.x + cos(x + y) * R.y

v5 = z * C.i + y * C.j + x * C.k
v6 = Matrix([z, y, x])
v7 = Matrix(v6)
v8 = z * R.x + y * R.y + x * R.z

p1, p2 = symbols("p1 p2")


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.parametrize(
    "e1, e2, r1, r2, label, quiver_kw, contour_kw, stream_kw, streamlines, scalar, params",
    [
        (sin(x - y), cos(x + y), None, None, None, None, None, None, False, False, None),
        (sin(x - y), cos(x + y), (x, -2, 2), None, None, None, None, None, False, False, None),
        (sin(x - y), cos(x + y), None, (y, -3, 3), None, None, None, None, False, False, None),
        (sin(x - y), cos(x + y), (x, -2, 2), (y, -3, 3), None, None, None, None, False, False, None),
        (sin(x - y), cos(x + y), (x, -2, 2), (y, -3, 3), "test", None, None, None, False, False, None),
        (v1, None, (x, -2, 2), (y, -3, 3), "test", None, None, None, False, False, None),
        (v2, None, (x, -2, 2), (y, -3, 3), "test", None, None, None, False, False, None),
        (v3, None, (x, -2, 2), (y, -3, 3), "test", None, None, None, False, False, None),
        (v4, None, (x, -2, 2), (y, -3, 3), "test", None, None, None, False, False, None),
        (sin(x - y), cos(x + y), (x, -2, 2), (y, -3, 3), "test", {"color": "r"}, {"linewidth": 0.5}, {"cmap": "Blues_r"}, False, False, None),
        (sin(x - y), cos(x + y), (x, -2, 2), (y, -3, 3), "test", {"color": "r"}, {"linewidth": 0.5}, {"cmap": "Blues_r"}, True, False, None),
        (sin(x - y), cos(x + y), (x, -2, 2), (y, -3, 3), "test", {"color": "r"}, {"linewidth": 0.5}, {"cmap": "Blues_r"}, False, True, None),
        (sin(x - y), cos(x + y), (x, -2, 2), (y, -3, 3), "test", {"color": "r"}, {"linewidth": 0.5}, {"cmap": "Blues_r"}, True, True, None),
        (sin(x - y), cos(x + y), (x, -2, 2), (y, -3, 3), "test", {"color": "r"}, {"linewidth": 0.5}, {"cmap": "Blues_r"}, False, False, {p1: (1, 0, 2)}),
])
def test_vector_field_2d(
    default_range, e1, e2, r1, r2, label, quiver_kw,
    contour_kw, stream_kw, streamlines, scalar, params
):
    kwargs = {}
    if params:
        kwargs["params"] = params
    if quiver_kw:
        kwargs["quiver_kw"] = quiver_kw
    if stream_kw:
        kwargs["stream_kw"] = quiver_kw
    if contour_kw:
        kwargs["contour_kw"] = contour_kw
    print(e1, e2)
    series = vector_field_2d(
        e1, e2, range1=r1, range2=r2,
        scalar=scalar, streamlines=streamlines, **kwargs)
    assert len(series) == 1 if not scalar else 2
    if len(series) == 1:
        vec_series = series[0]
    else:
        con_series, vec_series = series
    assert isinstance(vec_series, Vector2DSeries)
    r1c = default_range(x) if not r1 else r1
    r2c = default_range(y) if not r2 else r2
    if (not r1) and (not r2):
        assert (vec_series.ranges[0] == r1c) or (vec_series.ranges[0] == r2c)
        assert (vec_series.ranges[1] == r1c) or (vec_series.ranges[1] == r2c)
    else:
        assert vec_series.ranges[0] == r1c
        assert vec_series.ranges[1] == r2c
    if streamlines:
        assert vec_series.rendering_kw == {} if not stream_kw else stream_kw
    else:
        assert vec_series.rendering_kw == {} if not quiver_kw else quiver_kw
    assert vec_series.is_interactive == (len(vec_series.params) > 0)
    assert vec_series.params == {} if not params else params
    if len(series) > 1:
        assert isinstance(con_series, ContourSeries)
        assert con_series.expr == sqrt(sum(t**2 for t in vec_series.expr))
        assert con_series.get_label(False) == "Magnitude"
        assert con_series.rendering_kw == {} if not contour_kw else contour_kw
        assert con_series.is_interactive == (len(vec_series.params) > 0)
        assert con_series.params == {} if not params else params


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.filterwarnings("ignore:Not enough ranges were provided.")
@pytest.mark.parametrize(
    "e1, e2, e3, r1, r2, r3, label, quiver_kw, stream_kw, streamlines, params",
    [
        (z, y, x, None, None, None, None, None, None, False, None),
        (z, y, x, (x, -2, 2), None, None, None, None, None, False, None),
        (z, y, x, None, (y, -3, 3), None, None, None, None, False, None),
        (z, y, x, None, None, (z, -4, 4), None, None, None, False, None),
        (z, y, x, (x, -2, 2), (y, -3, 3), (z, -4, 4), None, None, None, False, None),
        (v5, None, None, (x, -2, 2), (y, -3, 3), (z, -4, 4), None, None, None, False, None),
        (v6, None, None, (x, -2, 2), (y, -3, 3), (z, -4, 4), None, None, None, False, None),
        (v7, None, None, (x, -2, 2), (y, -3, 3), (z, -4, 4), None, None, None, False, None),
        (v8, None, None, (x, -2, 2), (y, -3, 3), (z, -4, 4), None, None, None, False, None),
        (z, y, x, (x, -2, 2), (y, -3, 3), (z, -4, 4), "test", {"headlength": 10}, {"lw": 5}, False, None),
        (z, y, x, (x, -2, 2), (y, -3, 3), (z, -4, 4), "test", {"headlength": 10}, {"lw": 5}, True, None),
        (z, y, x, (x, -2, 2), (y, -3, 3), (z, -4, 4), "test", {"headlength": 10}, {"lw": 5}, False, {p1: (1, 0, 2)}),
])
def test_vector_field_3d(
    default_range, e1, e2, e3, r1, r2, r3, label,
    quiver_kw, stream_kw, streamlines, params
):
    kwargs = {}
    if params:
        kwargs["params"] = params
    if quiver_kw:
        kwargs["quiver_kw"] = quiver_kw
    if stream_kw:
        kwargs["stream_kw"] = quiver_kw
    series = vector_field_3d(
        e1, e2, e3, range1=r1, range2=r2, range3=r3,
        streamlines=streamlines, **kwargs)
    assert len(series) == 1
    assert isinstance(series[0], Vector3DSeries)
    r1c = default_range(x) if not r1 else r1
    r2c = default_range(y) if not r2 else r2
    r3c = default_range(z) if not r3 else r3
    if (not r1) and (not r2) and (not r3):
        assert any(series[0].ranges[0] == t for t in (r1c, r2c, r3c))
        assert any(series[0].ranges[1] == t for t in (r1c, r2c, r3c))
        assert any(series[0].ranges[2] == t for t in (r1c, r2c, r3c))
    elif (not r2) and (not r3):
        assert series[0].ranges[0] == r1c
        assert any(series[0].ranges[1] == t for t in (r2c, r3c))
        assert any(series[0].ranges[2] == t for t in (r2c, r3c))
    elif (not r1) and (not r3):
        assert series[0].ranges[1] == r2c
        assert any(series[0].ranges[0] == t for t in (r1c, r3c))
        assert any(series[0].ranges[2] == t for t in (r1c, r3c))
    elif (not r1) and (not r2):
        assert series[0].ranges[2] == r3c
        assert any(series[0].ranges[0] == t for t in (r1c, r2c))
        assert any(series[0].ranges[1] == t for t in (r1c, r2c))
    else:
        assert series[0].ranges[0] == r1c
        assert series[0].ranges[1] == r2c
        assert series[0].ranges[2] == r3c
    if streamlines:
        assert series[0].rendering_kw == {} if not stream_kw else stream_kw
    else:
        assert series[0].rendering_kw == {} if not quiver_kw else quiver_kw
    assert series[0].is_interactive == (len(series[0].params) > 0)
    assert series[0].params == {} if not params else params


def test_vector_field_3d_sliced_series():
    u, v = symbols("u, v")
    N = CoordSys3D("N")
    i, j, k = N.base_vectors()
    xn, yn, zn = N.base_scalars()

    t = 0.35    # half-cone angle in radians
    expr = -xn**2 * tan(t)**2 + yn**2 + zn**2    # cone surface equation
    g = gradient(expr)
    n = g / g.magnitude()    # unit normal vector
    n1, n2 = 10, 20   # number of discretization points for the vector field

    cone_surface = surface_parametric(
        u / tan(t), u * cos(v), u * sin(v), (u, 0, 1), (v, 0, 2*pi),
        n1=n1, n2=n2
    )[0]

    vf = vector_field_3d(
        n, range1=(xn, -5, 5), range2=(yn, -5, 5), range3=(zn, -5, 5),
        slice=cone_surface, use_cm=False,
        quiver_kw={"scale": 0.5, "pivot": "tail"}
    )
    assert len(vf) == 1
    assert isinstance(vf[0], SliceVector3DSeries)
    assert vf[0].rendering_kw == {"scale": 0.5, "pivot": "tail"}


@pytest.mark.parametrize(
    "start, direc, label, rkw, sil, params",
    [
        ((1, 2), (3, 4), None, None, True, None),
        ((1, 2), (3, 4), "test", {"color": "r"}, False, None),
        (Tuple(j, k), (3, 4), None, None, True, {j: (1, 0, 2), k: (2, 0, 3)}),
        (Tuple(j, k), (3, 4), "test", {"color": "r"}, False, {j: (1, 0, 2), k: (2, 0, 3)}),
    ]
)
def test_arrow_2d(start, direc, label, rkw, sil, params):
    kwargs = {}
    if params:
        kwargs["params"] = params
    series = arrow_2d(
        start, direc, label=label, rendering_kw=rkw, show_in_legend=sil,
        **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, Arrow2DSeries)
    assert s.show_in_legend is sil
    if not params:
        assert s.get_label(False) == (
            "(1.0, 2.0) -> (4.0, 6.0)" if not label else label)
    else:
        assert s.get_label(False) == (
            "(j, k) -> (j + 3, k + 4)" if not label else label)
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize(
    "start, direc, label, rkw, sil, params",
    [
        ((1, 2, 3), (4, 5, 6), None, None, True, None),
        ((1, 2, 3), (4, 5, 6), "test", {"color": "r"}, False, None),
        (Tuple(j, k, l), (4, 5, 6), None, None, True, {j: (1, 0, 2), k: (2, 0, 3), l: (3, 0, 4)}),
        (Tuple(j, k, l), (4, 5, 6), "test", {"color": "r"}, False, {j: (1, 0, 2), k: (2, 0, 3), l: (3, 0, 4)}),
    ]
)
def test_arrow_3d(start, direc, label, rkw, sil, params):
    kwargs = {}
    if params:
        kwargs["params"] = params
    series = arrow_2d(
        start, direc, label=label, rendering_kw=rkw, show_in_legend=sil,
        **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, Arrow2DSeries)
    assert s.show_in_legend is sil
    if not params:
        assert s.get_label(False) == (
            "(1.0, 2.0, 3.0) -> (5.0, 7.0, 9.0)" if not label else label)
    else:
        assert s.get_label(False) == (
            "(j, k, l) -> (j + 4, k + 5, l + 6)" if not label else label)
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params
