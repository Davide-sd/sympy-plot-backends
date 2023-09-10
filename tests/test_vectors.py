from pytest import raises, warns
from spb.backends.utils import get_seeds_points
from spb.interactive.panel import InteractivePlot
from spb.series import (
    Vector2DSeries, Vector3DSeries, SliceVector3DSeries, ContourSeries,
    PlaneSeries, SurfaceOver2DRangeSeries, ParametricSurfaceSeries
)
from spb.utils import _plot_sympify
from spb.vectors import _preprocess, _series, plot_vector, _split_vector
from spb import plot_vector, MB, prange
from sympy import symbols, Matrix, Tuple, sin, cos, sqrt, Plane, pi
from sympy.physics.mechanics import Vector as MechVector, ReferenceFrame
from sympy.vector import CoordSys3D
from sympy.external import import_module
import numpy as np
import pytest


@pytest.fixture
def pv_options(panel_options):
    panel_options["backend"] = MB
    panel_options["n"] = 3
    panel_options["nc"] = 3
    return panel_options


@pytest.fixture
def pv_options3d(panel_options):
    panel_options["backend"] = MB
    panel_options["n"] = 4
    return panel_options


def pw(*args):
    """_preprocess wrapper. Only need it to sympify the arguments before
    calling _preprocess."""
    args = _plot_sympify(args)
    return _preprocess(*args)


def to_float(t):
    return tuple(float(v) for v in t)


def test_preprocess():
    # verify that the preprocessing is correctly applied to the
    # input arguments

    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j + z * N.k
    v2 = z * N.i + x * N.j + y * N.k
    m1 = v1.to_matrix(N)
    m2 = v2.to_matrix(N)
    l1 = list(m1)
    l2 = list(m2)

    # passing in vectors
    r = pw(v1)[0]
    assert r[0] == v1
    assert r[1] == None

    r = pw(v1, (x, -5, 5), "test")[0]
    assert r == [v1, Tuple(x, -5, 5), "test"]

    r = pw((v1, (x, -5, 5), "test"))[0]
    assert r == [v1, Tuple(x, -5, 5), "test"]

    r = pw((v1, (x, -5, 5), "v1"), (v2, (x, -5, 5), (y, -2, 2)))
    assert r[0] == [v1, Tuple(x, -5, 5), "v1"]
    assert r[1] == [v2, Tuple(x, -5, 5), Tuple(y, -2, 2), None]

    r = pw(v1, v2, (x, -5, 5), (y, -2, 2), (z, -3, 3))
    assert r[0] == [v1, Tuple(x, -5, 5), Tuple(y, -2, 2), Tuple(z, -3, 3), None]
    assert r[1] == [v2, Tuple(x, -5, 5), Tuple(y, -2, 2), Tuple(z, -3, 3), None]

    # passing in matrices
    r = pw(m1, (x, -5, 5), "test")[0]
    assert r == [m1, Tuple(x, -5, 5), "test"]

    r = pw(m1, m2, (x, -5, 5), (y, -2, 2), (z, -3, 3))
    assert r[0] == [m1, Tuple(x, -5, 5), Tuple(y, -2, 2), Tuple(z, -3, 3), None]
    assert r[1] == [m2, Tuple(x, -5, 5), Tuple(y, -2, 2), Tuple(z, -3, 3), None]

    # passing in lists
    r = pw(l1, (x, -5, 5), "test")[0]
    assert r == [tuple(l1), Tuple(x, -5, 5), "test"]

    r = pw(l1, l2, (x, -5, 5), (y, -2, 2), (z, -3, 3))
    assert r[0] == [
        tuple(l1),
        Tuple(x, -5, 5),
        Tuple(y, -2, 2),
        Tuple(z, -3, 3),
        None,
    ]
    assert r[1] == [
        tuple(l2),
        Tuple(x, -5, 5),
        Tuple(y, -2, 2),
        Tuple(z, -3, 3),
        None,
    ]


def test_split_vector():
    # verify that the correct components of a vector are retrieved, no matter
    # the type of the input vector (list, matrix, symbolic vector, lambda
    # functions)

    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j + z * N.k
    v2 = z * N.i + x * N.j + y * N.k
    m1 = v1.to_matrix(N)
    m2 = v2.to_matrix(N)
    l1 = list(m1)
    l2 = list(m2)
    fx = lambda x, y, z: z
    fy = lambda x, y, z: x
    fz = lambda x, y, z: y
    A = ReferenceFrame("A")
    v3 = -sin(y) * A.x + cos(x) * A.y
    v4 = -sin(y) * A.x + cos(x) * A.y + cos(z) * A.z


    ranges_in = [Tuple(x, -5, 5)]
    ranges_out = [Tuple(x, -5, 5), Tuple(y, -10, 10), Tuple(z, -10, 10)]

    def do_test(expr_in, expr_out):
        exprs, ranges = _split_vector(expr_in, ranges_in)
        assert exprs == expr_out
        assert all([r in ranges_out for r in ranges])

    do_test(v1, (x, y, z))
    do_test(m1, (x, y, z))
    do_test(l1, (x, y, z))
    do_test(v2, (z, x, y))
    do_test(m2, (z, x, y))
    do_test(l2, (z, x, y))
    do_test([fx, fy, fz], (fx, fy, fz))
    do_test(v3, (-sin(y), cos(x), 0))
    do_test(v4, (-sin(y), cos(x), cos(z)))

    # too few or too many elements
    raises(ValueError, lambda: _split_vector([x], ranges_in))
    raises(ValueError, lambda: _split_vector([x, x, x, x], ranges_in))


def test_get_seeds_points():
    # verify that spb.backends.utils.get_seeds_points returns the correct
    # data type based on the parameters

    vtk = import_module('vtk', catch=(RuntimeError,))

    x, y, z = symbols("x:z")
    s = Vector3DSeries(z, y, x, (x, -5, 5), (y, -3, 3), (z, -2, 2))
    xx, yy, zz, uu, vv, ww = s.get_data()

    #
    # Case 1: search boundary points where the vector is pointing inward the
    # domain
    #
    d = get_seeds_points(xx, yy, zz, uu, vv, ww, True,
        **dict(starts=None))
    assert isinstance(d, np.ndarray)
    assert len(d.shape) == 2 and (d.shape[1] == 3)

    d = get_seeds_points(xx, yy, zz, uu, vv, ww, False,
        **dict(starts=None))
    assert isinstance(d, vtk.vtkPolyData)

    #
    # Case 2: user-provided starting points
    #
    xx2 = np.linspace(-5, 5, 10)
    yy2 = np.linspace(-3, 3, 10)
    zz2 = np.linspace(-2, 2, 10)
    d = get_seeds_points(xx, yy, zz, uu, vv, ww, True,
        **dict(starts={
            "x": xx2,
            "y": yy2,
            "z": zz2
        }))
    assert isinstance(d, np.ndarray)
    assert len(d.shape) == 2 and (d.shape == (10, 3))
    assert np.all(d[:, 0] == xx2)
    assert np.all(d[:, 1] == yy2)
    assert np.all(d[:, 2] == zz2)

    d = get_seeds_points(xx, yy, zz, uu, vv, ww, False,
        **dict(starts={
            "x": xx2,
            "y": yy2,
            "z": zz2
        }))
    assert isinstance(d, vtk.vtkPolyData)

    #
    # Case 3: generate random locations
    #
    d = get_seeds_points(xx, yy, zz, uu, vv, ww, True,
        **dict(starts=True, npoints=10))
    assert isinstance(d, np.ndarray)
    assert len(d.shape) == 2 and (d.shape == (10, 3))

    d = get_seeds_points(xx, yy, zz, uu, vv, ww, False,
        **dict(starts=True, npoints=10))
    assert isinstance(d, vtk.vtkPointSource)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_plot_vector_2d(pv_options):
    # verify that plot_vector is capable of creating data
    # series according to the documented modes of operation when dealing
    # with 2D vectors

    x, y, z = symbols("x:z")

    ###########################################################################
    ########### plot_vector(expr, range1, range2, label [opt]) ################
    ###########################################################################

    # default to a contour series and quivers
    p = plot_vector([-sin(y), cos(x)], (x, -3, 2), (y, -4, 6),
        **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries)
    assert s[0].expr == sqrt(sin(y)**2 + cos(x)**2)
    assert (s[0].var_x, s[0].start_x, s[0].end_x) == (x, -3, 2)
    assert (s[0].var_y, s[0].start_y, s[0].end_y) == (y, -4, 6)
    assert s[0].get_label(False) == "Magnitude"
    assert isinstance(s[1], Vector2DSeries)
    assert not s[1].is_streamlines
    assert to_float(s[1].ranges[0][1:]) == (-3, 2)
    assert to_float(s[1].ranges[1][1:]) == (-4, 6)
    assert s[1].get_label(False) == '(-sin(y), cos(x))'

    p = plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)}, **pv_options)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries) and s[0].is_interactive
    xx, yy, zz = s[0].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert s[0].get_label(False) == "Magnitude"
    assert isinstance(s[1], Vector2DSeries) and s[1].is_interactive
    assert not s[1].is_streamlines
    xx, yy, _, _ = s[1].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert s[1].get_label(False) == '(-sin(y*z), cos(x))'

    # same as before + set custom contour_kw and quiver_kw
    p = plot_vector([-sin(y), cos(x)], (x, -3, 2), (y, -4, 6),
        quiver_kw=dict(color="black"),
        contour_kw={"cmap": "Blues_r", "levels": 20}, **pv_options)
    s = p.series
    assert s[0].rendering_kw == {"cmap": "Blues_r", "levels": 20}
    assert s[1].rendering_kw == dict(color="black")

    p = plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)},
        quiver_kw=dict(color="black"),
        contour_kw={"cmap": "Blues_r", "levels": 20}, **pv_options)
    s = p.backend.series
    assert s[0].rendering_kw == {"cmap": "Blues_r", "levels": 20}
    assert s[1].rendering_kw == dict(color="black")

    # 2D vector field without contour series, only quivers, use streamlines
    p = plot_vector([-sin(y), cos(x)], (x, -3, 2), (y, -4, 6),
        scalar=None, streamlines=True, **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], Vector2DSeries)
    assert s[0].is_streamlines
    assert to_float(s[0].ranges[0][1:]) == (-3, 2)
    assert to_float(s[0].ranges[1][1:]) == (-4, 6)
    assert s[0].get_label(False) == '(-sin(y), cos(x))'

    p = plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)}, scalar=None, streamlines=True, **pv_options)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], Vector2DSeries) and s[0].is_interactive
    assert s[0].is_streamlines
    xx, yy, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert s[0].get_label(False) == '(-sin(y*z), cos(x))'

    # same as before + custom label (which might eventually be shown on the
    # colorbar). Note the use of scalar=False which is equal to scalar=None.
    # Also, apply stream_kw
    p = plot_vector([-sin(y), cos(x)], (x, -3, 2), (y, -4, 6), "test",
        scalar=False, streamlines=True, stream_kw={"cmap": "Blues_r"},
        **pv_options)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Blues_r"}
    p = plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6), "test",
        params={z: (1, 0, 2)}, scalar=False, streamlines=True,
        stream_kw={"cmap": "Blues_r"}, **pv_options)
    assert p.backend[0].get_label(False) == "test"
    assert p.backend[0].rendering_kw == {"cmap": "Blues_r"}

    ###########################################################################
    ############## plot_vector(expr1, expr2, range1, range2) ##################
    ###########################################################################

    # multiple 2D vector fields: only quivers will be shown
    p = plot_vector([-sin(y), cos(x)], [-y, x], (x, -3, 2), (y, -4, 6),
        **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, Vector2DSeries) for t in s)
    assert all(to_float(t.ranges[0][1:]) == (-3, 2) for t in s)
    assert all(to_float(t.ranges[1][1:]) == (-4, 6) for t in s)
    assert s[0].get_label(False) == '(-sin(y), cos(x))'
    assert s[1].get_label(False) == '(-y, x)'

    p = plot_vector([-sin(y * z), cos(x)], [-y, x], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)}, **pv_options)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, Vector2DSeries) and t.is_interactive for t in s)
    xx, yy, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert s[0].get_label(False) == '(-sin(y*z), cos(x))'
    xx, yy, _, _ = s[1].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert s[1].get_label(False) == '(-y, x)'

    # multiple 2D vector field with a scalar field: contour + quivers
    p = plot_vector([-sin(y), cos(x)], [-y, x], (x, -3, 2), (y, -4, 6),
        scalar=sqrt(x**2 + y**2), **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 3
    assert isinstance(s[0], ContourSeries)
    assert s[0].expr == sqrt(x**2 + y**2)
    assert (s[0].var_x, s[0].start_x, s[0].end_x) == (x, -3, 2)
    assert (s[0].var_y, s[0].start_y, s[0].end_y) == (y, -4, 6)
    assert isinstance(s[1], Vector2DSeries)
    assert isinstance(s[2], Vector2DSeries)
    assert all(to_float(t.ranges[0][1:]) == (-3, 2) for t in s[1:])
    assert all(to_float(t.ranges[1][1:]) == (-4, 6) for t in s[1:])
    assert s[0].get_label(False) == 'sqrt(x**2 + y**2)'
    assert s[1].get_label(False) == '(-sin(y), cos(x))'
    assert s[2].get_label(False) == '(-y, x)'

    p = plot_vector([-sin(y * z), cos(x)], [-y, x], (x, -3, 2), (y, -4, 6),
        scalar=sqrt(x**2 + y**2), params={z: (1, 0, 2)}, **pv_options)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 3
    assert isinstance(s[0], ContourSeries) and s[0].is_interactive
    assert s[0].expr == sqrt(x**2 + y**2)
    xx, yy, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert isinstance(s[1], Vector2DSeries) and s[1].is_interactive
    assert isinstance(s[2], Vector2DSeries) and s[2].is_interactive
    for t in s:
        d = t.get_data()
        xx, yy = d[:2]
        assert (xx.min(), xx.max()) == (-3, 2)
        assert (yy.min(), yy.max()) == (-4, 6)
    assert s[0].get_label(False) == 'sqrt(x**2 + y**2)'
    assert s[1].get_label(False) == '(-sin(y*z), cos(x))'
    assert s[2].get_label(False) == '(-y, x)'

    ###########################################################################
    ##### plot_vector((e1, r1, r2, lbl1 [opt]), (e2, r3, r4, lbl2 [opt])) #####
    ###########################################################################

    # multiple vector fiedls: default to only quivers
    p = plot_vector(
        ([-sin(y), cos(x)], (x, -3, 0), (y, -4, 0)),
        ([-y, x], (x, 0, 2), (y, 0, 6), "test"), **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, Vector2DSeries) for t in s)
    assert to_float(s[0].ranges[0][1:]) == (-3, 0)
    assert to_float(s[0].ranges[1][1:]) == (-4, 0)
    assert to_float(s[1].ranges[0][1:]) == (0, 2)
    assert to_float(s[1].ranges[1][1:]) == (0, 6)
    assert s[0].get_label(False) == '(-sin(y), cos(x))'
    assert s[1].get_label(False) == 'test'

    p = plot_vector(
        ([-sin(y * z), cos(x)], (x, -3, 0), (y, -4, 0)),
        ([-y, x], (x, 0, 2), (y, 0, 6), "test"),
        params={z: (1, 0, 2)}, **pv_options)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, Vector2DSeries) for t in s)
    xx, yy, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-3, 0)
    assert (yy.min(), yy.max()) == (-4, 0)
    xx, yy, _, _ = s[1].get_data()
    assert (xx.min(), xx.max()) == (0, 2)
    assert (yy.min(), yy.max()) == (0, 6)
    assert s[0].get_label(False) == '(-sin(y*z), cos(x))'
    assert s[1].get_label(False) == 'test'

    ###########################################################################
    ###################### Verify scalar keyword argument #####################
    ###########################################################################

    # scalar=True, scalar=False, scalar=None and scalar=expr has already been
    # verified before.

    # test scalar=[expr, label]
    p = plot_vector([-sin(y), cos(x)], (x, -3, 2), (y, -4, 6),
        scalar=[sqrt(x**2 + y**2), "test"], **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries)
    assert s[0].expr == sqrt(x**2 + y**2)
    assert s[0].get_label(False) == "test"
    assert (s[0].var_x, s[0].start_x, s[0].end_x) == (x, -3, 2)
    assert (s[0].var_y, s[0].start_y, s[0].end_y) == (y, -4, 6)
    assert isinstance(s[1], Vector2DSeries)
    assert to_float(s[1].ranges[0][1:]) == (-3, 2)
    assert to_float(s[1].ranges[1][1:]) == (-4, 6)
    assert s[1].get_label(False) == '(-sin(y), cos(x))'

    p = plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        scalar=[sqrt(x**2 + y**2), "test"], params={z: (1, 0, 2)},
        **pv_options)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries) and s[0].is_interactive
    assert s[0].expr == sqrt(x**2 + y**2)
    assert s[0].get_label(False) == "test"
    xx, yy, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert isinstance(s[1], Vector2DSeries) and s[1].is_interactive
    xx, yy, _, _ = s[1].get_data()
    assert (xx.min(), xx.max()) == (-3, 2)
    assert (yy.min(), yy.max()) == (-4, 6)
    assert s[1].get_label(False) == '(-sin(y*z), cos(x))'

    # test scalar=numerical function of 2 variables support vectorization
    p = plot_vector([-sin(y), cos(x)], (x, -3, 2), (y, -4, 6),
        scalar=lambda xx, yy: np.cos(xx * yy), **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries)
    assert callable(s[0].expr)
    assert s[0].get_label(False) == "Magnitude"
    assert (s[0].var_x, s[0].start_x, s[0].end_x) == (x, -3, 2)
    assert (s[0].var_y, s[0].start_y, s[0].end_y) == (y, -4, 6)
    assert isinstance(s[1], Vector2DSeries)
    assert to_float(s[1].ranges[0][1:]) == (-3, 2)
    assert to_float(s[1].ranges[1][1:]) == (-4, 6)
    assert s[1].get_label(False) == '(-sin(y), cos(x))'

    pl = lambda: plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        scalar=lambda xx, yy: np.cos(xx * yy), params={z: (1, 0, 2)},
        **pv_options)
    raises(TypeError, pl)

    # test scalar=[nume func of 2 variables, label]
    p = plot_vector([-sin(y), cos(x)], (x, -3, 2), (y, -4, 6),
        scalar=[lambda xx, yy: np.cos(xx * yy), "test"],
        **pv_options)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries)
    assert callable(s[0].expr)
    assert s[0].get_label(False) == "test"
    assert (s[0].var_x, s[0].start_x, s[0].end_x) == (x, -3, 2)
    assert (s[0].var_y, s[0].start_y, s[0].end_y) == (y, -4, 6)
    assert isinstance(s[1], Vector2DSeries)
    assert to_float(s[1].ranges[0][1:]) == (-3, 2)
    assert to_float(s[1].ranges[1][1:]) == (-4, 6)
    assert s[1].get_label(False) == '(-sin(y), cos(x))'

    pl = lambda: plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        scalar=[lambda xx, yy: np.cos(xx * yy), "test"], params={z: (1, 0, 2)},
        **pv_options)
    raises(TypeError, pl)


def test_plot_vector_3d(pv_options3d):
    # verify that plot_vector is capable of creating data
    # series according to the documented modes of operation when dealing
    # with 3d vector fields

    x, y, z, u = symbols("x:z u")

    ###########################################################################
    ########## plot_vector(expr, range1, range2, range3, label [opt]) #########
    ###########################################################################

    p = plot_vector([x, y, z], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        quiver_kw={"cmap": "Blues_r"}, **pv_options3d)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], Vector3DSeries)
    assert not s[0].is_streamlines
    assert to_float(s[0].ranges[0][1:]) == (-10, 9)
    assert to_float(s[0].ranges[1][1:]) == (-8, 7)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert s[0].expr == (x, y, z)
    assert s[0].get_label(False) == "(x, y, z)"
    assert s[0].rendering_kw == {"cmap": "Blues_r"}

    p = plot_vector([x * u, y, z], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        params={u: (1, 0, 2)}, quiver_kw={"cmap": "Blues_r"}, **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], Vector3DSeries) and s[0].is_interactive
    assert not s[0].is_streamlines
    xx, yy, zz, _, _, _ = s[0].get_data()
    assert s[0].expr == (u*x, y, z)
    assert s[0].get_label(False) == "(u*x, y, z)"
    assert (xx.min(), xx.max()) == (-10, 9)
    assert (yy.min(), yy.max()) == (-8, 7)
    assert (zz.min(), zz.max()) == (-6, 5)
    assert s[0].rendering_kw == {"cmap": "Blues_r"}

    # same as before, use streamlines instead + custom label
    p = plot_vector([x, y, z], (x, -10, 9), (y, -8, 7), (z, -6, 5), "test",
        streamlines=True, stream_kw={"cmap": "Blues_r"}, **pv_options3d)
    s = p.series
    assert s[0].is_streamlines
    assert s[0].get_label(False) == "test"
    assert s[0].rendering_kw == {"cmap": "Blues_r"}

    p = plot_vector([x * u, y, z], (x, -10, 9), (y, -8, 7), (z, -6, 5), "test",
        params={u: (1, 0, 2)}, streamlines=True, stream_kw={"cmap": "Blues_r"},
        **pv_options3d)
    s = p.backend.series
    assert s[0].is_streamlines
    assert s[0].get_label(False) == "test"
    assert s[0].rendering_kw == {"cmap": "Blues_r"}

    ###########################################################################
    ###### plot_vector(expr1, expr2, range1, range2, range3, label [opt]) #####
    ###########################################################################

    p = plot_vector([x, y, z], [-sin(y), cos(z), y],
        (x, -10, 9), (y, -8, 7), (z, -6, 5), **pv_options3d)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, Vector3DSeries) for t in s)
    assert all(not t.is_streamlines for t in s)
    assert to_float(all(t.ranges[0][1:]) == (-10, 9) for t in s)
    assert to_float(all(t.ranges[1][1:]) == (-8, 7) for t in s)
    assert to_float(all(t.ranges[2][1:]) == (-6, 5) for t in s)
    assert s[0].expr == (x, y, z)
    assert s[1].expr == (-sin(y), cos(z), y)
    assert s[0].get_label(False) == "(x, y, z)"
    assert s[1].get_label(False) == "(-sin(y), cos(z), y)"

    p = plot_vector([x * u, y, z], [-sin(u * y), cos(z), y],
        (x, -10, 9), (y, -8, 7), (z, -6, 5),
        params={u: (1, 0, 2)}, **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, Vector3DSeries) and t.is_interactive for t in s)
    assert all(not t.is_streamlines for t in s)
    for t in s:
        xx, yy, zz, _, _, _ = t.get_data()
        assert (xx.min(), xx.max()) == (-10, 9)
        assert (yy.min(), yy.max()) == (-8, 7)
        assert (zz.min(), zz.max()) == (-6, 5)
    assert s[0].expr == (u*x, y, z)
    assert s[0].get_label(False) == "(u*x, y, z)"
    assert s[1].expr == (-sin(u * y), cos(z), y)
    assert s[1].get_label(False) == "(-sin(u*y), cos(z), y)"

    ###########################################################################
    ## plot_vector(
    ##    (e1, r1, r2, r3, lbl1 [opt]),
    ##    (e2, r4, r5, r6, lbl2 [opt]))
    ###########################################################################

    p = plot_vector(
        ([x, y, z], (x, -10, 9), (y, -8, 7), (z, -6, 5)),
        ([-sin(y), cos(z), y], (x, -4, 3), (y, -2, 11), (z, -1, 12), "test"),
        **pv_options3d)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, Vector3DSeries) for t in s)
    assert all(not t.is_streamlines for t in s)
    assert to_float(s[0].ranges[0][1:]) == (-10, 9)
    assert to_float(s[0].ranges[1][1:]) == (-8, 7)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert to_float(s[1].ranges[0][1:]) == (-4, 3)
    assert to_float(s[1].ranges[1][1:]) == (-2, 11)
    assert to_float(s[1].ranges[2][1:]) == (-1, 12)
    assert s[0].expr == (x, y, z)
    assert s[1].expr == (-sin(y), cos(z), y)
    assert s[0].get_label(False) == "(x, y, z)"
    assert s[1].get_label(False) == "test"

    p = plot_vector(
        ([u * x, y, z], (x, -10, 9), (y, -8, 7), (z, -6, 5)),
        ([-sin(u * y), cos(z), y], (x, -4, 3), (y, -2, 11), (z, -1, 12), "test"),
        params={u: (1, 0, 2)}, **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, Vector3DSeries) and t.is_interactive for t in s)
    assert all(not t.is_streamlines for t in s)
    xx, yy, zz, _, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-10, 9)
    assert (yy.min(), yy.max()) == (-8, 7)
    assert (zz.min(), zz.max()) == (-6, 5)
    xx, yy, zz, _, _, _ = s[1].get_data()
    assert (xx.min(), xx.max()) == (-4, 3)
    assert (yy.min(), yy.max()) == (-2, 11)
    assert (zz.min(), zz.max()) == (-1, 12)
    assert s[0].expr == (u*x, y, z)
    assert s[0].get_label(False) == "(u*x, y, z)"
    assert s[1].expr == (-sin(u * y), cos(z), y)
    assert s[1].get_label(False) == "test"


def test_plot_vector_3d_slice(pv_options3d):
    # verify that plot_vector is capable of creating data
    # series according to the documented modes of operation when dealing
    # with 3d vector fields and slices

    x, y, z, u, v = symbols("x:z, u, v")
    pv_options3d["n"] = 5

    ###########################################################################
    ################################## Plane ##################################
    ###########################################################################

    # Single slice plane
    p = plot_vector([z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=Plane((-10, 0, 0), (1, 0, 0)),
        quiver_kw={"cmap": "Blues_r"}, **pv_options3d)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries)
    assert isinstance(s[0].slice_surf_series, PlaneSeries)
    assert to_float(s[0].ranges[0][1:]) == (-10, 9)
    assert to_float(s[0].ranges[1][1:]) == (-8, 7)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert s[0].get_label(False) == "(z, y, x)"
    assert s[0].rendering_kw == {"cmap": "Blues_r"}

    p = plot_vector([u * z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        params={u: (1, 0, 2)},
        slice=Plane((-10, 0, 0), (1, 0, 0)),
        quiver_kw={"cmap": "Blues_r"}, **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries) and s[0].is_interactive
    assert isinstance(s[0].slice_surf_series, PlaneSeries)
    xx, yy, zz, _, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-10, -10)
    assert (yy.min(), yy.max()) == (-8, 7)
    assert (zz.min(), zz.max()) == (-6, 5)
    assert s[0].get_label(False) == "(u*z, y, x)"
    assert s[0].rendering_kw == {"cmap": "Blues_r"}

    # Multiple slice planes
    p = plot_vector([z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=[
            Plane((-10, 0, 0), (1, 0, 0)),
            Plane((0, -8, 0), (0, 1, 0)),
            Plane((0, 0, -6), (0, 0, 1)),
        ], **pv_options3d)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 3
    assert all(isinstance(t, SliceVector3DSeries) for t in s)
    assert all(isinstance(t.slice_surf_series, PlaneSeries) for t in s)
    assert all(to_float(t.ranges[0][1:]) == (-10, 9) for t in s)
    assert all(to_float(t.ranges[1][1:]) == (-8, 7) for t in s)
    assert all(to_float(t.ranges[2][1:]) == (-6, 5) for t in s)
    assert all(t.get_label(False) == "(z, y, x)" for t in s)

    p = plot_vector([u * z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        params={u: (1, 0, 2)},
        slice=[
            Plane((-10 * u, 0, 0), (1, 0, 0)),
            Plane((0, -8, 0), (0, 1, 0)),
            Plane((0, 0, -6), (0, 0, 1)),
        ], **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 3
    assert all(isinstance(t, SliceVector3DSeries) and t.is_interactive for t in s)
    assert all(isinstance(t.slice_surf_series, PlaneSeries) for t in s)
    xx, yy, zz, _, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-10, -10)
    assert (yy.min(), yy.max()) == (-8, 7)
    assert (zz.min(), zz.max()) == (-6, 5)
    xx, yy, zz, _, _, _ = s[1].get_data()
    assert (xx.min(), xx.max()) == (-10, 9)
    assert (yy.min(), yy.max()) == (-8, -8)
    assert (zz.min(), zz.max()) == (-6, 5)
    xx, yy, zz, _, _, _ = s[2].get_data()
    assert (xx.min(), xx.max()) == (-10, 9)
    assert (yy.min(), yy.max()) == (-8, 7)
    assert (zz.min(), zz.max()) == (-6, -6)
    assert all(t.get_label(False) == "(u*z, y, x)" for t in s)

    ###########################################################################
    ########################## Symbolic Expression ############################
    ###########################################################################

    # symbolic expression (must have 2 variables)
    # this surface lies on the x-y plane
    p1a = plot_vector([z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        slice=cos(x**2 + y**2), **pv_options3d)
    s = p1a.series
    assert isinstance(p1a, MB)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries)
    assert isinstance(s[0].slice_surf_series, SurfaceOver2DRangeSeries)
    assert to_float(s[0].ranges[0][1:]) == (-2, 3)
    assert to_float(s[0].ranges[1][1:]) == (-3, 1)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert s[0].get_label(False) == "(z, y, x)"
    t = s[0].slice_surf_series
    assert (t.var_x, t.start_x, t.end_x) == (x, -2, 3)
    assert (t.var_y, t.start_y, t.end_y) == (y, -3, 1)
    d1a = s[0].get_data()

    p1b = plot_vector([u * z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        params={u: (1, 0, 2)}, slice=cos(u * x**2 + y**2), **pv_options3d)
    s = p1b.backend.series
    assert isinstance(p1b, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries) and s[0].is_interactive
    assert isinstance(s[0].slice_surf_series, SurfaceOver2DRangeSeries) and s[0].slice_surf_series.is_interactive
    d1b = s[0].get_data()
    xx, yy, zz, _, _, _ = d1b
    assert (xx.min(), xx.max()) == (-2, 3)
    assert (yy.min(), yy.max()) == (-3, 1)
    assert s[0].get_label(False) == "(u*z, y, x)"
    xx, yy, _ = s[0].slice_surf_series.get_data()
    assert (xx.min(), xx.max()) == (-2, 3)
    assert (yy.min(), yy.max()) == (-3, 1)


    # this surface lies on the y-z plane
    p2a = plot_vector([z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        slice=cos(z**2 + y**2), **pv_options3d)
    s = p2a.series
    assert isinstance(p2a, MB)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries)
    assert isinstance(s[0].slice_surf_series, SurfaceOver2DRangeSeries)
    assert to_float(s[0].ranges[0][1:]) == (-2, 3)
    assert to_float(s[0].ranges[1][1:]) == (-3, 1)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert s[0].get_label(False) == "(z, y, x)"
    t = s[0].slice_surf_series
    assert (t.var_x, t.start_x, t.end_x) == (y, -3, 1)
    assert (t.var_y, t.start_y, t.end_y) == (z, -6, 5)
    d2a = s[0].get_data()

    # since the two expressions lies on different planes, they must produce
    # different data
    for d1, d2 in zip(d1a, d2a):
        assert not np.allclose(d1, d2)

    p2b = plot_vector([u * z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        params={u: (1, 0, 2)}, slice=cos(u * z**2 + y**2), **pv_options3d)
    s = p2b.backend.series
    assert isinstance(p2b, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries) and s[0].is_interactive
    assert isinstance(s[0].slice_surf_series, SurfaceOver2DRangeSeries) and s[0].slice_surf_series.is_interactive
    d2b = s[0].get_data()
    xx, yy, zz, _, _, _ = d2b
    assert (yy.min(), yy.max()) == (-3, 1)
    assert (zz.min(), zz.max()) == (-6, 5)
    assert s[0].get_label(False) == "(u*z, y, x)"
    xx, yy, _ = s[0].slice_surf_series.get_data()
    assert (xx.min(), xx.max()) == (-3, 1)
    assert (yy.min(), yy.max()) == (-6, 5)


    # since the two expressions lies on different planes, they must produce
    # different data
    for d1, d2 in zip(d1b, d2b):
        assert not np.allclose(d1, d2)

    ###########################################################################
    ############################# Surface Series ##############################
    ###########################################################################

    ss = ParametricSurfaceSeries(u * cos(v), u * sin(v), u,
        (u, -2, 0), (v, 0, 2*pi), n1=5, n2=5)
    p = plot_vector([z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=ss, **pv_options3d)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries)
    pss = s[0].slice_surf_series
    assert isinstance(pss, ParametricSurfaceSeries)
    assert pss.expr == (u * cos(v), u * sin(v), u)
    assert (pss.var_u, pss.start_u, pss.end_u) == (u, -2, 0)
    assert (pss.var_v, pss.start_v, pss.end_v) == (v, 0, float(2*pi))
    assert to_float(s[0].ranges[0][1:]) == (-10, 9)
    assert to_float(s[0].ranges[1][1:]) == (-8, 7)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert s[0].get_label(False) == "(z, y, x)"

    ss = SurfaceOver2DRangeSeries(cos(z**2 + y**2), (z, -4, 4), (y, -2, 2),
        n1=5, n2=5)
    p3a = plot_vector([z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=ss, **pv_options3d)
    s = p3a.series
    assert isinstance(p3a, MB)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries)
    pss = s[0].slice_surf_series
    assert isinstance(pss, SurfaceOver2DRangeSeries)
    assert pss.expr == cos(z**2 + y**2)
    assert (pss.var_x, pss.start_x, pss.end_x) == (z, -4, 4)
    assert (pss.var_y, pss.start_y, pss.end_y) == (y, -2, 2)
    assert to_float(s[0].ranges[0][1:]) == (-10, 9)
    assert to_float(s[0].ranges[1][1:]) == (-8, 7)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert s[0].get_label(False) == "(z, y, x)"
    d3a = s[0].get_data()

    ss = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -4, 4), (y, -2, 2),
        n1=5, n2=5)
    p3b = plot_vector([z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=ss, **pv_options3d)
    s = p3b.series
    assert isinstance(p3b, MB)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries)
    pss = s[0].slice_surf_series
    assert isinstance(pss, SurfaceOver2DRangeSeries)
    assert pss.expr == cos(x**2 + y**2)
    assert (pss.var_x, pss.start_x, pss.end_x) == (x, -4, 4)
    assert (pss.var_y, pss.start_y, pss.end_y) == (y, -2, 2)
    assert to_float(s[0].ranges[0][1:]) == (-10, 9)
    assert to_float(s[0].ranges[1][1:]) == (-8, 7)
    assert to_float(s[0].ranges[2][1:]) == (-6, 5)
    assert s[0].get_label(False) == "(z, y, x)"
    d3b = s[0].get_data()

    # since the two expressions lies on different planes, they must produce
    # different data
    assert np.allclose(d3a[0], d3b[2])
    assert np.allclose(d3a[3], d3b[5])
    assert np.allclose(d3a[1], d3b[1])
    assert np.allclose(d3a[4], d3b[4])

    t = symbols("t")
    ss = ParametricSurfaceSeries(u * cos(v), u * sin(v), t*u,
        (u, -2, 0), (v, 0, 2*pi), n1=5, n2=5, params={t: 1})
    p = plot_vector([t * z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=ss, params={t: (1, 0.5, 1.5)}, **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries) and s[0].is_interactive
    pss = s[0].slice_surf_series
    assert isinstance(pss, ParametricSurfaceSeries) and pss.is_interactive
    assert pss.expr == (u * cos(v), u * sin(v), t * u)
    _, _, _, uu, vv = pss.get_data()
    assert (uu.min(), uu.max()) == (-2, 0)
    assert (vv.min(), vv.max()) == (0, float(2*pi))
    xx, yy, zz, _, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-2, 2)
    assert (yy.min(), yy.max()) == (-2, 2)
    assert (zz.min(), zz.max()) == (-2, 0)
    assert s[0].get_label(False) == "(t*z, y, x)"

    ss = SurfaceOver2DRangeSeries(cos(t * x**2 + y**2),
        (x, -4, 5), (y, -3, 2), n1=25, n2=25, params={t: 1})
    p = plot_vector([t * z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=ss, params={t: (1, 0.5, 1.5)}, **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries) and s[0].is_interactive
    pss = s[0].slice_surf_series
    assert isinstance(pss, SurfaceOver2DRangeSeries) and pss.is_interactive
    assert pss.expr == cos(t * x**2 + y**2)
    xx, yy, _ = pss.get_data()
    assert (xx.min(), xx.max()) == (-4, 5)
    assert (yy.min(), yy.max()) == (-3, 2)
    xx, yy, zz, _, _, _ = s[0].get_data()
    assert (xx.min(), xx.max()) == (-4, 5)
    assert (yy.min(), yy.max()) == (-3, 2)
    assert np.allclose([zz.min(), zz.max()], [-1, 1])
    assert s[0].get_label(False) == "(t*z, y, x)"

    ss = SurfaceOver2DRangeSeries(cos(t * z**2 + y**2),
        (z, -4, 5), (y, -3, 2), n1=25, n2=25, params={t: 1})
    p = plot_vector([t * z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=ss, params={t: (1, 0.5, 1.5)}, **pv_options3d)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries) and s[0].is_interactive
    pss = s[0].slice_surf_series
    assert isinstance(pss, SurfaceOver2DRangeSeries) and pss.is_interactive
    assert pss.expr == cos(t * z**2 + y**2)
    xx, yy, _ = pss.get_data()
    assert (xx.min(), xx.max()) == (-4, 5)
    assert (yy.min(), yy.max()) == (-3, 2)
    xx, yy, zz, _, _, _ = s[0].get_data()
    assert np.allclose([xx.min(), xx.max()], [-1, 1])
    assert (zz.min(), zz.max()) == (-4, 5)
    assert (yy.min(), yy.max()) == (-3, 2)
    assert s[0].get_label(False) == "(t*z, y, x)"


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_label_kw(pv_options, pv_options3d):
    # verify that the label keyword argument works, if the correct
    # number of labels is provided.

    x, y, z, t = symbols("x, y, z, t")

    # 2 series -> 2 labels
    p = plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        label=["a", "b"], **pv_options)
    assert p[0].get_label(False) == "a"
    assert p[1].get_label(False) == "b"

    p = plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)}, label=["a", "b"], **pv_options)
    s = p.backend.series
    assert s[0].get_label(False) == "a"
    assert s[1].get_label(False) == "b"

    # 1 series -> 2 labels = raise error
    p = lambda: plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        label=["a", "b"], scalar=False, **pv_options)
    raises(ValueError, p)

    p = lambda: plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)}, scalar=False,
        label=["a", "b"], **pv_options)
    raises(ValueError, p)

    # 1 series -> 1 label
    p = plot_vector([z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        slice=cos(z**2 + y**2), label="a", **pv_options3d)
    assert p[0].get_label(False) == "a"

    p = plot_vector([t * z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        slice=cos(z**2 + y**2),
        params={t: (1, 0, 2)}, label="a", **pv_options3d)
    s = p.backend.series
    assert s[0].get_label(False) == "a"

    # 3 series -> 3 labels
    p = plot_vector([z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=[
            Plane((-10, 0, 0), (1, 0, 0)),
            Plane((0, -8, 0), (0, 1, 0)),
            Plane((0, 0, -6), (0, 0, 1)),
        ], label=["a", "b", "c"], **pv_options3d)
    assert p[0].get_label(False) == "a"
    assert p[1].get_label(False) == "b"
    assert p[2].get_label(False) == "c"

    p = plot_vector([t * z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        params={t: (1, 0, 2)},
        slice=[
            Plane((-10 * t, 0, 0), (1, 0, 0)),
            Plane((0, -8, 0), (0, 1, 0)),
            Plane((0, 0, -6), (0, 0, 1)),
        ], label=["a", "b", "c"], **pv_options3d)
    s = p.backend.series
    assert s[0].get_label(False) == "a"
    assert s[1].get_label(False) == "b"
    assert s[2].get_label(False) == "c"


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_rendering_kw(pv_options, pv_options3d):
    # verify that the rendering_kw keyword argument works, if the correct
    # number of dictionaries is provided, and that it will ovveride the values
    # into quiver_kw, stream_kw, contour_kw

    x, y, z, t = symbols("x, y, z, t")

    # 2 series -> 2 dictionaries
    p = plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        contour_kw={"cmap": "viridis"}, quiver_kw={"color": "w"},
        rendering_kw=[{"cmap": "autumn"}, {"color": "k"}], **pv_options)
    assert p[0].rendering_kw == {"cmap": "autumn"}
    assert p[1].rendering_kw == {"color": "k"}

    p = plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)},
        contour_kw={"cmap": "viridis"}, quiver_kw={"color": "w"},
        rendering_kw=[{"cmap": "autumn"}, {"color": "k"}], **pv_options)
    s = p.backend.series
    assert s[0].rendering_kw == {"cmap": "autumn"}
    assert s[1].rendering_kw == {"color": "k"}

    # 1 series -> 2 dictionaries = raise error
    p = lambda: plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        scalar=False, rendering_kw=[{"cmap": "autumn"}, {"color": "k"}],
        **pv_options)
    raises(ValueError, p)

    p = lambda: plot_vector([-sin(y * z), cos(x)], (x, -3, 2), (y, -4, 6),
        params={z: (1, 0, 2)}, scalar=False,
        rendering_kw=[{"cmap": "autumn"}, {"color": "k"}], **pv_options)
    raises(ValueError, p)

    # 1 series -> 1 dictionary
    p = plot_vector([z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        slice=cos(z**2 + y**2), use_cm=False,
        rendering_kw={"color": "k"}, **pv_options3d)
    assert p[0].rendering_kw == {"color": "k"}

    p = plot_vector([t * z, y, x], (x, -2, 3), (y, -3, 1), (z, -6, 5),
        slice=cos(z**2 + y**2),
        params={t: (1, 0, 2)}, use_cm=False,
        rendering_kw={"color": "k"}, **pv_options3d)
    s = p.backend.series
    assert s[0].rendering_kw == {"color": "k"}

    # 3 series -> 3 dictionaries
    p = plot_vector([z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        slice=[
            Plane((-10, 0, 0), (1, 0, 0)),
            Plane((0, -8, 0), (0, 1, 0)),
            Plane((0, 0, -6), (0, 0, 1)),
        ], use_cm=False,
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}],
        **pv_options3d)
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "g"}
    assert p[2].rendering_kw == {"color": "b"}

    p = plot_vector([t * z, y, x], (x, -10, 9), (y, -8, 7), (z, -6, 5),
        params={t: (1, 0, 2)},
        slice=[
            Plane((-10 * t, 0, 0), (1, 0, 0)),
            Plane((0, -8, 0), (0, 1, 0)),
            Plane((0, 0, -6), (0, 0, 1)),
        ], use_cm=False,
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}],
        **pv_options3d)
    s = p.backend.series
    assert s[0].rendering_kw == {"color": "r"}
    assert s[1].rendering_kw == {"color": "g"}
    assert s[2].rendering_kw == {"color": "b"}


def test_plot_vector_lambda_functions(pv_options3d):
    # verify that plot_vector generates the correct data series when lambda
    # functions are used instead of symbolic expressions

    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j
    v2 = z * N.i + x * N.j + y * N.k

    # verify that plotting symbolic expressions and lambda functions produces
    # the same results
    p1 = plot_vector(v1, (x, -5, 5), (y, -2, 2), **pv_options3d)
    p2 = plot_vector(
        [lambda x, y: x, lambda x, y: y], ("x", -5, 5), ("y", -2, 2),
        **pv_options3d)
    assert len(p1.series) == len(p2.series) == 2
    assert all(np.allclose(t1, t2) for t1, t2 in
        zip(p1[0].get_data(), p2[0].get_data()))
    assert all(np.allclose(t1, t2) for t1, t2 in
        zip(p1[1].get_data(), p2[1].get_data()))

    # verify the use of a lambda function scalar field
    p3 = plot_vector(v1, (x, -5, 5), (y, -2, 2),
        scalar=sqrt(x**2 + y**2), **pv_options3d)
    p4 = plot_vector(
        [lambda x, y: x, lambda x, y: y], ("x", -5, 5), ("y", -2, 2),
        scalar=lambda x, y: np.sqrt(x**2 + y**2), **pv_options3d)
    assert all(np.allclose(t1, t2) for t1, t2 in
        zip(p3[0].get_data(), p4[0].get_data()))


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_parametric_ranges_2d_quiver(pv_options):
    # verify that interactive plots of 2d quivers with scalar fields,
    # are going to update both the scalar field (contour) and vector fields
    # (quivers)

    x, y, a, b, u = symbols("x, y, a, b, u")

    p = plot_vector((-cos(u*y), sin(x)),
        prange(x, -2*a, 2*b), prange(y, -2*b, 2*a),
        params={u: (1, 0, 2), a: (1, 0, 2), b: (1, 0, 2)}, **pv_options)
    d1 = p.backend[0].get_data()
    d2 = p.backend[1].get_data()
    p.backend.update_interactive({u: 1, a: 0.5, b: 1.5})
    d3 = p.backend[0].get_data()
    d4 = p.backend[1].get_data()
    for s, t in zip(d1, d3):
        assert not np.allclose(s, t)
    for s, t in zip(d2, d4):
        assert not np.allclose(s, t)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_number_discretization_points():
    # verify the different ways of setting the numbers of discretization points
    x, y, z = symbols("x, y, z")
    options = dict(show=False, backend=MB)

    p = plot_vector([-cos(y), sin(x)], (x, -10, 10), (y, -10, 10), **options)
    assert p[0].n[:2] == [100, 100]
    assert p[1].n[:2] == [25, 25]
    p = plot_vector([-cos(y), sin(x)], (x, -10, 10), (y, -10, 10),
        n=15, nc=200, **options)
    assert p[0].n[:2] == [200, 200]
    assert p[1].n[:2] == [15, 15]
    p = plot_vector([-cos(y), sin(x)], (x, -10, 10), (y, -10, 10),
        n1=15, n2=20, nc=50, **options)
    assert p[0].n[:2] == [50, 50]
    assert p[1].n[:2] == [15, 20]

    p = plot_vector(Matrix([cos(z), -cos(y), sin(x)]),
        (x, -10, 10), (y, -10, 10), (z, -10, 10), **options)
    assert p[0].n == [10, 10, 10]
    p = plot_vector(Matrix([cos(z), -cos(y), sin(x)]),
        (x, -10, 10), (y, -10, 10), (z, -10, 10), n=5, **options)
    assert p[0].n == [5, 5, 5]
    p = plot_vector(Matrix([cos(z), -cos(y), sin(x)]),
        (x, -10, 10), (y, -10, 10), (z, -10, 10), n1=5, n2=7, n3=9, **options)
    assert p[0].n == [5, 7, 9]

    p = plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
        n=8, use_cm=False, slice=[
            Plane((-10, 0, 0), (1, 0, 0)),
            Plane((0, 10, 0), (0, 2, 0)),
            Plane((0, 0, -10), (0, 0, 1))], **options)
    assert all(s.n == [8, 8, 8] for s in p.series)
