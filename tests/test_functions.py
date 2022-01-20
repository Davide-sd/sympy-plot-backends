import os
from tempfile import TemporaryDirectory, NamedTemporaryFile, mkdtemp
from sympy import (
    symbols, Piecewise, And, Eq, Interval, cos, sin, Abs,
    Sum, oo, S, Heaviside, real_root, log, I, LambertW,
    exp, sympify, pi, Ne, meijerg, sqrt, exp_polar, Integral,
    tan, Or, re
)
from sympy.testing.pytest import skip, warns
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager
from spb.functions import (
    plot_piecewise, plot, plot_list, plot3d_parametric_line,
    plot_parametric, plot3d, plot_contour, plot3d_parametric_surface,
    plot_implicit
)
from spb.series import LineOver1DRangeSeries, List2DSeries
from spb.backends.matplotlib import MB, unset_show
from pytest import raises

# use MatplotlibBackend for the tests whenever the backend is not specified
from spb.defaults import set_defaults, cfg
cfg["backend_2D"] = "matplotlib"
cfg["backend_3D"] = "matplotlib"
set_defaults(cfg)

np = import_module('numpy', catch=(RuntimeError,))

unset_show()

# NOTE:
#
# These tests are meant to verify that the plotting functions generate the
# correct data series.
# Also, legacy tests from the old sympy.plotting module are included.
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to test_series.py.
# If your issue il related to the preprocessing and generation of a
# Vector series or a Complex Series, consider adding tests to
# test_build_series. 
# If your issue is related to a particular keyword affecting a backend
# behaviour, consider adding tests to test_backends.py
#


def test_plot_list():
    # verify that plot_list creates the correct data series
    
    xx1 = np.linspace(-3, 3)
    yy1 = np.cos(xx1)
    xx2 = np.linspace(-5, 5)
    yy2 = np.sin(xx2)

    p = plot_list(xx1, yy1, backend=MB, show=False)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == ""

    p = plot_list(xx1, yy1, "test", backend=MB, show=False)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == "test"

    p = plot_list((xx1, yy1), (xx2, yy2), backend=MB, show=False)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert all(t.label == "" for t in p.series)

    p = plot_list((xx1, yy1, "cos"), (xx2, yy2, "sin"),
        backend=MB, show=False)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert p.series[0].label == "cos"
    assert p.series[1].label == "sin"


def test_process_sums():
    # verify that Sum containing infinity in its boundary, gets replaced with
    # a Sum with arbitrary big numbers instead.
    x, y = symbols("x, y")

    expr = Sum(1 / x ** y, (x, 1, oo))
    p1 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=1000, show=False)
    assert p1[0].expr.args[-1] == (x, 1, 1000)
    p2 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=500, show=False)
    assert p2[0].expr.args[-1] == (x, 1, 500)

    expr = Sum(1 / x ** y, (x, -oo, -1))
    p1 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=1000, show=False)
    assert p1[0].expr.args[-1] == (x, -1000, -1)
    p2 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=500, show=False)
    assert p2[0].expr.args[-1] == (x, -500, -1)

    expr = Sum(1 / x ** y, (x, -oo, oo))
    p1 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=1000, show=False)
    assert p1[0].expr.args[-1] == (x, -1000, 1000)
    p2 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=500, show=False)
    assert p2[0].expr.args[-1] == (x, -500, 500)


def test_plot_piecewise():
    # Verify that univariate Piecewise objects are processed in such a way to
    # create multiple series, each one with the correct range.
    # Series representing filled dots should be last in order.

    x = symbols("x")
    f = Piecewise(
        (-1, x < -1),
        (x, And(-1 <= x, x < 0)),
        (x**2, And(0 <= x, x < 1)),
        (x**3, x >= 1)
    )
    p = plot_piecewise(f, (x, -5, 5), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 10
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 4
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 6
    assert (s[0].expr == -S(1)) and np.isclose(s[0].start.real, -5) and np.isclose(s[0].end.real, -1 - 1e-06)
    assert np.allclose(s[1].list_x, -1) and np.allclose(s[1].list_y, -1) and (not s[1].is_filled)
    assert (s[2].expr == x) and np.isclose(s[2].start.real, -1 - 1e-06) and np.isclose(s[2].end.real, -1e-06)
    assert np.allclose(s[3].list_x, -1e-06) and np.allclose(s[3].list_y, -1e-06) and (not s[3].is_filled)
    assert (s[4].expr == x**2) and np.isclose(s[4].start.real, 0) and np.isclose(s[4].end.real, 1 - 1e-06)
    assert np.allclose(s[5].list_x, 1 - 1e-06) and np.allclose(s[5].list_y, 1 - 2e-06) and (not s[5].is_filled)
    assert (s[6].expr == x**3) and np.isclose(s[6].start.real, 1) and np.isclose(s[6].end.real, 5)
    assert np.allclose(s[7].list_x, -1) and np.allclose(s[7].list_y, -1) and s[7].is_filled
    assert np.allclose(s[8].list_x, 0) and np.allclose(s[8].list_y, 0) and s[8].is_filled
    assert np.allclose(s[9].list_x, 1) and np.allclose(s[9].list_y, 1) and s[9].is_filled

    f = Heaviside(x, 0).rewrite(Piecewise)
    p = plot_piecewise(f, (x, -10, 10), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 4
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 2
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 2
    assert (s[0].expr == 0) and np.isclose(s[0].start.real, -10) and np.isclose(s[0].end.real, 0)
    assert (s[1].expr == 1) and np.isclose(s[1].start.real, 1e-06) and np.isclose(s[1].end.real, 10)
    assert np.allclose(s[2].list_x, 1e-06) and np.allclose(s[2].list_y, 1) and (not s[2].is_filled)
    assert np.allclose(s[3].list_x, 0) and np.allclose(s[3].list_y, 0) and s[3].is_filled

    f = Piecewise((x, Interval(0, 1).contains(x)), (0, True))
    p = plot_piecewise(f, (x, -10, 10), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 7
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 4
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2

    f = Piecewise((x, x < 1), (x**2, (x >= -1) & (x <= 3)), (x, x > 3))
    p = plot_piecewise(f, (x, -10, 10), backend=MB, show=False)
    assert not p.legend
    s = p.series
    assert len(s) == 7
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 4
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2

    # NotImplementedError: as_set is not implemented for relationals with
    # periodic solutions
    p1 = Piecewise((cos(x), x < 0), (0, True))
    f = Piecewise((0, Eq(p1, 0)), (p1 / Abs(p1), True))
    raises(NotImplementedError, lambda: plot_piecewise(f, (x, -10, 10),
        backend=MB, show=False))

    # The range is smaller than the function "domain"
    f = Piecewise(
        (1, x < -5),
        (x, Eq(x, 0)),
        (x**2, Eq(x, 2)),
        (x**3, (x > 0) & (x < 2)),
        (x**4, True)
    )
    p = plot_piecewise(f, (x, -3, 3), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 9
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 6
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2


###############################################################################
################### PLOT - PLOT_PARAMETRIC - PLOT3D-RELATED ###################
###############################################################################

def test_plot_and_save_1():
    x, y, z = symbols("x, y, z")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        ###
        # Examples from the 'introduction' notebook
        ###
        p = plot(x, legend=True)
        p = plot(x * sin(x), x * cos(x))
        p.extend(p)
        p[0].line_color = lambda a: a
        p[1].line_color = "b"
        p.title = "Big title"
        p.xlabel = "the x axis"
        p[1].label = "straight line"
        p.legend = True
        p.aspect = (1, 1)
        p.xlim = (-15, 20)
        filename = "test_basic_options_and_colors.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p.extend(plot(x + 1))
        p.append(plot(x + 3, x ** 2)[1])
        filename = "test_plot_extend_append.png"
        p.save(os.path.join(tmpdir, filename))

        p[2] = plot(x ** 2, (x, -2, 3))
        filename = "test_plot_setitem.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x), (x, -2 * pi, 4 * pi))
        filename = "test_line_explicit.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x))
        filename = "test_line_default_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot((x ** 2, (x, -5, 5)), (x ** 3, (x, -3, 3)))
        filename = "test_line_multiple_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        raises(ValueError, lambda: plot(x, y))

        # Piecewise plots
        p = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1))
        filename = "test_plot_piecewise.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(Piecewise((x, x < 1), (x ** 2, True)), (x, -3, 3))
        filename = "test_plot_piecewise_2.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # test issue 7471
        p1 = plot(x)
        p2 = plot(3)
        p1.extend(p2)
        filename = "test_horizontal_line.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # test issue 10925
        f = Piecewise(
            (-1, x < -1),
            (x, And(-1 <= x, x < 0)),
            (x ** 2, And(0 <= x, x < 1)),
            (x ** 3, x >= 1),
        )
        p = plot(f, (x, -3, 3))
        filename = "test_plot_piecewise_3.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_2():
    x, y, z = symbols("x, y, z")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        # parametric 2d plots.
        # Single plot with default range.
        p = plot_parametric(sin(x), cos(x))
        filename = "test_parametric.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single plot with range.
        p = plot_parametric(sin(x), cos(x), (x, -5, 5), legend=True)
        filename = "test_parametric_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple plots with same range.
        p = plot_parametric((sin(x), cos(x)), (x, sin(x)))
        filename = "test_parametric_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple plots with different ranges.
        p = plot_parametric((sin(x), cos(x), (x, -3, 3)), (x, sin(x), (x, -5, 5)))
        filename = "test_parametric_multiple_ranges.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # depth of recursion specified.
        p = plot_parametric(x, sin(x), depth=13)
        filename = "test_recursion_depth.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # No adaptive sampling.
        p = plot_parametric(cos(x), sin(x), adaptive=False, n=500)
        filename = "test_adaptive.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # 3d parametric plots
        p = plot3d_parametric_line(sin(x), cos(x), x, legend=True)
        filename = "test_3d_line.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d_parametric_line(
            (sin(x), cos(x), x, (x, -5, 5)), (cos(x), sin(x), x, (x, -3, 3))
        )
        filename = "test_3d_line_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d_parametric_line(sin(x), cos(x), x, n=30)
        filename = "test_3d_line_points.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # 3d surface single plot.
        p = plot3d(x * y)
        filename = "test_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple 3D plots with same range.
        p = plot3d(-x * y, x * y, (x, -5, 5))
        filename = "test_surface_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple 3D plots with different ranges.
        p = plot3d((x * y, (x, -3, 3), (y, -3, 3)), (-x * y, (x, -3, 3), (y, -3, 3)))
        filename = "test_surface_multiple_ranges.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single Parametric 3D plot
        p = plot3d_parametric_surface(sin(x + y), cos(x - y), x - y)
        filename = "test_parametric_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Parametric 3D plots.
        p = plot3d_parametric_surface(
            (x * sin(z), x * cos(z), z, (x, -5, 5), (z, -5, 5)),
            (sin(x + y), cos(x - y), x - y, (x, -5, 5), (y, -5, 5)),
        )
        filename = "test_parametric_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single Contour plot.
        p = plot_contour(sin(x) * sin(y), (x, -5, 5), (y, -5, 5))
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Contour plots with same range.
        p = plot_contour(x ** 2 + y ** 2, x ** 3 + y ** 3, (x, -5, 5), (y, -5, 5))
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Contour plots with different range.
        p = plot_contour(
            (x ** 2 + y ** 2, (x, -5, 5), (y, -5, 5)),
            (x ** 3 + y ** 3, (x, -3, 3), (y, -3, 3)),
        )
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_4():
    x, y = symbols("x, y")

    ###
    # Examples from the 'advanced' notebook
    ###

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        i = Integral(log((sin(x) ** 2 + 1) * sqrt(x ** 2 + 1)), (x, 0, y))
        p = plot(i, (y, 1, 5))
        filename = "test_advanced_integral.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_5():
    x, y = symbols("x, y")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        s = Sum(1 / x ** y, (x, 1, oo))
        p = plot(s, (y, 2, 10), adaptive=False, only_integers=True)
        filename = "test_advanced_inf_sum.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(Sum(1 / x, (x, 1, y)), (y, 2, 10), adaptive=False,
            only_integers=True, steps=True)
        filename = "test_advanced_fin_sum.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_6():
    x = symbols("x")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        filename = "test.png"
        ###
        # Test expressions that can not be translated to np and generate complex
        # results.
        ###
        p = plot(sin(x) + I * cos(x))
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(sqrt(-x)))
        p.save(os.path.join(tmpdir, filename))
        p = plot(LambertW(x))
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(LambertW(x)))
        p.save(os.path.join(tmpdir, filename))

        # Characteristic function of a StudentT distribution with nu=10
        x1 = 5 * x ** 2 * exp_polar(-I * pi) / 2
        m1 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x1)
        x2 = 5 * x ** 2 * exp_polar(I * pi) / 2
        m2 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x2)
        expr = (m1 + m2) / (48 * pi)
        p = plot(expr, (x, 1e-6, 1e-2), adaptive=False, n=20)
        p.save(os.path.join(tmpdir, filename))


def test_issue_11461():
    x = symbols("x")

    expr = real_root((log(x / (x - 2))), 3)
    p = plot(expr, backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) > 30

    # plot_piecewise is not able to deal with ConditionSet
    raises(TypeError, lambda: plot_piecewise(expr, backend=MB, show=False))


def test_issue_11865():
    k = symbols("k", integer=True)
    f = Piecewise(
        (-I * exp(I * pi * k) / k + I * exp(-I * pi * k) / k, Ne(k, 0)),
        (2 * pi, True)
    )
    p = plot(f, backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) > 30

    p = plot_piecewise(f, backend=MB, show=False)
    assert len(p[0].get_data()[0]) > 30


def test_issue_16572():
    x = symbols("x")
    p = plot(LambertW(x), backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


def test_logplot_PR_16796():
    x = symbols("x")
    p = plot(x, (x, 0.001, 100), backend=MB, xscale="log", show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30
    assert p[0].end == 100.0
    assert p[0].start == 0.001


def test_issue_17405():
    x = symbols("x")
    f = x ** 0.3 - 10 * x ** 3 + x ** 2
    p = plot(f, (x, -10, 10), backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


def test_issue_15265():
    x = symbols("x")
    eqn = sin(x)

    p = plot(eqn, xlim=(-S.Pi, S.Pi), ylim=(-1, 1))
    p.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(-S.Pi, S.Pi))
    p.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(sympify("-3.14"), sympify("3.14")))
    p.close()

    p = plot(eqn, xlim=(sympify("-3.14"), sympify("3.14")), ylim=(-1, 1))
    p.close()

    raises(ValueError, lambda: plot(eqn, xlim=(-S.ImaginaryUnit, 1), ylim=(-1, 1)))

    raises(ValueError, lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.ImaginaryUnit)))

    raises(ValueError, lambda: plot(eqn, xlim=(S.NegativeInfinity, 1), ylim=(-1, 1)))

    raises(ValueError, lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.Infinity)))


def test_append_issue_7140():
    x = symbols("x")
    p1 = plot(x, backend=MB, show=False)
    p2 = plot(x ** 2, backend=MB, show=False)

    # append a series
    p2.append(p1[0])
    assert len(p2._series) == 2

    with raises(TypeError):
        p1.append(p2)

    with raises(TypeError):
        p1.append(p2._series)


def test_plot_limits():
    x = symbols("x")
    p = plot(x, x ** 2, (x, -10, 10), backend=MB, show=False)

    xmin, xmax = p.fig.axes[0].get_xlim()
    assert abs(xmin + 10) < 2
    assert abs(xmax - 10) < 2
    ymin, ymax = p.fig.axes[0].get_ylim()
    assert abs(ymin + 10) < 10
    assert abs(ymax - 100) < 10


def test_plot3d_parametric_line_limits():
    x = symbols("x")

    v1 = (2 * cos(x), 2 * sin(x), 2 * x, (x, -5, 5))
    v2 = (sin(x), cos(x), x, (x, -5, 5))
    p = plot3d_parametric_line(v1, v2)

    xmin, xmax = p.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = p.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = p.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2

    p = plot3d_parametric_line(v2, v1)

    xmin, xmax = p.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = p.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = p.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2



###############################################################################
################################ PLOT IMPLICIT ################################
###############################################################################


def tmp_file(dir=None, name=""):
    return NamedTemporaryFile(suffix=".png", dir=dir, delete=False).name


def plot_and_save(expr, *args, name="", dir=None, **kwargs):
    p = plot_implicit(expr, *args, **kwargs)
    p.save(tmp_file(dir=dir, name=name))
    # Close the plot to avoid a warning from matplotlib
    p.close()


def plot_implicit_tests(name):
    temp_dir = mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    x, y = symbols("x, y")
    # implicit plot tests
    plot_and_save(Eq(y, cos(x)), (x, -5, 5), (y, -2, 2),
        name=name, dir=temp_dir, adaptive=False, n=50)
    plot_and_save(
        Eq(y ** 2, x ** 3 - x), (x, -5, 5), (y, -4, 4),
        name=name, dir=temp_dir, adaptive=False, n=50)
    plot_and_save(y > 1 / x, (x, -5, 5), (y, -2, 2),
        name=name, dir=temp_dir, adaptive=False, n=50)
    plot_and_save(y < 1 / tan(x), (x, -5, 5), (y, -2, 2),
        name=name, dir=temp_dir, adaptive=False, n=50)
    plot_and_save(
        y >= 2 * sin(x) * cos(x), (x, -5, 5), (y, -2, 2),
        name=name, dir=temp_dir, adaptive=False, n=50)
    plot_and_save(y <= x ** 2, (x, -3, 3), (y, -1, 5),
        name=name, dir=temp_dir, adaptive=False, n=50)

    # Test all input args for plot_implicit
    plot_and_save(Eq(y ** 2, x ** 3 - x), (x, -5, 5), (y, -5, 5),
        dir=temp_dir, adaptive=False, n=50)
    plot_and_save(Eq(y ** 2, x ** 3 - x), (x, -5, 5), (y, -5, 5),
        adaptive=False, n=50, dir=temp_dir)
    plot_and_save(Eq(y ** 2, x ** 3 - x), (x, -5, 5), (y, -5, 5),
        adaptive=True, dir=temp_dir)
    plot_and_save(y > x, (x, -5, 5), (y, -5, 5),
        dir=temp_dir, adaptive=False, n=50)
    plot_and_save(And(y > exp(x), y > x + 2), (x, -5, 5), (y, -5, 5),
        dir=temp_dir, adaptive=True)
    plot_and_save(Or(y > x, y > -x), (x, -5, 5), (y, -5, 5), dir=temp_dir)
    plot_and_save(x ** 2 - 1, (x, -5, 5), (y, -5, 5), dir=temp_dir)
    plot_and_save(x ** 2 - 1, (x, -5, 5), (y, -5, 5), dir=temp_dir)
    plot_and_save(y > x, (x, -5, 5), (y, -5, 5),
        adaptive=True, depth=-5, dir=temp_dir)
    plot_and_save(y > x, (x, -5, 5), (y, -5, 5),
        adaptive=True, depth=5, dir=temp_dir)
    plot_and_save(y > cos(x), (x, -5, 5), (y, -5, 5), adaptive=False, dir=temp_dir)
    plot_and_save(y < cos(x), (x, -5, 5), (y, -5, 5), adaptive=False, dir=temp_dir)
    plot_and_save(And(y > cos(x), Or(y > x, Eq(y, x))), (x, -5, 5), (y, -5, 5),
        dir=temp_dir, adaptive=True)
    plot_and_save(y - cos(pi / x), (x, -5, 5), (y, -5, 5), dir=temp_dir)

    # Test plots which cannot be rendered using the adaptive algorithm
    with warns(UserWarning, match="Adaptive meshing could not be applied"):
        plot_and_save(
            Eq(y, re(cos(x) + I * sin(x))), adaptive=True, name=name, dir=temp_dir
        )

    plot_and_save(x ** 2 - 1, title="An implicit plot", dir=temp_dir)


def test_plot_implicit_matplotlib():
    matplotlib = import_module(
        "matplotlib", min_module_version="1.1.0", catch=(RuntimeError,)
    )
    if matplotlib:
        try:
            plot_implicit_tests("test")
        finally:
            TmpFileManager.cleanup()
    else:
        skip("Matplotlib not the default backend")


def test_plot_implicit_region_and():
    matplotlib = import_module(
        "matplotlib", min_module_version="1.1.0", catch=(RuntimeError,)
    )
    if not matplotlib:
        skip("Matplotlib not the default backend")

    from matplotlib.testing.compare import compare_images

    test_directory = os.path.dirname(os.path.abspath(__file__))

    try:
        temp_dir = mkdtemp()
        TmpFileManager.tmp_folder(temp_dir)

        x, y = symbols("x y")

        r1 = (x - 1) ** 2 + y ** 2 < 2
        r2 = (x + 1) ** 2 + y ** 2 < 2

        test_filename = tmp_file(dir=temp_dir, name="test_region_and")
        cmp_filename = os.path.join(test_directory, "test_region_and.png")
        p = plot_implicit(r1 & r2, (x, -5, 5), (y, -5, 5))
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_or")
        cmp_filename = os.path.join(test_directory, "test_region_or.png")
        p = plot_implicit(r1 | r2, (x, -5, 5), (y, -5, 5))
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_not")
        cmp_filename = os.path.join(test_directory, "test_region_not.png")
        p = plot_implicit(~r1, (x, -5, 5), (y, -5, 5))
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_xor")
        cmp_filename = os.path.join(test_directory, "test_region_xor.png")
        p = plot_implicit(r1 ^ r2, (x, -5, 5), (y, -5, 5))
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
    finally:
        TmpFileManager.cleanup()