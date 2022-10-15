from spb.defaults import cfg
from spb.interactive import InteractivePlot
from spb.series import (
    ComplexPointSeries, ComplexPointInteractiveSeries,
    ComplexSurfaceSeries, ComplexSurfaceInteractiveSeries,
    ComplexDomainColoringSeries, ComplexDomainColoringInteractiveSeries,
    LineOver1DRangeSeries, LineInteractiveSeries,
    InteractiveSeries, ContourSeries, ContourInteractiveSeries,
    Vector2DSeries, Vector2DInteractiveSeries,
    AbsArgLineSeries, AbsArgLineInteractiveSeries
)
from spb.utils import _plot_sympify
from spb import (
    plot_complex, plot_complex_list, plot_complex_vector,
    plot_real_imag, plot_vector, PB, MB
)
from sympy import (
    exp, symbols, I, pi, sin, cos, asin, sqrt, log,
    re, im, arg, Float
)
from sympy.external import import_module
from pytest import raises

# NOTE:
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to test_series.py.
# If your issue is related to a particular keyword affecting a backend
# behaviour, consider adding tests to test_backends.py
#


def test_plot_complex_list():
    # verify that plot_complex_list is capable of creating data
    # series according to the documented modes of operation

    x, y, z = symbols("x:z")

    # single complex number
    p = plot_complex_list(3 + 2 * I, backend=MB, show=False)
    assert isinstance(p, MB)
    assert len(p.series) == 1
    assert isinstance(p.series[0], ComplexPointSeries)

    p = plot_complex_list(x * 3 + 2 * I, params={x: (1, 0, 2)},
        backend=MB, show=False)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) == 1
    assert isinstance(p.backend.series[0], ComplexPointInteractiveSeries)

    # list of complex numbers, each one with its own label
    p = plot_complex_list((3+2*I, "a"), (5 * I, "b"), backend=MB, show=False)
    assert isinstance(p, MB)
    assert len(p.series) == 2
    assert all(isinstance(t, ComplexPointSeries) for t in p.series)

    p = plot_complex_list((3+2*I, "a"), (5 * I, "b"), params={x: (1, 0, 2)},
        backend=MB, show=False)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) == 2
    assert all(isinstance(t, ComplexPointInteractiveSeries) for t in p.backend.series)

    # lists of grouped complex numbers with labels
    p = plot_complex_list(
        [3 + 2 * I, 2 * I, 3],
        [2 + 3 * I, -2 * I, -3],
        backend=MB, show=False)
    assert isinstance(p, MB)
    assert len(p.series) == 2
    assert all(isinstance(t, ComplexPointSeries) for t in p.series)

    p = plot_complex_list(
        [3 + 2 * I, 2 * I, 3],
        [2 + 3 * I, -2 * I, -3],
        params={x: (1, 0, 2)}, backend=MB, show=False)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) == 2
    assert all(isinstance(t, ComplexPointInteractiveSeries) for t in p.backend.series)

    p = plot_complex_list(
        ([3 + 2 * I, 2 * I, 3], "a"),
        ([2 + 3 * I, -2 * I, -3], "b"),
        backend=MB, show=False)
    assert isinstance(p, MB)
    assert len(p.series) == 2
    assert all(isinstance(t, ComplexPointSeries) for t in p.series)

    p = plot_complex_list(
        ([3 + 2 * I, 2 * I, 3], "a"),
        ([2 + 3 * I, -2 * I, -3], "b"),
        params={x: (1, 0, 2)}, backend=MB, show=False)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) == 2
    assert all(isinstance(t, ComplexPointInteractiveSeries) for t in p.backend.series)


def test_plot_real_imag_1d():
    # verify that plot_real_imag is capable of creating data
    # series according to the documented modes of operation when it comes to
    # plotting lines

    x, y, z = symbols("x:z")
    xmin, xmax = cfg["plot_range"]["min"], cfg["plot_range"]["max"]

    ###########################################################################
    ### plot_real_imag(expr, range [opt], label [opt], rendering_kw [opt]) ####
    ###########################################################################

    # plot the real and imaginary part
    p = plot_real_imag(sqrt(x), backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert (s[0].var, s[0].start, s[0].end) == (x, xmin, xmax)
    assert s[0].get_label(False) == "Re(sqrt(x))"

    p = plot_real_imag(sqrt(x)**y, params={y: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert isinstance(s[0], LineInteractiveSeries)

    # same as the previous case, different range, custom label and
    # rendering keywords
    p = plot_real_imag(sqrt(x), (x, -5, 4), "f", {"color": "k"},
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert (s[0].var, s[0].start, s[0].end) == (x, -5, 4)
    assert s[0].get_label(False) == "Re(f)"
    assert s[1].get_label(False) == "Im(f)"
    assert all(ss.rendering_kw == {"color": "k"} for ss in s)

    p = plot_real_imag(sqrt(x)**y, (x, -5, 4), "f", {"color": "k"},
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert s[0].get_label(False) == "Re(f)"
    assert s[1].get_label(False) == "Im(f)"
    assert all(ss.rendering_kw == {"color": "k"} for ss in s)

    # real part of the function
    p = plot_real_imag(sqrt(x), real=True, imag=False, backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == re(sqrt(x))
    assert (s[0].var, s[0].start, s[0].end) == (x, xmin, xmax)

    p = plot_real_imag(sqrt(x)**y, real=True, imag=False,
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == re(sqrt(x)**y)

    # imaginary part of the function
    p = plot_real_imag(sqrt(x), real=False, imag=True,
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == im(sqrt(x))
    assert (s[0].var, s[0].start, s[0].end) == (x, xmin, xmax)

    p = plot_real_imag(sqrt(x)**y, real=False, imag=True,
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == im(sqrt(x)**y)

    # absolute value of the function
    p = plot_real_imag(sqrt(x), real=False, imag=False, abs=True,
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == sqrt(re(sqrt(x))**2 + im(sqrt(x))**2)
    assert (s[0].var, s[0].start, s[0].end) == (x, xmin, xmax)

    p = plot_real_imag(sqrt(x)**y, real=False, imag=False,
        abs=True, params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == sqrt(re(sqrt(x)**y)**2 + im(sqrt(x)**y)**2)

    # argument of the function
    p = plot_real_imag(sqrt(x), real=False, imag=False,
        abs=False, arg=True, backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == arg(sqrt(x))
    assert (s[0].var, s[0].start, s[0].end) == (x, xmin, xmax)

    p = plot_real_imag(sqrt(x)**y, real=False, imag=False,
        abs=False, arg=True, params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == arg(sqrt(x)**y)

    # multiple line series over a 1D real range
    p = plot_real_imag(sqrt(x), real=True, imag=True, abs=True, arg=True,
            backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    correct_exprs = set([
        re(sqrt(x)), im(sqrt(x)), sqrt(re(sqrt(x))**2 + im(sqrt(x))**2), arg(sqrt(x))
    ])
    for ser in s:
        correct_exprs = correct_exprs.difference([ser.get_expr()])
    assert len(correct_exprs) == 0
    assert all((ser.var, ser.start, ser.end) == (x, xmin, xmax) for ser in s)

    p = plot_real_imag(sqrt(x)**y, real=True, imag=True, abs=True, arg=True,
            params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, LineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)

    ###########################################################################
    ## plot_real_imag(expr1, expr2, range [opt], rend_kw [opt]) ##
    ###########################################################################

    # multiple expressions with a common range
    p = plot_real_imag(sqrt(x), asin(x), (x, -8, 8), {"color": "k"}, real=True,
        imag=True, abs=False, arg=False, backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == -8) and (ss.end == 8) for ss in s)
    assert all(ss.rendering_kw == {"color": "k"} for ss in s)

    p = plot_real_imag(sqrt(x)**y, asin(x), (x, -8, 8), {"color": "k"}, real=True,
        imag=True, abs=False, arg=False, backend=MB, show=False,
        params={y: (1, 0, 2)}, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, LineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == -8) and (ss.end == 8) for ss in s)
    assert all(ss.rendering_kw == {"color": "k"} for ss in s)

    # multiple expressions with a common unspecified range
    p = plot_real_imag(sqrt(x), asin(x), real=True, imag=True, abs=False,
            arg=False, absarg=False, backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == xmin) and (ss.end == xmax) for ss in s)

    p = plot_real_imag(sqrt(x)**y, asin(x), real=True, imag=True, abs=False,
            arg=False, absarg=False, backend=MB, show=False,
            params={y: (1, 0, 2)}, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, LineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == xmin) and (ss.end == xmax) for ss in s)

    ###########################################################################
    ## plot_real_imag((e1, r1 [opt], lbl1 [opt], rk1 [opt]),
    ##      (e2, r2 [opt], lbl2 [opt], rk2 [opt]), ...)
    ###########################################################################

    # multiple expressions each one with its label and range
    p = plot_real_imag(
        (sqrt(x), (x, -5, 5), "f", {"color": "k"}),
        (asin(x), (x, -8, 8), "g"), backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert (s[0].start == -5) and (s[0].end == 5)
    assert (s[1].start == -5) and (s[1].end == 5)
    assert (s[2].start == -8) and (s[2].end == 8)
    assert (s[3].start == -8) and (s[3].end == 8)
    assert s[0].rendering_kw == {"color": "k"}
    assert s[1].rendering_kw == {"color": "k"}
    assert s[0].get_label(False) == "Re(f)"
    assert s[1].get_label(False) == "Im(f)"
    assert s[2].get_label(False) == "Re(g)"
    assert s[3].get_label(False) == "Im(g)"

    p = plot_real_imag(
        (sqrt(x)**y, (x, -5, 5), "f"),
        (asin(x), (x, -8, 8), "g"),
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, LineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    r = list(s[0].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[1].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[2].ranges.values())[0]
    assert (r[0] == -8) and (r[-1] == 8)
    r = list(s[3].ranges.values())[0]
    assert (r[0] == -8) and (r[-1] == 8)

    # multiple expressions each one with its label and a common range
    p = plot_real_imag((sqrt(x), "f"), (asin(x), "g"), (x, -5, 5),
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((t.start == -5) and (t.end == 5) for t in s)

    p = plot_real_imag((sqrt(x)**y, "f"), (asin(x), "g"), (x, -5, 5),
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, LineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    r = list(s[0].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[1].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[2].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[3].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)

    # multiple expressions each one with its label and range + multiple kwargs
    p = plot_real_imag(
        (sqrt(x), (x, -5, 5), "f", {"color": "k"}),
        (asin(x), (x, -8, 8), "g"),
        real=True, imag=True, backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((s[i].start == -5) and (s[i].end == 5) for i in [0, 1])
    assert all((s[i].start == -8) and (s[i].end == 8) for i in [2, 3])
    assert all(s[i].rendering_kw == {"color": "k"} for i in [0, 1])
    assert all(s[i].rendering_kw == dict() for i in [2, 3])

    p = plot_real_imag(
        (sqrt(x)**y, (x, -5, 5), "f"),
        (asin(x), (x, -8, 8), "g"),
        real=True, imag=True, params={y: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, LineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((s[i].start == -5) and (s[i].end == 5) for i in [0, 1])
    assert all((s[i].start == -8) and (s[i].end == 8) for i in [2, 3])


def test_plot_real_imag_2d_3d():
    # verify that plot_real_imag is capable of creating data
    # series according to the documented modes of operation when it comes to
    # plotting surfaces or contours

    x, y, z = symbols("x:z")
    xmin, xmax = cfg["plot_range"]["min"], cfg["plot_range"]["max"]

    ###########################################################################
    ### plot_real_imag(expr, range [opt], label [opt], rendering_kw [opt]) ####
    ###########################################################################

    # default kwargs correspond to real=True, imag=True.
    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), backend=MB, show=False,
        n=5, threed=True)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(ss, ComplexSurfaceSeries) for ss in s)
    assert all(ss.is_3Dsurface for ss in s)
    assert s[0].get_label(False) == "Re(sin(z))"
    assert s[1].get_label(False) == "Im(sin(z))"

    p = plot_real_imag(sin(y * z), (z, -5-5j, 5+5j), params={y: (1, 0, 2)},
        backend=MB, show=False, n=5, threed=True)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(ss, ComplexSurfaceInteractiveSeries) for ss in s)
    assert all(ss.is_3Dsurface for ss in s)
    assert s[0].get_label(False) == "Re(sin(y*z))"
    assert s[1].get_label(False) == "Im(sin(y*z))"

    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), backend=MB, show=False,
        n=5, threed=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(ss, ComplexSurfaceSeries) for ss in s)
    assert all((not ss.is_3Dsurface) and ss.is_contour for ss in s)
    assert s[0].get_label(False) == "Re(sin(z))"
    assert s[1].get_label(False) == "Im(sin(z))"

    p = plot_real_imag(sin(y * z), (z, -5-5j, 5+5j), params={y: (1, 0, 2)},
        backend=MB, show=False, n=5, threed=False)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(ss, ComplexSurfaceInteractiveSeries) for ss in s)
    assert all((not ss.is_3Dsurface) and ss.is_contour for ss in s)
    assert s[0].get_label(False) == "Re(sin(y*z))"
    assert s[1].get_label(False) == "Im(sin(y*z))"

    # real part of the function
    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=True, imag=False,
        backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == re(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=True, imag=False,
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == re(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # imaginary part of the function
    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=False, imag=True,
        backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == im(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=False, imag=True,
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == im(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # absolute value of the function
    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=False, imag=False,
        abs=True, backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == sqrt(re(sin(z))**2 + im(sin(z))**2)
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=False, imag=False,
        abs=True, params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == sqrt(re(sin(z))**2 + im(sin(z))**2)
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # argument of the function
    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=False, imag=False,
        abs=False, arg=True, backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == arg(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=False, imag=False,
        abs=False, arg=True, params={y: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == arg(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # multiple 2D plots (contours) of a complex function over a complex range
    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=True, imag=True,
        abs=True, arg=True, threed=False, backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(ss, ComplexSurfaceSeries) for ss in s)
    assert all((not t.is_3Dsurface) and t.is_contour for t in s)

    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=True, imag=True,
        abs=True, arg=True, threed=False, params={y: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, ComplexSurfaceInteractiveSeries) for t in s)
    assert all((not t.is_3Dsurface) and t.is_contour for t in s)

    # multiple 3D plots (surfaces) of a complex function over a complex range
    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=True, imag=True,
        abs=True, arg=True, threed=True, backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 4
    assert all(isinstance(t, ComplexSurfaceSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    p = plot_real_imag(sin(z), (z, -5-5j, 5+5j), real=True, imag=True,
        abs=True, arg=True, threed=True, params={y: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 4
    assert all(isinstance(t, ComplexSurfaceInteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    ###########################################################################
    ## plot_real_imag(e1, e2, range [opt], label [opt], rendering_kw [opt]) ###
    ###########################################################################

    # multiple 3D plots (surfaces) of a complex function over a complex range
    p = plot_real_imag(sin(z), cos(z), (z, -5-5j, 5+5j),
        real=True, imag=True, abs=True, arg=True, threed=True,
        backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 8
    assert all(isinstance(t, ComplexSurfaceSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    p = plot_real_imag(sin(z), cos(z), (z, -5-5j, 5+5j),
        real=True, imag=True, abs=True, arg=True, threed=True,
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 8
    assert all(isinstance(t, ComplexSurfaceInteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    ###########################################################################
    ## plot_real_imag((e1, r1 [opt], lbl1 [opt], rkw1 [opt]),
    ##       (e2, r2 [opt], lbl2 [opt], rkw2 [opt]))
    ###########################################################################

    p = plot_real_imag(
        (sin(z), (z, -5-5j, 5+5j), "a"),
        (cos(z), (z, -4-3j, 2+1j), {"color": "k"}),
        real=True, imag=True, abs=True, arg=True, threed=True,
        backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 8
    assert all(isinstance(t, ComplexSurfaceSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)
    assert all((s[i].var, s[i].start, s[i].end) == (z, -5-5j, 5+5j) for i in range(4))
    assert all((s[i].var, s[i].start, s[i].end) == (z, -4-3j, 2+1j) for i in range(4, 8))
    assert all(s[i].rendering_kw == dict() for i in range(4))
    assert all(s[i].rendering_kw == {"color": "k"} for i in range(4, 8))
    correct_labels = set(['Re(a)', 'Im(a)', 'Abs(a)', 'Arg(a)'])
    for i in range(4):
        correct_labels = correct_labels.difference([s[i].get_label(False)])
    assert len(correct_labels) == 0

    p = plot_real_imag(
        (sin(z), (z, -5-5j, 5+5j), "a"),
        (cos(z), (z, -4-3j, 2+1j), {"color": "k"}),
        absarg=True, real=True, imag=True,
        abs=True, arg=True, threed=True, params={y: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 8
    assert all(isinstance(t, ComplexSurfaceInteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)
    assert all((s[i].var, s[i].start, s[i].end) == (z, -5-5j, 5+5j) for i in range(4))
    assert all((s[i].var, s[i].start, s[i].end) == (z, -4-3j, 2+1j) for i in range(4, 8))
    assert all(s[i].rendering_kw == dict() for i in range(4))
    assert all(s[i].rendering_kw == {"color": "k"} for i in range(4, 8))
    correct_labels = set(['Re(a)', 'Im(a)', 'Abs(a)', 'Arg(a)'])
    for i in range(4):
        correct_labels = correct_labels.difference([s[i].get_label(False)])
    assert len(correct_labels) == 0


def test_plot_complex_1d():
    # verify that plot_complex is capable of creating data
    # series according to the documented modes of operation when it comes to
    # plotting lines

    x, y, z = symbols("x:z")
    xmin, xmax = cfg["plot_range"]["min"], cfg["plot_range"]["max"]

    ###########################################################################
    ### plot_complex(expr, range [opt], label [opt], rendering_kw [opt]) ####
    ###########################################################################

    p = plot_complex(sqrt(x), backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], AbsArgLineSeries)
    assert (s[0].var, s[0].start, s[0].end) == (x, xmin, xmax)
    assert s[0].get_label(False) == "Arg(sqrt(x))"

    p = plot_complex(sqrt(x)**y, params={y: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], AbsArgLineInteractiveSeries)
    assert s[0].get_label(False) == "Arg(x**(y/2))"

    # same as the previous case, different range, custom label and
    # rendering keywords
    p = plot_complex(sqrt(x), (x, -5, 4), "f", {"color": "k"},
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], AbsArgLineSeries)
    assert (s[0].var, s[0].start, s[0].end) == (x, -5, 4)
    assert s[0].get_label(False) == "Arg(f)"
    assert s[0].rendering_kw == {"color": "k"}

    p = plot_complex(sqrt(x)**y, (x, -5, 4), "f", {"color": "k"},
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], AbsArgLineInteractiveSeries)
    assert s[0].get_label(False) == "Arg(f)"
    assert s[0].rendering_kw == {"color": "k"}

    ###########################################################################
    ## plot_complex(expr1, expr2, range [opt], rend_kw [opt]) ##
    ###########################################################################

    # multiple expressions with a common range
    p = plot_complex(sqrt(x), asin(x), (x, -8, 8), {"color": "k"},
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == -8) and (ss.end == 8) for ss in s)
    assert all(ss.rendering_kw == {"color": "k"} for ss in s)

    p = plot_complex(sqrt(x)**y, asin(x), (x, -8, 8), {"color": "k"},
        backend=MB, show=False, params={y: (1, 0, 2)}, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == -8) and (ss.end == 8) for ss in s)
    assert all(ss.rendering_kw == {"color": "k"} for ss in s)

    # multiple expressions with a common unspecified range
    p = plot_complex(sqrt(x), asin(x), backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == xmin) and (ss.end == xmax) for ss in s)

    p = plot_complex(sqrt(x)**y, asin(x), real=True, imag=True, abs=False,
            arg=False, absarg=False, backend=MB, show=False,
            params={y: (1, 0, 2)}, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == xmin) and (ss.end == xmax) for ss in s)

    ###########################################################################
    ## plot_complex((e1, r1 [opt], lbl1 [opt], rk1 [opt]),
    ##      (e2, r2 [opt], lbl2 [opt], rk2 [opt]), ...)
    ###########################################################################

    # multiple expressions each one with its label and range
    p = plot_complex(
        (sqrt(x), (x, -5, 5), "f", {"color": "k"}),
        (asin(x), (x, -8, 8), "g"), backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert (s[0].start == -5) and (s[0].end == 5)
    assert (s[1].start == -8) and (s[1].end == 8)
    assert s[0].rendering_kw == {"color": "k"}
    assert s[1].rendering_kw == dict()
    assert s[0].get_label(False) == "Arg(f)"
    assert s[1].get_label(False) == "Arg(g)"

    p = plot_complex(
        (sqrt(x)**y, (x, -5, 5), "f", {"color": "k"}),
        (asin(x), (x, -8, 8), "g"),
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    r = list(s[0].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[1].ranges.values())[0]
    assert (r[0] == -8) and (r[-1] == 8)
    assert s[0].rendering_kw == {"color": "k"}
    assert s[1].rendering_kw == dict()
    assert s[0].get_label(False) == "Arg(f)"
    assert s[1].get_label(False) == "Arg(g)"

    # multiple expressions each one with its label and a common range
    p = plot_complex((sqrt(x), "f"), (asin(x), "g"), (x, -5, 5),
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((t.start == -5) and (t.end == 5) for t in s)
    assert s[0].get_label(False) == "Arg(f)"
    assert s[1].get_label(False) == "Arg(g)"

    p = plot_complex((sqrt(x)**y, "f"), (asin(x), "g"), (x, -5, 5),
        params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    r = list(s[0].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[1].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    assert s[0].get_label(False) == "Arg(f)"
    assert s[1].get_label(False) == "Arg(g)"


def test_plot_complex_2d_3d():
    # verify that plot_complex is capable of creating data
    # series according to the documented modes of operation when it comes to
    # plotting surfaces or contours

    x, y, z = symbols("x:z")
    xmin, xmax = cfg["plot_range"]["min"], cfg["plot_range"]["max"]

    ###########################################################################
    ### plot_complex(expr, range [opt], label [opt], rendering_kw [opt]) ####
    ###########################################################################

    p = plot_complex(sin(z), (z, -5-5j, 5+5j), backend=MB, show=False,
        n=5, threed=True)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], ComplexDomainColoringSeries)
    assert s[0].is_3Dsurface
    assert s[0].get_label(False) == "sin(z)"

    p = plot_complex(sin(y * z), (z, -5-5j, 5+5j), params={y: (1, 0, 2)},
        backend=MB, show=False, n=5, threed=True)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], ComplexDomainColoringInteractiveSeries)
    assert s[0].is_3Dsurface
    assert s[0].get_label(False) == "sin(y*z)"

    p = plot_complex(sin(z), (z, -5-5j, 5+5j), backend=MB, show=False,
        n=5, threed=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], ComplexDomainColoringSeries)
    assert (not s[0].is_3Dsurface) and s[0].is_domain_coloring
    assert s[0].get_label(False) == "sin(z)"

    p = plot_complex(sin(y * z), (z, -5-5j, 5+5j), params={y: (1, 0, 2)},
        backend=MB, show=False, n=5, threed=False)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], ComplexDomainColoringInteractiveSeries)
    assert (not s[0].is_3Dsurface) and s[0].is_domain_coloring
    assert s[0].get_label(False) == "sin(y*z)"

    ###########################################################################
    ## plot_complex(e1, e2, range [opt], label [opt], rendering_kw [opt]) ###
    ###########################################################################

    # multiple 2D plots (domain) of a complex function over a complex range
    p = plot_complex(sin(z), cos(z), (z, -5-5j, 5+5j),
        threed=False, backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, ComplexDomainColoringSeries) for t in s)
    assert all((not t.is_3Dsurface) and t.is_domain_coloring for t in s)

    p = plot_complex(sin(y * z), cos(z), (z, -5-5j, 5+5j),
        threed=False, params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, ComplexDomainColoringInteractiveSeries) for t in s)
    assert all((not t.is_3Dsurface) and t.is_domain_coloring for t in s)

    # multiple 3D plots (surfaces) of a complex function over a complex range
    p = plot_complex(sin(z), cos(z), (z, -5-5j, 5+5j),
        threed=True, backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, ComplexDomainColoringSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    p = plot_complex(sin(y * z), cos(z), (z, -5-5j, 5+5j),
        threed=True, params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, ComplexDomainColoringInteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    ###########################################################################
    ## plot_complex((e1, r1 [opt], lbl1 [opt], rkw1 [opt]),
    ##       (e2, r2 [opt], lbl2 [opt], rkw2 [opt]))
    ###########################################################################

    p = plot_complex(
        (sin(z), (z, -5-5j, 5+5j), "a"),
        (cos(z), (z, -4-3j, 2+1j), {"color": "k"}),
        threed=True, backend=MB, show=False, n=5)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, ComplexDomainColoringSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)
    assert (s[0].var, s[0].start, s[0].end) == (z, -5-5j, 5+5j)
    assert (s[1].var, s[1].start, s[1].end) == (z, -4-3j, 2+1j)
    assert s[0].rendering_kw == dict()
    assert s[1].rendering_kw == {"color": "k"}
    correct_labels = set(['a', 'cos(z)'])
    for i in range(2):
        correct_labels = correct_labels.difference([s[i].get_label(False)])
    assert len(correct_labels) == 0

    p = plot_complex(
        (sin(y * z), (z, -5-5j, 5+5j), "a"),
        (cos(z), (z, -4-3j, 2+1j), {"color": "k"}),
        threed=True, params={y: (1, 0, 2)}, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, ComplexDomainColoringInteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)
    assert (s[0].var, s[0].start, s[0].end) == (z, -5-5j, 5+5j)
    assert (s[1].var, s[1].start, s[1].end) == (z, -4-3j, 2+1j)
    assert s[0].rendering_kw == dict()
    assert s[1].rendering_kw == {"color": "k"}
    correct_labels = set(['a', 'cos(z)'])
    for i in range(2):
        correct_labels = correct_labels.difference([s[i].get_label(False)])
    assert len(correct_labels) == 0


def test_plot_complex_vector():
    # verify that plot_complex_vector is capable of creating data
    # series according to the documented modes of operation

    x, z = symbols("x, z")

    ###########################################################################
    ########## plot_complex_vector(expr, range [opt], label [opt]) ############
    ###########################################################################

    p = plot_complex_vector(z**2, (z, -5 - 2j, 4 + 3j), backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries)
    assert s[0].get_label(False) == "Magnitude"
    assert (s[0].start_x, s[0].end_x) == (-5, 4)
    assert (s[0].start_y, s[0].end_y) == (-2, 3)
    assert isinstance(s[1], Vector2DSeries)
    assert s[1].ranges[0][1:] == (-5, 4)
    assert s[1].ranges[1][1:] == (-2, 3)

    p = plot_complex_vector(z**x, (z, -5 - 2j, 4 + 3j), params={x: (1, 0, 2)},
        backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert isinstance(s[0], ContourInteractiveSeries)
    assert s[0].get_label(False) == "Magnitude"
    xx, yy = s[0].ranges.values()
    assert (xx.min(), xx.max()) == (-5, 4)
    assert (yy.min(), yy.max()) == (-2, 3)
    assert isinstance(s[1], Vector2DInteractiveSeries)
    xx, yy = s[1].ranges.values()
    assert (xx.min(), xx.max()) == (-5, 4)
    assert (yy.min(), yy.max()) == (-2, 3)

    p = plot_complex_vector(z**2, (z, -5 - 2j, 4 + 3j), "test", scalar=False,
        backend=MB, show=False)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 1
    assert isinstance(s[0], Vector2DSeries)
    assert s[0].ranges[0][1:] == (-5, 4)
    assert s[0].ranges[1][1:] == (-2, 3)
    assert s[0].get_label(False) == "test"

    p = plot_complex_vector(z**x, (z, -5 - 2j, 4 + 3j), "test",
        params={x: (1, 0, 2)}, scalar=False, backend=MB, show=False, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 1
    assert isinstance(s[0], Vector2DInteractiveSeries)
    xx, yy = s[0].ranges.values()
    assert (xx.min(), xx.max()) == (-5, 4)
    assert (yy.min(), yy.max()) == (-2, 3)
    assert s[0].get_label(False) == "test"

    ###########################################################################
    # plot_complex_vector(
    #    (e1, range1 [opt], lbl1 [opt], rkw1 [opt]),
    #    (e2, range2 [opt], lbl2 [opt], rkw1 [opt]))
    ###########################################################################

    p = plot_complex_vector(
        (z**2, (z, -5 - 2j, 0 + 3j)),
        (z**3, (z, 0 - 2j, 4 + 3j), "test"),
        backend=MB, show=False, use_cm=True)
    s = p.series
    assert isinstance(p, MB)
    assert len(s) == 2
    assert all(isinstance(t, Vector2DSeries) for t in s)
    assert s[0].ranges[0][1:] == (-5, 0)
    assert s[0].ranges[1][1:] == (-2, 3)
    assert s[1].ranges[0][1:] == (0, 4)
    assert s[1].ranges[1][1:] == (-2, 3)
    assert s[0].get_label(False) == 'z**2'
    assert s[1].get_label(False) == 'test'

    p = plot_complex_vector(
        (z**x, (z, -5 - 2j, 0 + 3j)),
        (z**3, (z, 0 - 2j, 4 + 3j), "test"),
        params={x: (1, 0, 2)}, backend=MB, show=False, use_cm=True, n=5)
    s = p.backend.series
    assert isinstance(p, InteractivePlot)
    assert len(s) == 2
    assert all(isinstance(t, Vector2DInteractiveSeries) for t in s)
    xx, yy = s[0].ranges.values()
    assert (xx.min(), xx.max()) == (-5, 0)
    assert (yy.min(), yy.max()) == (-2, 3)
    xx, yy = s[1].ranges.values()
    assert (xx.min(), xx.max()) == (0, 4)
    assert (yy.min(), yy.max()) == (-2, 3)
    assert s[0].get_label(False) == 'z**x'
    assert s[1].get_label(False) == 'test'


def test_issue_6():
    phi = symbols('phi', real=True)
    vec = cos(phi) + cos(phi - 2 * pi / 3) * exp(I * 2 * pi / 3) + cos(phi - 4 * pi / 3) * exp(I * 4 * pi / 3)
    p = plot_real_imag(vec, (phi, 0, 2 * pi), real=True, imag=True,
        backend=MB, show=False)
    s = p.series
    assert len(s) == 2
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert s[0].expr == re(vec)
    assert s[1].expr == im(vec)


def test_lambda_functions():
    # verify that plotting functions raises errors if they do not support
    # lambda functions.

    np = import_module('numpy', catch=(RuntimeError,))

    p = plot_complex(lambda x: np.cos(x) + sin(x * 1j), backend=MB, show=False)
    assert len(p.series) == 1
    assert isinstance(p[0], AbsArgLineSeries)
    assert callable(p[0].get_expr())

    raises(TypeError, lambda : plot_real_imag(lambda t: t))
    raises(TypeError, lambda : plot_complex_list(lambda t: t))
    raises(TypeError, lambda : plot_complex_vector(lambda t: t))


def test_plot_real_imag_ambiguity_1():
    # the following expression used to create ambiguities in the creation
    # of the data series, then raising an error. Verify that everything
    # works correctly.

    γ = symbols('γ', real=True, positive=True)
    expr = -γ + Float('2.1393924371185387')*sqrt(-γ + Float('0.87393489185055717')*(γ + Float('-0.14999999999999999'))**2) + Float('0.29425000000000001') + Float('1.14425')*(Float('0.9348448490795449')*γ - sqrt(-γ + Float('0.87393489185055717')*(γ + Float('-0.14999999999999999'))**2) + Float('-0.14022672736193173'))**2/γ
    p = plot_real_imag(expr, "expr", adaptive=False, n=5,
        backend=PB, show=False)
    assert len(p.series) == 2
    assert all(isinstance(t, LineOver1DRangeSeries) for t in p.series)


def test_plot_real_imag_expression_order():
    # plot_real_imag creates a variable number of data series depending on the
    # value of the keyword arguments. For example, 1 series for the real part,
    # 1 for the imaginary part, ...
    # Verify that data series are generated with a given order: this will
    # guarantee a consistent user experience, especially considering the
    # label keyword argument.

    x = symbols("x")
    p = plot_real_imag(sqrt(x), (x, -4, 4),
        imag=True, real=True, arg=True, abs=True,
        backend=MB, show=False, n=5)
    assert len(p.series) == 4
    assert p[0].get_expr() == re(sqrt(x))
    assert p[1].get_expr() == im(sqrt(x))
    assert p[2].get_expr() == sqrt(re(sqrt(x))**2 + im(sqrt(x))**2)
    assert p[3].get_expr() == arg(sqrt(x))


def test_plot_real_imag_1d_label_kw():
    # verify that the label keyword argument works, if the correct
    # number of labels is provided.

    x, t = symbols("x, t")

    # one expression -> 2 series -> 2 labels
    p = plot_real_imag(sqrt(x), (x, -4, 4), backend=MB, n=5, show=False,
        label=["re(f)", "im(f)"])
    assert len(p.series) == 2
    assert p[0].get_label(False) == "re(f)"
    assert p[1].get_label(False) == "im(f)"

    p = plot_real_imag(sqrt(x)**t, (x, -4, 4), backend=MB, n=5, show=False,
        label=["re(f)", "im(f)"], params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 2
    assert s[0].get_label(False) == "re(f)"
    assert s[1].get_label(False) == "im(f)"

    # one expression -> 4 series -> 4 labels
    p = plot_real_imag(sqrt(x), (x, -4, 4), backend=MB, n=5, show=False,
        abs=True, arg=True, label=["re(f)", "im(f)", "a", "b"])
    assert len(p.series) == 4
    assert p[0].get_label(False) == "re(f)"
    assert p[1].get_label(False) == "im(f)"
    assert p[2].get_label(False) == "a"
    assert p[3].get_label(False) == "b"

    p = plot_real_imag(sqrt(x)**t, (x, -4, 4), backend=MB, n=5, show=False,
        abs=True, arg=True, label=["re(f)", "im(f)", "a", "b"],
        params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 4
    assert s[0].get_label(False) == "re(f)"
    assert s[1].get_label(False) == "im(f)"
    assert s[2].get_label(False) == "a"
    assert s[3].get_label(False) == "b"

    # one expression -> 2 series -> 3 labels -> raise error
    p = lambda: plot_real_imag(sqrt(x), (x, -4, 4), backend=MB, n=5, show=False,
        label=["re(f)", "im(f)", "c"])
    raises(ValueError, p)

    p = lambda: plot_real_imag(sqrt(x)**t, (x, -4, 4), backend=MB, n=5, show=False,
        label=["re(f)", "im(f)", "c"], params={t: (1, 0, 2)})
    raises(ValueError, p)

    # two expression -> 4 series -> 4 labels
    p = plot_real_imag(sqrt(x), log(x), (x, -4, 4), backend=MB, n=5, show=False,
        label=["re(f)", "im(f)", "re(g)", "im(g)"])
    assert len(p.series) == 4
    assert p[0].get_label(False) == "re(f)"
    assert p[1].get_label(False) == "im(f)"
    assert p[2].get_label(False) == "re(g)"
    assert p[3].get_label(False) == "im(g)"

    p = plot_real_imag(sqrt(x)**t, log(x)**t, (x, -4, 4), backend=MB, n=5,
        show=False, label=["re(f)", "im(f)", "re(g)", "im(g)"],
        params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 4
    assert s[0].get_label(False) == "re(f)"
    assert s[1].get_label(False) == "im(f)"
    assert s[2].get_label(False) == "re(g)"
    assert s[3].get_label(False) == "im(g)"


def test_plot_real_imag_2d_3d_label_kw():
    # verify that the label keyword argument works, if the correct
    # number of labels is provided.

    x, t = symbols("x, t")

    # one expression -> 2 series -> 2 labels
    p = plot_real_imag(sqrt(x), (x, -4-4j, 4+4j), backend=MB, n=5, show=False,
        label=["re(f)", "im(f)"])
    assert len(p.series) == 2
    assert p[0].get_label(False) == "re(f)"
    assert p[1].get_label(False) == "im(f)"

    p = plot_real_imag(sqrt(x)**t, (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, label=["re(f)", "im(f)"], params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 2
    assert s[0].get_label(False) == "re(f)"
    assert s[1].get_label(False) == "im(f)"

    # one expression -> 4 series -> 4 labels
    p = plot_real_imag(sqrt(x), (x, -4-4j, 4+4j), backend=MB, n=5, show=False,
        abs=True, arg=True, label=["re(f)", "im(f)", "a", "b"])
    assert len(p.series) == 4
    assert p[0].get_label(False) == "re(f)"
    assert p[1].get_label(False) == "im(f)"
    assert p[2].get_label(False) == "a"
    assert p[3].get_label(False) == "b"

    p = plot_real_imag(sqrt(x)**t, (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, abs=True, arg=True,
        label=["re(f)", "im(f)", "a", "b"], params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 4
    assert s[0].get_label(False) == "re(f)"
    assert s[1].get_label(False) == "im(f)"
    assert s[2].get_label(False) == "a"
    assert s[3].get_label(False) == "b"

    # one expression -> 2 series -> 3 labels -> raise error
    p = lambda: plot_real_imag(sqrt(x), (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, label=["re(f)", "im(f)", "c"])
    raises(ValueError, p)

    p = lambda: plot_real_imag(sqrt(x)**t, (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, label=["re(f)", "im(f)", "c"], params={t: (1, 0, 2)})
    raises(ValueError, p)

    # two expression -> 4 series -> 4 labels
    p = plot_real_imag(sqrt(x), log(x), (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, label=["re(f)", "im(f)", "re(g)", "im(g)"])
    assert len(p.series) == 4
    assert p[0].get_label(False) == "re(f)"
    assert p[1].get_label(False) == "im(f)"
    assert p[2].get_label(False) == "re(g)"
    assert p[3].get_label(False) == "im(g)"

    p = plot_real_imag(sqrt(x)**t, log(x)**t, (x, -4-4j, 4+4j), backend=MB,
        n=5, show=False, label=["re(f)", "im(f)", "re(g)", "im(g)"],
        params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 4
    assert s[0].get_label(False) == "re(f)"
    assert s[1].get_label(False) == "im(f)"
    assert s[2].get_label(False) == "re(g)"
    assert s[3].get_label(False) == "im(g)"


def test_plot_complex_1d_label_kw():
    # verify that the label keyword argument works, if the correct
    # number of labels is provided.

    x, t = symbols("x, t")

    # one series -> one label
    p = plot_complex(cos(x) + sin(I * x), (x, -2, 2),
        backend=MB, show=False, label="f")
    assert len(p.series) == 1
    assert p[0].get_label(False) == "f"

    p = plot_complex(cos(x) + sin(I * x * t), (x, -2, 2),
        backend=MB, show=False, label="f", params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 1
    assert s[0].get_label(False) == "f"

    # one series -> 2 labels -> raise error
    p = lambda: plot_complex(cos(x) + sin(I * x), (x, -2, 2),
        backend=MB, show=False, label=["f", "g"])
    raises(ValueError, p)

    p = lambda: plot_complex(cos(x) + sin(I * x * t), (x, -2, 2),
        backend=MB, show=False, label=["f", "g"], params={t: (1, 0, 2)})
    raises(ValueError, p)

    # two series -> 2 labels
    p = plot_complex(cos(x) + sin(I * x), exp(I * x) * I * sin(x), (x, -2, 2),
        backend=MB, show=False, label=["f", "g"])
    assert len(p.series) == 2
    assert p[0].get_label(False) == "f"
    assert p[1].get_label(False) == "g"

    p = plot_complex(
        cos(x) + sin(I * x * t), exp(I * x) * I * sin(t * x), (x, -2, 2),
        backend=MB, show=False, label=["f", "g"], params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 2
    assert s[0].get_label(False) == "f"
    assert s[1].get_label(False) == "g"


def test_plot_complex_2d_3d_label_kw():
    # verify that the label keyword argument works, if the correct
    # number of labels is provided.

    x, t = symbols("x, t")

    # one series -> one label
    p = plot_complex(sin(x), (x, -2-2j, 2+2j),
        backend=MB, show=False, label="f")
    assert len(p.series) == 1
    assert p[0].get_label(False) == "f"

    p = plot_complex(sin(t * x), (x, -2-2j, 2+2j),
        backend=MB, show=False, label="f", params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 1
    assert s[0].get_label(False) == "f"

    # one series -> 2 labels -> raise error
    p = lambda: plot_complex(sin(x), (x, -2-2j, 2+2j),
        backend=MB, show=False, label=["f", "g"])
    raises(ValueError, p)

    p = lambda: plot_complex(sin(t * x), (x, -2-2j, 2+2j),
        backend=MB, show=False, label=["f", "g"], params={t: (1, 0, 2)})
    raises(ValueError, p)

    # two series -> 2 labels
    p = plot_complex((sin(x), (x, -2-2j, 2j)), (cos(x), (x, -2j, 2+2j)),
        backend=MB, show=False, label=["f", "g"])
    assert len(p.series) == 2
    assert p[0].get_label(False) == "f"
    assert p[1].get_label(False) == "g"

    p = plot_complex(
        (sin(t * x), (x, -2-2j, 2j)),
        (cos(t * x), (x, -2j, 2+2j)),
        backend=MB, show=False, label=["f", "g"], params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 2
    assert s[0].get_label(False) == "f"
    assert s[1].get_label(False) == "g"


def test_plot_complex_list_label_kw():
    # verify that the label keyword argument works, if the correct
    # number of labels is provided.

    x = symbols("x")

    # single complex number -> 1 label
    p = plot_complex_list(3 + 2 * I, backend=MB, show=False, label="f")
    assert len(p.series) == 1
    assert p[0].get_label(False) == "f"

    p = plot_complex_list(x * 3 + 2 * I, params={x: (1, 0, 2)},
        backend=MB, show=False, label="f")
    s = p.backend.series
    assert len(s) == 1
    assert s[0].get_label(False) == "f"

    # 2 complex numbers -> 2 labels
    p = plot_complex_list(3 + 2 * I, 5 - 4 * I, backend=MB, show=False,
        label=["f", "g"])
    assert len(p.series) == 2
    assert p[0].get_label(False) == "f"
    assert p[1].get_label(False) == "g"

    p = plot_complex_list(x * 3 + 2 * I, 5 * x - 4 * I, params={x: (1, 0, 2)},
        backend=MB, show=False, label=["f", "g"])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].get_label(False) == "f"
    assert s[1].get_label(False) == "g"

    # 2 complex numbers -> 1 labels -> raise error
    p = lambda: plot_complex_list(3 + 2 * I, 5 - 4 * I, backend=MB, show=False,
        label="f")
    raises(ValueError, p)

    p = lambda: plot_complex_list(
        x * 3 + 2 * I, 5 * x - 4 * I, params={x: (1, 0, 2)},
        backend=MB, show=False, label="f")
    raises(ValueError, p)

    # 2 lists of grouped complex numbers -> 2 labels
    p = plot_complex_list(
        [3 + 2 * I, 2 * I, 3],
        [2 + 3 * I, -2 * I, -3],
        backend=MB, show=False, label=["f", "g"])
    assert len(p.series) == 2
    assert p[0].get_label(False) == "f"
    assert p[1].get_label(False) == "g"

    p = plot_complex_list(
        [3 * x + 2 * I, 2 * I, 3],
        [2 + 3 * I, -2 * I, -3],
        params={x: (1, 0, 2)}, backend=MB, show=False, label=["f", "g"])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].get_label(False) == "f"
    assert s[1].get_label(False) == "g"


def plot_complex_vector_label_kw():
    # verify that the label keyword argument works, if the correct
    # number of labels is provided.

    x, t = symbols("x, t")

    # one expression -> 2 series -> 2 labels
    p = plot_complex_vector(x**2 + sin(x), (x, -5-5j, 5+5j),
        backend=MB, show=False, label=["a", "b"])
    assert len(p.series) == 2
    assert p[0].get_label(False) == "a"
    assert p[1].get_label(False) == "b"

    p = plot_complex_vector(x**2 + sin(t * x), (x, -5-5j, 5+5j),
        backend=MB, show=False, label=["a", "b"], params={t: (1, 0, 2)})
    s = p.backend.series
    assert len(s) == 2
    assert s[0].get_label(False) == "a"
    assert s[1].get_label(False) == "b"

    # one expression -> 1 series -> 2 labels -> raise error
    p = lambda: plot_complex_vector(x**2 + sin(x), (x, -5-5j, 5+5j),
        backend=MB, show=False, scalar=False, label=["a", "b"])
    raises(ValueError, p)

    p = lambda: plot_complex_vector(x**2 + sin(t * x), (x, -5-5j, 5+5j),
        backend=MB, show=False, scalar=False, label=["a", "b"],
        params={t: (1, 0, 2)})
    raises(ValueError, p)


def test_plot_real_imag_1d_rendering_kw():
    # verify that the rendering_kw keyword argument works, if the correct
    # number of dictionaries is provided.

    x, t = symbols("x, t")

    # one expression -> 2 series -> 2 dictionaries
    p = plot_real_imag(sqrt(x), (x, -4, 4), backend=MB, n=5, show=False,
        rendering_kw=[{"color": "r"}, {"color": "g"}])
    assert len(p.series) == 2
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "g"}

    p = plot_real_imag(sqrt(x)**t, (x, -4, 4), backend=MB, n=5, show=False,
        params={t: (1, 0, 2)}, rendering_kw=[{"color": "r"}, {"color": "g"}])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].rendering_kw == {"color": "r"}
    assert s[1].rendering_kw == {"color": "g"}

    # one expression -> 4 series -> 4 dictionaries
    p = plot_real_imag(sqrt(x), (x, -4, 4), backend=MB, n=5, show=False,
        abs=True, arg=True,
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}, {"color": "k"}])
    assert len(p.series) == 4
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "g"}
    assert p[2].rendering_kw == {"color": "b"}
    assert p[3].rendering_kw == {"color": "k"}

    p = plot_real_imag(sqrt(x)**t, (x, -4, 4), backend=MB, n=5, show=False,
        abs=True, arg=True, params={t: (1, 0, 2)},
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}, {"color": "k"}])
    s = p.backend.series
    assert len(s) == 4
    assert s[0].rendering_kw == {"color": "r"}
    assert s[1].rendering_kw == {"color": "g"}
    assert s[2].rendering_kw == {"color": "b"}
    assert s[3].rendering_kw == {"color": "k"}

    # one expression -> 2 series -> 3 dictionaries -> raise error
    p = lambda: plot_real_imag(sqrt(x), (x, -4, 4), backend=MB, n=5, show=False,
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}])
    raises(ValueError, p)

    p = lambda: plot_real_imag(sqrt(x)**t, (x, -4, 4), backend=MB, n=5,
        show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}])
    raises(ValueError, p)

    # two expression -> 4 series -> 4 dictionaries
    p = plot_real_imag(sqrt(x), log(x), (x, -4, 4), backend=MB, n=5, show=False,
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}, {"color": "k"}])
    assert len(p.series) == 4
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "g"}
    assert p[2].rendering_kw == {"color": "b"}
    assert p[3].rendering_kw == {"color": "k"}

    p = plot_real_imag(sqrt(x)**t, log(x)**t, (x, -4, 4), backend=MB, n=5,
        show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"color": "r"}, {"color": "g"}, {"color": "b"}, {"color": "k"}])
    s = p.backend.series
    assert len(s) == 4
    assert s[0].rendering_kw == {"color": "r"}
    assert s[1].rendering_kw == {"color": "g"}
    assert s[2].rendering_kw == {"color": "b"}
    assert s[3].rendering_kw == {"color": "k"}


def test_plot_real_imag_2d_3d_rendering_kw():
    # verify that the rendering_kw keyword argument works, if the correct
    # number of dictionaries is provided.

    x, t = symbols("x, t")

    # one expression -> 2 series -> 2 dictionaries
    p = plot_real_imag(sqrt(x), (x, -4-4j, 4+4j), backend=MB, n=5, show=False,
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}])
    assert len(p.series) == 2
    assert p[0].rendering_kw == {"cmap": "winter"}
    assert p[1].rendering_kw == {"cmap": "Greens"}

    p = plot_real_imag(sqrt(x)**t, (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].rendering_kw == {"cmap": "winter"}
    assert s[1].rendering_kw == {"cmap": "Greens"}

    # one expression -> 4 series -> 4 dictionaries
    p = plot_real_imag(sqrt(x), (x, -4-4j, 4+4j), backend=MB, n=5, show=False,
        abs=True, arg=True,
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}, {"cmap": "autumn"}, {"cmap": "viridis"}])
    assert len(p.series) == 4
    assert p[0].rendering_kw == {"cmap": "winter"}
    assert p[1].rendering_kw == {"cmap": "Greens"}
    assert p[2].rendering_kw == {"cmap": "autumn"}
    assert p[3].rendering_kw == {"cmap": "viridis"}

    p = plot_real_imag(sqrt(x)**t, (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, abs=True, arg=True, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}, {"cmap": "autumn"}, {"cmap": "viridis"}])
    s = p.backend.series
    assert len(s) == 4
    assert s[0].rendering_kw == {"cmap": "winter"}
    assert s[1].rendering_kw == {"cmap": "Greens"}
    assert s[2].rendering_kw == {"cmap": "autumn"}
    assert s[3].rendering_kw == {"cmap": "viridis"}

    # one expression -> 2 series -> 3 dictionaries -> raise error
    p = lambda: plot_real_imag(sqrt(x), (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False,
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}, {"cmap": "autumn"}])
    raises(ValueError, p)

    p = lambda: plot_real_imag(sqrt(x)**t, (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}, {"cmap": "autumn"}])
    raises(ValueError, p)

    # two expression -> 4 series -> 4 dictionaries
    p = plot_real_imag(sqrt(x), log(x), (x, -4-4j, 4+4j), backend=MB, n=5,
        show=False,
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}, {"cmap": "autumn"}, {"cmap": "viridis"}])
    assert len(p.series) == 4
    assert p[0].rendering_kw == {"cmap": "winter"}
    assert p[1].rendering_kw == {"cmap": "Greens"}
    assert p[2].rendering_kw == {"cmap": "autumn"}
    assert p[3].rendering_kw == {"cmap": "viridis"}

    p = plot_real_imag(sqrt(x)**t, log(x)**t, (x, -4-4j, 4+4j), backend=MB,
        n=5, show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "winter"}, {"cmap": "Greens"}, {"cmap": "autumn"}, {"cmap": "viridis"}])
    s = p.backend.series
    assert len(s) == 4
    assert s[0].rendering_kw == {"cmap": "winter"}
    assert s[1].rendering_kw == {"cmap": "Greens"}
    assert s[2].rendering_kw == {"cmap": "autumn"}
    assert s[3].rendering_kw == {"cmap": "viridis"}


def test_plot_complex_1d_rendering_kw():
    # verify that the rendering_kw keyword argument works, if the correct
    # number of dictionaries is provided.

    x, t = symbols("x, t")

    # one series -> one dictionary
    p = plot_complex(cos(x) + sin(I * x), (x, -2, 2),
        backend=MB, show=False, rendering_kw={"cmap": "autumn"})
    assert len(p.series) == 1
    assert p[0].rendering_kw == {"cmap": "autumn"}

    p = plot_complex(cos(x) + sin(I * x * t), (x, -2, 2),
        backend=MB, show=False, params={t: (1, 0, 2)},
        rendering_kw={"cmap": "autumn"})
    s = p.backend.series
    assert len(s) == 1
    assert s[0].rendering_kw == {"cmap": "autumn"}

    # one series -> 2 dictionaries -> raise error
    p = lambda: plot_complex(cos(x) + sin(I * x), (x, -2, 2),
        backend=MB, show=False,
        rendering_kw=[{"cmap": "autumn"}, {"cmap": "winter"}])
    raises(ValueError, p)

    p = lambda: plot_complex(cos(x) + sin(I * x * t), (x, -2, 2),
        backend=MB, show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "autumn"}, {"cmap": "winter"}])
    raises(ValueError, p)

    # two series -> 2 dictionaries
    p = plot_complex(cos(x) + sin(I * x), exp(I * x) * I * sin(x), (x, -2, 2),
        backend=MB, show=False,
        rendering_kw=[{"cmap": "autumn"}, {"cmap": "winter"}])
    assert len(p.series) == 2
    assert p[0].rendering_kw == {"cmap": "autumn"}
    assert p[1].rendering_kw == {"cmap": "winter"}

    p = plot_complex(
        cos(x) + sin(I * x * t), exp(I * x) * I * sin(t * x), (x, -2, 2),
        backend=MB, show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "autumn"}, {"cmap": "winter"}])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].rendering_kw == {"cmap": "autumn"}
    assert s[1].rendering_kw == {"cmap": "winter"}


def test_plot_complex_2d_3d_rendering_kw():
    # verify that the rendering_kw keyword argument works, if the correct
    # number of dictionaries is provided.

    x, t = symbols("x, t")

    # one series -> one dictionary
    p = plot_complex(sin(x), (x, -2-2j, 2+2j),
        backend=MB, show=False, rendering_kw={"interpolation": "bilinear"})
    assert len(p.series) == 1
    assert p[0].rendering_kw == {"interpolation": "bilinear"}

    p = plot_complex(sin(t * x), (x, -2-2j, 2+2j),
        backend=MB, show=False, params={t: (1, 0, 2)},
        rendering_kw={"interpolation": "bilinear"})
    s = p.backend.series
    assert len(s) == 1
    assert s[0].rendering_kw == {"interpolation": "bilinear"}

    # one series -> 2 dictionaries -> raise error
    p = lambda: plot_complex(sin(x), (x, -2-2j, 2+2j),
        backend=MB, show=False,
        rendering_kw=[{"interpolation": "bilinear"}, {"interpolation": "none"}])
    raises(ValueError, p)

    p = lambda: plot_complex(sin(t * x), (x, -2-2j, 2+2j),
        backend=MB, show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"interpolation": "bilinear"}, {"interpolation": "none"}])
    raises(ValueError, p)

    # two series -> 2 dictionaries
    p = plot_complex((sin(x), (x, -2-2j, 2j)), (cos(x), (x, -2j, 2+2j)),
        backend=MB, show=False,
        rendering_kw=[{"interpolation": "bilinear"}, {"interpolation": "none"}])
    assert len(p.series) == 2
    assert p[0].rendering_kw == {"interpolation": "bilinear"}
    assert p[1].rendering_kw == {"interpolation": "none"}

    p = plot_complex(
        (sin(t * x), (x, -2-2j, 2j)),
        (cos(t * x), (x, -2j, 2+2j)),
        backend=MB, show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"interpolation": "bilinear"}, {"interpolation": "none"}])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].rendering_kw == {"interpolation": "bilinear"}
    assert s[1].rendering_kw == {"interpolation": "none"}


def test_plot_complex_list_rendering_kw():
    # verify that the rendering_kw keyword argument works, if the correct
    # number of dictionaries is provided.

    x = symbols("x")

    # single complex number -> 1 dictionary
    p = plot_complex_list(3 + 2 * I, backend=MB, show=False,
        rendering_kw={"marker": "+"})
    assert len(p.series) == 1
    assert p[0].rendering_kw == {"marker": "+"}

    p = plot_complex_list(x * 3 + 2 * I, params={x: (1, 0, 2)},
        backend=MB, show=False, label="f")
    s = p.backend.series
    assert len(s) == 1
    assert s[0].get_label(False) == "f"

    # 2 complex numbers -> 2 dictionaries
    p = plot_complex_list(3 + 2 * I, 5 - 4 * I, backend=MB, show=False,
        rendering_kw=[{"marker": "+"}, {"marker": "o"}])
    assert len(p.series) == 2
    assert p[0].rendering_kw == {"marker": "+"}
    assert p[1].rendering_kw == {"marker": "o"}

    p = plot_complex_list(x * 3 + 2 * I, 5 * x - 4 * I, params={x: (1, 0, 2)},
        backend=MB, show=False, rendering_kw=[{"marker": "+"}, {"marker": "o"}])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].rendering_kw == {"marker": "+"}
    assert s[1].rendering_kw == {"marker": "o"}

    # 2 complex numbers -> 3 dictionary -> raise error
    p = lambda: plot_complex_list(3 + 2 * I, 5 - 4 * I, backend=MB, show=False,
        rendering_kw=[{"marker": "+"}, {"marker": "."}, {"marker": "o"}])
    raises(ValueError, p)

    p = lambda: plot_complex_list(
        x * 3 + 2 * I, 5 * x - 4 * I, params={x: (1, 0, 2)},
        backend=MB, show=False,
        rendering_kw=[{"marker": "+"}, {"marker": "."}, {"marker": "o"}])
    raises(ValueError, p)

    # 2 lists of grouped complex numbers -> 2 dictionaries
    p = plot_complex_list(
        [3 + 2 * I, 2 * I, 3],
        [2 + 3 * I, -2 * I, -3],
        backend=MB, show=False, rendering_kw=[{"marker": "+"}, {"marker": "o"}])
    assert len(p.series) == 2
    assert p[0].rendering_kw == {"marker": "+"}
    assert p[1].rendering_kw == {"marker": "o"}

    p = plot_complex_list(
        [3 * x + 2 * I, 2 * I, 3],
        [2 + 3 * I, -2 * I, -3],
        params={x: (1, 0, 2)}, backend=MB, show=False,
        rendering_kw=[{"marker": "+"}, {"marker": "o"}])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].rendering_kw == {"marker": "+"}
    assert s[1].rendering_kw == {"marker": "o"}


def plot_complex_vector_rendering_kw():
    # verify that the rendering_kw keyword argument works, if the correct
    # number of dictionaries is provided.

    x, t = symbols("x, t")

    # one expression -> 2 series -> 2 dictionaries
    p = plot_complex_vector(x**2 + sin(x), (x, -5-5j, 5+5j),
        backend=MB, show=False,
        rendering_kw=[{"cmap": "autumn"}, {"color": "w"}])
    assert len(p.series) == 2
    assert p[0].rendering_kw == {"cmap": "autumn"}
    assert p[1].rendering_kw == {"color": "w"}

    p = plot_complex_vector(x**2 + sin(t * x), (x, -5-5j, 5+5j),
        backend=MB, show=False, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "autumn"}, {"color": "w"}])
    s = p.backend.series
    assert len(s) == 2
    assert s[0].rendering_kw == {"cmap": "autumn"}
    assert s[1].rendering_kw == {"color": "w"}

    # one expression -> 1 series -> 2 dictionaries -> raise error
    p = lambda: plot_complex_vector(x**2 + sin(x), (x, -5-5j, 5+5j),
        backend=MB, show=False, scalar=False,
        rendering_kw=[{"cmap": "autumn"}, {"color": "w"}])
    raises(ValueError, p)

    p = lambda: plot_complex_vector(x**2 + sin(t * x), (x, -5-5j, 5+5j),
        backend=MB, show=False, scalar=False, params={t: (1, 0, 2)},
        rendering_kw=[{"cmap": "autumn"}, {"color": "w"}])
    raises(ValueError, p)
