from sympy import exp
from sympy.core.symbol import symbols
from sympy.core.numbers import I, pi
from sympy.functions.elementary.trigonometric import sin, cos, asin
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import re, im, arg
from sympy.geometry.plane import Plane
from spb.ccomplex.complex import _build_series as _build_complex_series
from spb.vectors import _preprocess, _build_series as _build_vector_series
from spb.utils import _plot_sympify
from spb.series import (
    ComplexPointSeries, ComplexPointInteractiveSeries,
    ComplexSurfaceSeries, ComplexSurfaceInteractiveSeries,
    ComplexDomainColoringSeries, ComplexDomainColoringInteractiveSeries,
    LineOver1DRangeSeries,
    InteractiveSeries, ContourSeries, Vector2DSeries,
    Vector3DSeries, SliceVector3DSeries,
    AbsArgLineSeries,
    LineInteractiveSeries, AbsArgLineInteractiveSeries
)

# NOTE:
#
# The _build_series functions are going to create different Complex or
# Vector series depending on the provided arguments. The aim of the
# following tests is to assure that the expected objects are being created.
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to test_series.py.
# If your issue is related to the processing and generation of *Series
# objects not involving Complex or Vector series, consider adding tests
# to test_functions.py.
# If your issue is related to a particular keyword affecting a backend
# behaviour, consider adding tests to test_backends.py
#


def bcs(*args, **kwargs):
    # use Numpy/Scipy for the following tests to speed things up
    kwargs["modules"] = None
    args = _plot_sympify(args)
    return _build_complex_series(*args, **kwargs)


def bvs(*args, **kwargs):
    args = _plot_sympify(args)
    args = _preprocess(*args)
    return _build_vector_series(*args, **kwargs)


def test_build_complex_point_series():
    x, y, z = symbols("x:z")

    # single complex number
    s = bcs(3 + 2 * I, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], ComplexPointSeries)

    s = bcs(3 + 2 * I, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], ComplexPointInteractiveSeries)

    # list of grouped complex numbers with labels
    s = bcs(
        ([3 + 2 * I, 2 * I, 3], "a"),
        ([2 + 3 * I, -2 * I, -3], "b"),
        interactive=False)
    assert len(s) == 2
    assert all(isinstance(t, ComplexPointSeries) for t in s)

    s = bcs(
        ([3 + 2 * I, 2 * I, 3], "a"),
        ([2 + 3 * I, -2 * I, -3], "b"),
        interactive=True)
    assert len(s) == 2
    assert all(isinstance(t, ComplexPointInteractiveSeries) for t in s)

    # list of complex numbers, each one with its own label
    s = bcs((3+2*I, "a"), (5 * I, "b"), interactive=False)
    assert len(s) == 2
    assert all(isinstance(t, ComplexPointSeries) for t in s)

    s = bcs((3+2*I, "a"), (5 * I, "b"), interactive=True)
    assert len(s) == 2
    assert all(isinstance(t, ComplexPointInteractiveSeries) for t in s)


def test_build_complex_line_series():
    x, y, z = symbols("x:z")

    # default kwargs correspond to absarg=True. Only one series should be
    # created.
    s = bcs(sqrt(x), interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], AbsArgLineSeries)

    s = bcs(sqrt(x), interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], AbsArgLineInteractiveSeries)

    # same as the previous case
    s = bcs(sqrt(x), absarg=True, interactive=False)
    assert len(s) == 1
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)

    s = bcs(sqrt(x), absarg=True, interactive=True)
    assert len(s) == 1
    assert all(isinstance(t, InteractiveSeries) for t in s)

    # real part of the function
    s = bcs(sqrt(x), absarg=False, real=True, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == re(sqrt(x))

    s = bcs(sqrt(x), absarg=False, real=True, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == re(sqrt(x))

    # imaginary part of the function
    s = bcs(sqrt(x), absarg=False, real=False, imag=True, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == im(sqrt(x))

    s = bcs(sqrt(x), absarg=False, real=False, imag=True, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == im(sqrt(x))

    # absolute value of the function
    s = bcs(sqrt(x), absarg=False, real=False, imag=False, abs=True,
        interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == sqrt(re(sqrt(x))**2 + im(sqrt(x))**2)

    s = bcs(sqrt(x), absarg=False, real=False, imag=False, abs=True,
        interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == sqrt(re(sqrt(x))**2 + im(sqrt(x))**2)

    # argument of the function
    s = bcs(sqrt(x), absarg=False, real=False, imag=False, abs=False,
        arg=True, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], LineOver1DRangeSeries)
    assert s[0].expr == arg(sqrt(x))

    s = bcs(sqrt(x), absarg=False, real=False, imag=False, abs=False,
        arg=True, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], LineInteractiveSeries)
    assert s[0].expr == arg(sqrt(x))

    # multiple line series over a 1D real range
    s = bcs(sqrt(x), real=True, imag=True, abs=True, arg=True,
            absarg=True, interactive=False)
    assert len(s) == 5
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert isinstance(s[0], AbsArgLineSeries)

    s = bcs(sqrt(x), real=True, imag=True, abs=True, arg=True,
            absarg=True, interactive=True)
    assert len(s) == 5
    assert all(isinstance(t, LineInteractiveSeries) for t in s)
    assert isinstance(s[0], AbsArgLineInteractiveSeries)
    assert all(t.is_2Dline for t in s)

    # multiple expressions with a common range
    s = bcs(sqrt(x), asin(x), (x, -8, 8), real=True, imag=True, abs=False,
            arg=False, absarg=False, interactive=False)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == -8) and (ss.end == 8) for ss in s)

    # multiple expressions with a common unspecified range
    s = bcs(sqrt(x), asin(x), real=True, imag=True, abs=False,
            arg=False, absarg=False, interactive=False)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((ss.start == -10) and (ss.end == 10) for ss in s)

    # multiple expressions each one with its label and range
    s = bcs((sqrt(x), (x, -5, 5), "f"), (asin(x), (x, -8, 8), "g"),
                interactive=False)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert (s[0].start == -5) and (s[0].end == 5)
    assert (s[1].start == -8) and (s[1].end == 8)

    s = bcs((sqrt(x), (x, -5, 5), "f"), (asin(x), (x, -8, 8), "g"),
                interactive=True, n1=10)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    r = list(s[0].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[1].ranges.values())[0]
    assert (r[0] == -8) and (r[-1] == 8)

    # multiple expressions each one with its label and a common range
    s = bcs((sqrt(x), "f"), (asin(x), "g"), (x, -5, 5),
                interactive=False)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((t.start == -5) and (t.end == 5) for t in s)

    s = bcs((sqrt(x), "f"), (asin(x), "g"), (x, -5, 5),
                interactive=True, n1=10)
    assert len(s) == 2
    assert all(isinstance(t, AbsArgLineInteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    r = list(s[0].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[1].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)

    # multiple expressions each one with its label and range + multiple kwargs
    s = bcs((sqrt(x), (x, -5, 5), "f"), (asin(x), (x, -8, 8), "g"),
                real=True, imag=True, interactive=False)
    assert len(s) == 6
    assert all(isinstance(s[i], AbsArgLineSeries) for i in [0, 3])
    assert all(isinstance(s[i], LineOver1DRangeSeries) for i in [1, 2, 4, 5])
    assert all(t.is_2Dline for t in s)
    assert all((s[i].start == -5) and (s[i].end == 5) for i in [0, 1, 2])
    assert all((s[i].start == -8) and (s[i].end == 8) for i in [3, 4, 5])

    s = bcs((sqrt(x), (x, -5, 5), "f"), (asin(x), (x, -8, 8), "g"),
                real=True, imag=True, interactive=True, n1=10)
    assert len(s) == 6
    assert all(isinstance(s[i], AbsArgLineInteractiveSeries) for i in [0, 3])
    assert all(isinstance(s[i], LineInteractiveSeries) for i in [1, 2, 4, 5])
    assert all(t.is_2Dline for t in s)
    assert all((s[i].start == -5) and (s[i].end == 5) for i in [0, 1, 2])
    assert all((s[i].start == -8) and (s[i].end == 8) for i in [3, 4, 5])


def test_build_complex_surface_series():
    x, y, z = symbols("x:z")

    # default kwargs correspond to absarg=True. Only one series should be
    # created.
    s = bcs(sin(z), (z, -5-5j, 5+5j), interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], ComplexDomainColoringSeries)
    assert s[0].is_domain_coloring

    s = bcs(sin(z), (z, -5-5j, 5+5j), interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], ComplexDomainColoringInteractiveSeries)
    assert s[0].is_domain_coloring

    # real part of the function
    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=True,
        interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == re(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=True,
        interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == re(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # imaginary part of the function
    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=False, imag=True,
        interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == im(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=False, imag=True,
        interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == im(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # absolute value of the function
    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=False, imag=False,
        abs=True, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == sqrt(re(sin(z))**2 + im(sin(z))**2)
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=False, imag=False,
        abs=True, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == sqrt(re(sin(z))**2 + im(sin(z))**2)
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # argument of the function
    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=False, imag=False,
        abs=False, arg=True, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == arg(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=False, real=False, imag=False,
        abs=False, arg=True, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], ComplexSurfaceInteractiveSeries)
    assert s[0].is_contour and (not s[0].is_3Dsurface)
    assert s[0].expr == arg(sin(z))
    assert (s[0].start == -5 - 5j) and (s[0].end == 5 + 5j)

    # multiple 2D plots (contours) of a complex function over a complex range
    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=True, real=True, imag=True,
        abs=True, arg=True, threed=False, interactive=False)
    assert len(s) == 5
    assert isinstance(s[0], ComplexDomainColoringSeries)
    assert all(isinstance(s[i], ComplexSurfaceSeries) for i in range(1, len(s)))
    # when threed=False, there are no 3D surface series, but contours.
    assert all(not t.is_3Dsurface for t in s)
    assert all(s[i].is_contour for i in range(1, len(s)))

    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=True, real=True, imag=True,
        abs=True, arg=True, threed=False, interactive=True)
    assert len(s) == 5
    assert isinstance(s[0], ComplexDomainColoringInteractiveSeries)
    assert all(isinstance(s[i], ComplexSurfaceInteractiveSeries) for i in range(1, len(s)))
    # when threed=False, there are no 3D surface series, but contours.
    assert all(not t.is_3Dsurface for t in s)
    assert all(s[i].is_contour for i in range(1, len(s)))

    # multiple 3D plots (surfaces) of a complex function over a complex range
    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=True, real=True, imag=True,
        abs=True, arg=True, threed=True, interactive=False)
    assert len(s) == 5
    assert isinstance(s[0], ComplexDomainColoringSeries)
    assert all(isinstance(s[i], ComplexSurfaceSeries) for i in range(1, len(s)))
    assert all(t.is_3Dsurface for t in s)

    s = bcs(sin(z), (z, -5-5j, 5+5j), absarg=True, real=True, imag=True,
        abs=True, arg=True, threed=True, interactive=True)
    assert len(s) == 5
    assert isinstance(s[0], ComplexDomainColoringInteractiveSeries)
    assert all(isinstance(s[i], ComplexSurfaceInteractiveSeries) for i in range(1, len(s)))
    assert all(t.is_3Dsurface for t in s)


def test_issue_6():
    phi = symbols('phi', real=True)
    vec = cos(phi) + cos(phi - 2 * pi / 3) * exp(I * 2 * pi / 3) + cos(phi - 4 * pi / 3) * exp(I * 4 * pi / 3)
    s = bcs(vec, (phi, 0, 2 * pi), absarg=False, real=True, imag=True)
    assert len(s) == 2
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert s[0].expr == re(vec)
    assert s[1].expr == im(vec)


def test_build_vector_series():
    x, y, z = symbols("x:z")

    # 2D vector field: default to a contour series and quivers
    s = bvs([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3), interactive=False)
    assert len(s) == 2
    assert isinstance(s[0], ContourSeries)
    assert isinstance(s[1], Vector2DSeries)

    s = bvs([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3), interactive=True)
    assert len(s) == 2
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert s[0].is_contour
    assert s[1].is_2Dvector

    # 2D vector field with contour series, only quivers
    s = bvs([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
            scalar=None, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], Vector2DSeries)

    s = bvs([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
            scalar=None, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], InteractiveSeries)
    assert s[0].is_2Dvector

    # multiple 2D vector field: only quivers
    s = bvs([-sin(y), cos(x)], [-y, x], (x, -3, 3), (y, -3, 3),
            interactive=False)
    assert len(s) == 2
    assert all(isinstance(t, Vector2DSeries) for t in s)

    s = bvs([-sin(y), cos(x)], [-y, x], (x, -3, 3), (y, -3, 3),
            interactive=True)
    assert len(s) == 2
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert all(t.is_2Dvector for t in s)

    # multiple 2D vector field with a scalar field: contour + quivers
    s = bvs([-sin(y), cos(x)], [-y, x], (x, -3, 3), (y, -3, 3),
            scalar=sqrt(x**2 + y**2), interactive=False)
    assert len(s) == 3
    assert isinstance(s[0], ContourSeries)
    assert isinstance(s[1], Vector2DSeries)
    assert isinstance(s[2], Vector2DSeries)

    s = bvs([-sin(y), cos(x)], [-y, x], (x, -3, 3), (y, -3, 3),
            scalar=sqrt(x**2 + y**2), interactive=True)
    assert len(s) == 3
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert s[0].is_contour
    assert s[1].is_2Dvector
    assert s[2].is_2Dvector

    # 3D vector field
    s = bvs([x, y, z], (x, -10, 10), (y, -10, 10), (z, -10, 10),
            n=8, interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], Vector3DSeries)

    s = bvs([x, y, z], (x, -10, 10), (y, -10, 10), (z, -10, 10),
            n=8, interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], InteractiveSeries)
    assert s[0].is_3Dvector

    # 3D vector field with slices
    s = bvs([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10), n=8,
            slice=Plane((-10, 0, 0), (1, 0, 0)), interactive=False)
    assert len(s) == 1
    assert isinstance(s[0], SliceVector3DSeries)

    s = bvs([x, y, z], (x, -10, 10), (y, -10, 10), (z, -10, 10), n=8,
            slice=Plane((-10, 0, 0), (1, 0, 0)), interactive=True)
    assert len(s) == 1
    assert isinstance(s[0], InteractiveSeries)
    assert s[0].is_3Dvector and s[0].is_slice
