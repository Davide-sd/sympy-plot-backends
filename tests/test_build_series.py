from sympy import symbols, I, sqrt, sin, asin, cos, Plane
from spb.ccomplex.complex import _build_series as _build_complex_series
from spb.vectors import _preprocess, _build_series as _build_vector_series
from spb.utils import _plot_sympify
from spb.series import (
    ComplexPointSeries, ComplexPointInteractiveSeries,
    ComplexSeries, ComplexInteractiveSeries,
    LineOver1DRangeSeries, SurfaceOver2DRangeSeries,
    InteractiveSeries, ContourSeries, Vector2DSeries,
    Vector3DSeries, SliceVector3DSeries
)

# The _build_series functions are going to create different objects
# depending on the provided arguments. The aim of the following tests
# is to assure that the expected objects are being created.

def bcs(*args, **kwargs):
    # use Numpy for the following tests to speed things up
    kwargs["modules"] = None
    args = _plot_sympify(args)
    return _build_complex_series(*args, **kwargs)

def bvs(*args, **kwargs):
    args = _plot_sympify(args)
    args = _preprocess(*args)
    return _build_vector_series(*args, **kwargs)

def test_build_complex_series():
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

    # two surface series when an expression with two ranges is given
    s = bcs(sqrt(x), (x, -5, 5), (y, -5, 5), real=True, interactive=False)
    assert len(s) == 2
    assert all(isinstance(t, SurfaceOver2DRangeSeries) for t in s)

    s = bcs(sqrt(x), (x, -5, 5), (y, -5, 5), real=True, interactive=True,
            threed=True)
    assert len(s) == 2
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    # multiple surface series when an expression with two ranges is given,
    # together with real, imag, abs, arg flags
    s = bcs(sqrt(x), (x, -5, 5), (y, -5, 5),
            real=True, imag=True, abs=True, arg=True, interactive=False)
    assert len(s) == 4
    assert all(isinstance(t, SurfaceOver2DRangeSeries) for t in s)

    s = bcs(sqrt(x), (x, -5, 5), (y, -5, 5),
            real=True, imag=True, abs=True, arg=True, interactive=True,
            threed=True)
    assert len(s) == 4
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    # list of complex numbers, each one with its own label
    s = bcs((3+2*I, "a"), (5 * I, "b"), interactive=False)
    assert len(s) == 2
    assert all(isinstance(t, ComplexPointSeries) for t in s)

    s = bcs((3+2*I, "a"), (5 * I, "b"), interactive=True)
    assert len(s) == 2
    assert all(isinstance(t, ComplexPointInteractiveSeries) for t in s)

    # multiple line series over a 1D real range
    s = bcs(sqrt(x), real=True, imag=True, abs=True, arg=True,
            absarg=True, interactive=False)
    assert len(s) == 5
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)

    s = bcs(sqrt(x), real=True, imag=True, abs=True, arg=True,
            absarg=True, interactive=True)
    assert len(s) == 5
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)

    # if only absarg=True, then only one line series should be added
    s = bcs(sqrt(x), absarg=True, interactive=False)
    assert len(s) == 1
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)

    s = bcs(sqrt(x), absarg=True, interactive=True)
    assert len(s) == 1
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)

    # multiple 3D plots of a complex function over a complex range
    s = bcs(sin(z), (z, -5-5j, 5+5j), real=True, imag=True, abs=True, arg=True,
            threed=True, interactive=False)
    assert len(s) == 4
    assert all(isinstance(t, ComplexSeries) for t in s)

    s = bcs(sin(z), (z, -5-5j, 5+5j), real=True, imag=True, abs=True, arg=True,
            threed=True, interactive=True)
    assert len(s) == 4
    assert all(isinstance(t, ComplexInteractiveSeries) for t in s)
    assert all(t.is_3Dsurface for t in s)

    # domain coloring
    s = bcs(sin(z), (z, -5-5j, 5+5j), interactive=False)
    assert len(s) == 1
    assert all(isinstance(t, ComplexSeries) for t in s)
    assert all(t.is_domain_coloring for t in s)

    s = bcs(sin(z), (z, -5-5j, 5+5j), interactive=True)
    assert len(s) == 1
    assert all(isinstance(t, ComplexInteractiveSeries) for t in s)
    assert all(t.is_domain_coloring for t in s)

    # multiple expressions each one with its label and range
    s = bcs((sqrt(x), (x, -5, 5), "f"), (asin(x), (x, -8, 8), "g"),
                interactive=False)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert (s[0].start == -5) and (s[0].end == 5)
    assert (s[1].start == -5) and (s[1].end == 5)
    assert (s[2].start == -8) and (s[2].end == 8)
    assert (s[3].start == -8) and (s[3].end == 8)

    s = bcs((sqrt(x), (x, -5, 5), "f"), (asin(x), (x, -8, 8), "g"),
                interactive=True, n1=10)
    assert len(s) == 4
    assert all(isinstance(t, InteractiveSeries) for t in s)
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
    s = bcs((sqrt(x), "f"), (asin(x), "g"), (x, -5, 5),
                interactive=False)
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    assert all((t.start == -5) and (t.end == 5) for t in s)

    s = bcs((sqrt(x), "f"), (asin(x), "g"), (x, -5, 5),
                interactive=True, n1=10)
    assert len(s) == 4
    assert all(isinstance(t, InteractiveSeries) for t in s)
    assert all(t.is_2Dline for t in s)
    r = list(s[0].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[1].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[2].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)
    r = list(s[3].ranges.values())[0]
    assert (r[0] == -5) and (r[-1] == 5)


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
