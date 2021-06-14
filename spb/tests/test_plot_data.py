from sympy import symbols, cos, sin, log, Eq, I, Abs, exp, pi, gamma
from sympy.vector import CoordSys3D
from pytest import raises
from spb.plot_data import _build_series
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries, InteractiveSeries,
    ImplicitSeries, Vector2DSeries, Vector3DSeries, ComplexSeries
)
import numpy as np

def test_build_series():
    x, y, u, v = symbols("x, y, u, v")

    # test automatic algoritm

    s = _build_series(cos(x), (x, -5, 5))
    assert isinstance(s, LineOver1DRangeSeries)

    s = _build_series((cos(x), sin(x)), (x, -5, 5))
    assert isinstance(s, Parametric2DLineSeries)

    s = _build_series(cos(x), sin(x), (x, -5, 5))
    assert isinstance(s, Parametric2DLineSeries)

    s = _build_series(cos(x), sin(x), x, (x, -5, 5))
    assert isinstance(s, Parametric3DLineSeries)

    s = _build_series(cos(x + y), (x, -5, 5))
    assert isinstance(s, SurfaceOver2DRangeSeries)

    s = _build_series(cos(x + y), sin(x + y), x, (x, -5, 5))
    assert isinstance(s, ParametricSurfaceSeries)

    s = _build_series((cos(x + y), sin(x + y), x), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ParametricSurfaceSeries)

    s = _build_series(u * cos(x), (x, -5, 5), params={u: 1})
    assert isinstance(s, InteractiveSeries)

    s = _build_series(Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ImplicitSeries)

    s = _build_series(Eq(x**2 + y**2, 5) & (x > y), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ImplicitSeries)

    # test mapping
    s = _build_series(cos(x), (x, -5, 5), pt="p")
    assert isinstance(s, LineOver1DRangeSeries)

    s = _build_series(cos(x), sin(x), (x, -5, 5), pt="pp")
    assert isinstance(s, Parametric2DLineSeries)

    s = _build_series(cos(x), sin(x), x, (x, -5, 5), pt="p3dl")
    assert isinstance(s, Parametric3DLineSeries)

    s = _build_series(cos(x + y), (x, -5, 5), (y, -3, 3), pt="p3d")
    assert isinstance(s, SurfaceOver2DRangeSeries)

    # one missing rage
    s = _build_series(cos(x + y), (x, -5, 5), pt="p3d")
    assert isinstance(s, SurfaceOver2DRangeSeries)

    s = _build_series(cos(x + y), sin(x + y), x, (x, -5, 5), (y, -3, 3),
            pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)

    # missing ranges
    s = _build_series(cos(x + y), sin(x + y), x, pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)

    s = _build_series(u * cos(x), (x, -5, 5), params={u: 1}, pt="pinter")
    assert isinstance(s, InteractiveSeries)

def test_vectors():
    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j
    v2 = z * N.i + x * N.j + y * N.k
    m1 = v1.to_matrix(N)
    m2 = v2.to_matrix(N)
    l1 = list(m1)
    # I need a 2D vector: delete the last component, which is zero
    l1 = l1[:-1]
    l2 = list(m2)

    # 2D vectors
    s = _build_series(v1, (x, -10, 10), (y, -5, 5))
    assert isinstance(s, Vector2DSeries)
    s = _build_series(m1, (x, -10, 10), (y, -5, 5))
    assert isinstance(s, Vector2DSeries)
    s = _build_series(l1, (x, -10, 10), (y, -5, 5))
    assert isinstance(s, Vector2DSeries)
    s = _build_series(v1, (x, -10, 10), (y, -5, 5), pt="v2d")
    assert isinstance(s, Vector2DSeries)
    s = _build_series(m1, (x, -10, 10), (y, -5, 5), pt="v2d")
    assert isinstance(s, Vector2DSeries)
    s = _build_series(l1, (x, -10, 10), (y, -5, 5), pt="v2d")
    assert isinstance(s, Vector2DSeries)

    s = _build_series(v2, (x, -10, 10), (y, -5, 5), (z, -8, 8))
    assert isinstance(s, Vector3DSeries)
    s = _build_series(m2, (x, -10, 10), (y, -5, 5), (z, -8, 8))
    assert isinstance(s, Vector3DSeries)
    s = _build_series(l2, (x, -10, 10), (y, -5, 5), (z, -8, 8))
    assert isinstance(s, Vector3DSeries)
    s = _build_series(v2, (x, -10, 10), (y, -5, 5), (z, -8, 8), pt="v3d")
    assert isinstance(s, Vector3DSeries)
    s = _build_series(m2, (x, -10, 10), (y, -5, 5), (z, -8, 8), pt="v3d")
    assert isinstance(s, Vector3DSeries)
    s = _build_series(l2, (x, -10, 10), (y, -5, 5), (z, -8, 8), pt="v3d")
    assert isinstance(s, Vector3DSeries)

def test_complex():
    x, y, z = symbols("x:z")
    e1 = 1 + exp(-Abs(x)) * sin(I * sin(5 * x))

    def do_test_1(s, n):
        assert isinstance(s, ComplexSeries)
        data = s.get_data()
        assert len(data) == n
        return data
    
    ### Complex line plots

    # return x, mag(e1), arg(e1)
    s1 = _build_series(e1, (x, -5, 5), absarg=True)
    data1 = do_test_1(s1, 3)
    s2 = _build_series(e1, (x, -5, 5), absarg=True, pt="c")
    data2 = do_test_1(s2, 3)
    assert all(t1 == t2 for t1, t2 in zip(data1[0], data2[0]))
    assert all(t1 == t2 for t1, t2 in zip(data1[1], data2[1]))
    assert all(t1 == t2 for t1, t2 in zip(data1[2], data2[2]))

    # return x, real(e1), imag(e1)
    s1 = _build_series(e1, (x, -5, 5))
    data1 = do_test_1(s1, 3)
    s2 = _build_series(e1, (x, -5, 5), pt="c")
    data2 = do_test_1(s2, 3)
    assert all(t1 == t2 for t1, t2 in zip(data1[0], data2[0]))
    assert all(t1 == t2 for t1, t2 in zip(data1[1], data2[1]))
    assert all(t1 == t2 for t1, t2 in zip(data1[2], data2[2]))
    xx, real, imag = data1

    # return x, real(e1)
    s1 = _build_series(e1, (x, -5, 5), imag=False)
    data1 = do_test_1(s1, 2)
    assert np.all(data1[0] == xx)
    assert np.all(data1[1] == real)
    s2 = _build_series(e1, (x, -5, 5), imag=False, pt="c")
    data2 = do_test_1(s2, 2)
    assert all(t1 == t2 for t1, t2 in zip(data1[0], data2[0]))
    assert all(t1 == t2 for t1, t2 in zip(data1[1], data2[1]))

    # return x, imag(e1)
    s1 = _build_series(e1, (x, -5, 5), real=False)
    data1 = do_test_1(s1, 2)
    assert np.all(data1[0] == xx)
    assert np.all(data1[1] == imag)
    s2 = _build_series(e1, (x, -5, 5), real=False, pt="c")
    data2 = do_test_1(s2, 2)
    assert all(t1 == t2 for t1, t2 in zip(data1[0], data2[0]))
    assert all(t1 == t2 for t1, t2 in zip(data1[1], data2[1]))

    # return x, e1 (complex numbers)
    s1 = _build_series(e1, (x, -5, 5), real=False, imag=False)
    data1 = do_test_1(s1, 2)
    assert np.all(data1[0] == xx)
    assert any(isinstance(d, complex) for d in data1[1].flatten())
    s2 = _build_series(e1, (x, -5, 5), real=False, imag=False, pt="c")
    data2 = do_test_1(s2, 2)
    assert all(t1 == t2 for t1, t2 in zip(data1[0], data2[0]))
    assert all(t1 == t2 for t1, t2 in zip(data1[1], data2[1]))


    ### Lists of complex numbers: returns real, imag
    e2 = z * exp(2 * pi * I * z)
    l2 = [e2.subs(z, t / 20) for t in range(20)]
    s = _build_series(l2)
    do_test_1(s, 2)


    ### Domain coloring: returns x, y, z where z is an array of complex numbers
    s1 = _build_series(gamma(z), (z, -3 - 3*I, 3 + 3*I))
    data1 = do_test_1(s1, 3)
    assert any(isinstance(d, complex) for d in data1[-1].flatten())
    s2 = _build_series(gamma(z), (z, -3 - 3*I, 3 + 3*I), pt="c")
    data2 = do_test_1(s2, 3)
    assert all(t1 == t2 for t1, t2 in zip(data1[0].flatten(), data2[0].flatten()))
    assert all(t1 == t2 for t1, t2 in zip(data1[1].flatten(), data2[1].flatten()))
    assert all(t1 == t2 for t1, t2 in zip(data1[2].flatten(), data2[2].flatten()))
