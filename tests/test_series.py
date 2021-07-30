from sympy import (
    symbols, cos, sin, log, sqrt,
    Tuple, pi, Plane, S, I, im
)
from spb.series import (
    LineOver1DRangeSeries,
    Parametric2DLineSeries,
    Parametric3DLineSeries,
    ParametricSurfaceSeries,
    SurfaceOver2DRangeSeries,
    InteractiveSeries,
    ImplicitSeries,
    Vector2DSeries,
    Vector3DSeries,
    ComplexSeries,
    ComplexInteractiveSeries,
    SliceVector3DSeries,
)
import numpy as np
from pytest import warns

def test_lin_log_scale():
    # Verify that data series create the correct spacing in the data.
    x, y, z = symbols("x, y, z")

    s = LineOver1DRangeSeries(x, (x, 1, 10), adaptive=False, n=50, xscale="linear")
    xx, _ = s.get_data()
    assert np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = LineOver1DRangeSeries(x, (x, 1, 10), adaptive=False, n=50, xscale="log")
    xx, _ = s.get_data()
    assert not np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = Parametric2DLineSeries(
        cos(x), sin(x), (x, pi / 2, 1.5 * pi), adaptive=False, n=50, xscale="linear"
    )
    _, _, param = s.get_data()
    assert np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = Parametric2DLineSeries(
        cos(x), sin(x), (x, pi / 2, 1.5 * pi), adaptive=False, n=50, xscale="log"
    )
    _, _, param = s.get_data()
    assert not np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = Parametric3DLineSeries(
        cos(x), sin(x), x, (x, pi / 2, 1.5 * pi), adaptive=False, n=50, xscale="linear"
    )
    _, _, _, param = s.get_data()
    assert np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = Parametric3DLineSeries(
        cos(x), sin(x), x, (x, pi / 2, 1.5 * pi), adaptive=False, n=50, xscale="log"
    )
    _, _, _, param = s.get_data()
    assert not np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2),
        (x, 1, 5),
        (y, 1, 5),
        n=10,
        xscale="linear",
        yscale="linear",
    )
    xx, yy, _ = s.get_data()
    assert np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    s = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, 1, 5), (y, 1, 5), n=10, xscale="log", yscale="log"
    )
    xx, yy, _ = s.get_data()
    assert not np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert not np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    s = ImplicitSeries(
        cos(x ** 2 + y ** 2) > 0,
        (x, 1, 5),
        (y, 1, 5),
        n=10,
        xscale="linear",
        yscale="linear",
        adaptive=False,
    )
    xx, yy, _, _, _ = s.get_data()
    assert np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])
    assert np.isclose(yy[1] - yy[0], yy[-1] - yy[-2])

    s = ImplicitSeries(
        cos(x ** 2 + y ** 2) > 0,
        (x, 1, 5),
        (y, 1, 5),
        n=10,
        xscale="log",
        yscale="log",
        adaptive=False,
    )
    xx, yy, _, _, _ = s.get_data()
    assert not np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])
    assert not np.isclose(yy[1] - yy[0], yy[-1] - yy[-2])

    s = InteractiveSeries([log(x)], [(x, 1e-05, 1e05)], n=10, xscale="linear")
    xx, yy = s.get_data()
    assert np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = InteractiveSeries([log(x)], [(x, 1e-05, 1e05)], n=10, xscale="log")
    xx, yy = s.get_data()
    assert not np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = ComplexSeries(
        cos(x),
        (x, 1e-05, 1e05),
        n=10,
        xscale="linear",
        adaptive=False,
    )
    xx, yy, _ = s.get_data()
    assert np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])

    s = ComplexSeries(
        cos(x),
        (x, 1e-05, 1e05),
        n=10,
        xscale="log",
        adaptive=False,
    )
    xx, yy, _ = s.get_data()
    assert not np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])

    s = Vector3DSeries(
        x,
        y,
        z,
        (x, 1, 1e05),
        (y, 1, 1e05),
        (z, 1, 1e05),
        xscale="linear",
        yscale="linear",
        zscale="linear",
    )
    xx, yy, zz, _, _, _ = s.get_data()
    assert np.isclose(
        xx[0, :, 0][1] - xx[0, :, 0][0], xx[0, :, 0][-1] - xx[0, :, 0][-2]
    )
    assert np.isclose(
        yy[:, 0, 0][1] - yy[:, 0, 0][0], yy[:, 0, 0][-1] - yy[:, 0, 0][-2]
    )
    assert np.isclose(
        zz[0, 0, :][1] - zz[0, 0, :][0], zz[0, 0, :][-1] - zz[0, 0, :][-2]
    )

    s = Vector3DSeries(
        x,
        y,
        z,
        (x, 1, 1e05),
        (y, 1, 1e05),
        (z, 1, 1e05),
        xscale="log",
        yscale="log",
        zscale="log",
    )
    xx, yy, zz, _, _, _ = s.get_data()
    assert not np.isclose(
        xx[0, :, 0][1] - xx[0, :, 0][0], xx[0, :, 0][-1] - xx[0, :, 0][-2]
    )
    assert not np.isclose(
        yy[:, 0, 0][1] - yy[:, 0, 0][0], yy[:, 0, 0][-1] - yy[:, 0, 0][-2]
    )
    assert not np.isclose(
        zz[0, 0, :][1] - zz[0, 0, :][0], zz[0, 0, :][-1] - zz[0, 0, :][-2]
    )


def test_data_shape():
    # Verify that the series produces the correct data shape when the input
    # expression is a number.
    u, x, y, z = symbols("u, x:z")

    # scalar expression: it should return a numpy ones array
    s = LineOver1DRangeSeries(1, (x, -5, 5))
    xx, yy = s.get_data()
    assert len(xx) == len(yy)
    assert np.all(yy == 1)

    s = LineOver1DRangeSeries(1, (x, -5, 5), adaptive=False)
    xx, yy = s.get_data()
    assert len(xx) == len(yy)
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(sin(x), 1, (x, 0, pi))
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(1, sin(x), (x, 0, pi))
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = Parametric2DLineSeries(sin(x), 1, (x, 0, pi), adaptive=False)
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(1, sin(x), (x, 0, pi), adaptive=False)
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = Parametric3DLineSeries(cos(x), sin(x), 1, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(zz == 1)

    s = Parametric3DLineSeries(cos(x), 1, x, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric3DLineSeries(1, sin(x), x, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = SurfaceOver2DRangeSeries(1, (x, -2, 2), (y, -3, 3))
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1))
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(xx == 1)

    s = ParametricSurfaceSeries(1, 1, y, (x, 0, 1), (y, 0, 1))
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(yy == 1)

    s = ParametricSurfaceSeries(x, 1, 1, (x, 0, 1), (y, 0, 1))
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ComplexSeries(1, (x, -5, 5), modules=None)
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ComplexSeries(1, (x, -5, 5), modules="mpmath")
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ComplexSeries(1, (x, -5 - 2 * I, 5 + 2 * I), domain_coloring=True,
            modules=None)
    rr, ii, mag, arg, colors, _ = s.get_data()
    assert (rr.shape == ii.shape) and (rr.shape[:2] == colors.shape[:2])
    assert (rr.shape == mag.shape) and (rr.shape == arg.shape)

    s = ComplexSeries(1, (x, -5 - 2 * I, 5 + 2 * I), domain_coloring=True,
            modules="mpmath")
    rr, ii, mag, arg, colors, _ = s.get_data()
    assert (rr.shape == ii.shape) and (rr.shape[:2] == colors.shape[:2])
    assert (rr.shape == mag.shape) and (rr.shape == arg.shape)

    # Corresponds to LineOver1DRangeSeries
    s = InteractiveSeries([S.One], [Tuple(x, -5, 5)])
    s.update_data(dict())
    xx, yy = s.get_data()
    assert len(xx) == len(yy)
    assert np.all(yy == 1)

    # Corresponds to Parametric2DLineSeries
    s = InteractiveSeries([S.One, sin(x)], [Tuple(x, 0, pi)])
    s.update_data(dict())
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = InteractiveSeries([sin(x), S.One], [Tuple(x, 0, pi)])
    s.update_data(dict())
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    # Corresponds to Parametric3DLineSeries
    s = InteractiveSeries([cos(x), sin(x), S.One], [(x, 0, 2 * pi)])
    s.update_data(dict())
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param)) and (len(xx) == len(zz))
    assert np.all(zz == 1)

    s = InteractiveSeries([S.One, sin(x), x], [(x, 0, 2 * pi)])
    s.update_data(dict())
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param)) and (len(xx) == len(zz))
    assert np.all(xx == 1)

    s = InteractiveSeries([cos(x), S.One, x], [(x, 0, 2 * pi)])
    s.update_data(dict())
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param)) and (len(xx) == len(zz))
    assert np.all(yy == 1)

    # Corresponds to SurfaceOver2DRangeSeries
    s = InteractiveSeries([S.One], [(x, -2, 2), (y, -3, 3)])
    s.update_data(dict())
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    # Corresponds to ParametricSurfaceSeries
    s = InteractiveSeries([S.One, x, y], [(x, 0, 1), (y, 0, 1)])
    s.update_data(dict())
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(xx == 1)

    s = InteractiveSeries([x, S.One, y], [(x, 0, 1), (y, 0, 1)])
    s.update_data(dict())
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(yy == 1)

    s = InteractiveSeries([x, y, S.One], [(x, 0, 1), (y, 0, 1)])
    s.update_data(dict())
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ComplexInteractiveSeries(S.One, (x, -5, 5), real=True, imag=False,
            modules=None)
    s.update_data(dict())
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)


def test_interactive():
    u, x, y, z = symbols("u, x:z")

    # verify that InteractiveSeries produces the same numerical data as their
    # corresponding non-interactive series.
    def do_test(data1, data2):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert np.allclose(d1, d2)

    s1 = InteractiveSeries([u * cos(x)], [(x, -5, 5)], "", params={u: 1}, n1=50)
    s2 = LineOver1DRangeSeries(cos(x), (x, -5, 5), "", adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())

    s1 = InteractiveSeries(
        [u * cos(x), u * sin(x)], [(x, -5, 5)], "", params={u: 1}, n1=50
    )
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5), "", adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())

    s1 = InteractiveSeries(
        [u * cos(x), u * sin(x), u * x], [(x, -5, 5)], "", params={u: 1}, n1=50
    )
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5), "", adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())

    s1 = InteractiveSeries(
        [cos(x ** 2 + y ** 2)],
        [(x, -3, 3), (y, -3, 3)],
        "",
        params={u: 1},
        n1=50,
        n2=50,
    )
    s2 = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, -3, 3), (y, -3, 3), "", adaptive=False, n1=50, n2=50
    )
    do_test(s1.get_data(), s2.get_data())

    s1 = InteractiveSeries(
        [cos(x + y), sin(x + y), x - y],
        [(x, -3, 3), (y, -3, 3)],
        "",
        params={u: 1},
        n1=50,
        n2=50,
    )
    s2 = ParametricSurfaceSeries(
        cos(x + y),
        sin(x + y),
        x - y,
        (x, -3, 3),
        (y, -3, 3),
        "",
        adaptive=False,
        n1=50,
        n2=50,
    )
    do_test(s1.get_data(), s2.get_data())

    s1 = InteractiveSeries(
        [-u * y, u * x], [(x, -3, 3), (y, -2, 2)], "", params={u: 1}, n1=15, n2=15
    )
    s2 = Vector2DSeries(-y, x, (x, -3, 3), (y, -2, 2), "", n1=15, n2=15)
    do_test(s1.get_data(), s2.get_data())

    s1 = InteractiveSeries(
        [u * z, -u * y, u * x],
        [(x, -3, 3), (y, -2, 2), (z, -1, 1)],
        "",
        params={u: 1},
        n1=15,
        n2=15,
        n3=15,
    )
    s2 = Vector3DSeries(
        z, -y, x, (x, -3, 3), (y, -2, 2), (z, -1, 1), "", n1=15, n2=15, n3=15
    )
    do_test(s1.get_data(), s2.get_data())

    s1 = InteractiveSeries(
        [u * z, -u * y, u * x],
        [(x, -3, 3), (y, -2, 2), (z, -1, 1)],
        "",
        params={u: 1},
        slice=Plane((-1, 0, 0), (1, 0, 0)),
        n1=15,
        n2=15,
        n3=15,
    )
    s2 = SliceVector3DSeries(
        Plane((-1, 0, 0), (1, 0, 0)),
        z,
        -y,
        x,
        (x, -3, 3),
        (y, -2, 2),
        (z, -1, 1),
        "",
        n1=15,
        n2=15,
        n3=15,
    )
    do_test(s1.get_data(), s2.get_data())

    ### Test InteractiveSeries and ComplexInteractiveSeries with complex
    ### functions

    # complex function evaluated over a real line with numpy
    s1 = InteractiveSeries(
        [(z ** 2 + 1) / (z ** 2 - 1)], [(z, -3, 3)], "", n1=50,
        is_complex=True, modules=None)
    s2 = LineOver1DRangeSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3, 3), "", adaptive=False,
        n=50, is_complex=True, modules=None)
    do_test(s1.get_data(), s2.get_data())

    # complex function evaluated over a real line with mpmath
    s1 = InteractiveSeries(
        [(z ** 2 + 1) / (z ** 2 - 1)], [(z, -3, 3)], "",
        n1=11, is_complex=True, modules="mpmath")
    s2 = LineOver1DRangeSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3, 3), "", adaptive=False,
        n=11, is_complex=True, modules="mpmath")
    do_test(s1.get_data(), s2.get_data())

    # abs/arg values of complex function evaluated over a real line wit numpy
    expr = (z ** 2 + 1) / (z ** 2 - 1)
    s1 = InteractiveSeries(
        [expr], [(z, -3, 3)], "",
        n1=50, is_complex=True, absarg=expr, modules=None)
    s2 = LineOver1DRangeSeries(
        expr, (z, -3, 3), "", adaptive=False,
        n=50, is_complex=True, absarg=expr, modules=None)
    do_test(s1.get_data(), s2.get_data())

    # abs/arg values of complex function evaluated over a real line wit mpmath
    expr = (z ** 2 + 1) / (z ** 2 - 1)
    s1 = InteractiveSeries(
        [expr], [(z, -3, 3)], "",
        n1=50, is_complex=True, absarg=expr, modules="mpmath")
    s2 = LineOver1DRangeSeries(
        expr, (z, -3, 3), "", adaptive=False,
        n=50, is_complex=True, absarg=expr, modules="mpmath")
    do_test(s1.get_data(), s2.get_data())

    # domain coloring or 3D
    s1 = ComplexInteractiveSeries(
        u * (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I), "",
        n1=20, n2=20, domain_coloring=True, params = {u: 1}, modules=None
    )
    s2 = ComplexSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I), "",
        n1=20, n2=20, domain_coloring=True, modules=None
    )
    do_test(s1.get_data(), s2.get_data())

def test_complex_discretization():
    x, y, z = symbols("x:z")

    # test complex discretization for LineOver1DRangeSeries and
    # SurfaceOver2DRangeSeries and InteractiveSeries

    # Mpmath and Numpy produces different results
    s1 = LineOver1DRangeSeries(im(sqrt(-x)), (x, -10, 10), "s1",
            adaptive=False, is_complex=True, modules=None, n=10)
    s2 = LineOver1DRangeSeries(im(sqrt(-x)), (x, -10, 10), "s1",
            adaptive=False, is_complex=True, modules="mpmath", n=10)
    d1, d2 = s1.get_data(), s2.get_data()
    assert (d1[-1][-1] < 0) and (d2[-1][-1] > 0)
    assert np.array_equal(d1[-1], -d2[-1])

    def do_test(data1, data2, compare=True):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert (d1.dtype == np.float64) and (d2.dtype == np.float64)
            if compare:
                assert np.array_equal(d1, d2)

    # using Numpy and a real discretization will produce NaN value when x<0.
    with warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        s1 = LineOver1DRangeSeries(sqrt(x), (x, -10, 10), "s1",
                adaptive=False, is_complex=False, modules=None, n=20)
        s1.get_data()

    # using Numpy or Mpmath with complex discretization won't raise warnings.
    # Results between Numpy as Mpmath shoudl be really close
    s2 = LineOver1DRangeSeries(sqrt(x), (x, -10, 10), "s2",
            adaptive=False, is_complex=True, modules=None, n=20)
    s3 = LineOver1DRangeSeries(sqrt(x), (x, -10, 10), "s3",
            adaptive=False, is_complex=True, modules="mpmath", n=20)
    do_test(s2.get_data(), s3.get_data())


    # using Numpy and a real discretization will produce NaN value when x<0.
    with warns(RuntimeWarning, match="invalid value encountered in sqrt"):
        s4 = LineOver1DRangeSeries(sqrt(x), (x, -10, 10), "s4",
                adaptive=True, is_complex=False, modules=None)
        s4.get_data()

    # using Numpy or Mpmath with complex discretization won't raise warnings.
    # Results between Numpy as Mpmath shoudl be really close.
    # NOTE: changed the function because the adaptive algorithm works by
    # checking the collinearity between three points (the x, y coordinates must
    # be real). Instead, with "mpmath" the y coordinate is a complex number.
    s5 = LineOver1DRangeSeries(im(sqrt(x)), (x, -10, 10), "s5",
            adaptive=True, is_complex=True, modules=None)
    s6 = LineOver1DRangeSeries(im(sqrt(x)), (x, -10, 10), "s6",
            adaptive=True, is_complex=True, modules="mpmath")
    # can't directly compare the results because of the adaptive sampling
    do_test(s5.get_data(), s6.get_data(), False)


    # Mpmath and Numpy produces different results
    s1 = SurfaceOver2DRangeSeries(im(sqrt(-x)), (x, -5, 5), (y, -5, 5),
            is_complex=False, modules=None)
    s2 = SurfaceOver2DRangeSeries(im(sqrt(-x)), (x, -5, 5), (y, -5, 5),
            is_complex=True, modules="mpmath")
    d1, d2 = s1.get_data(), s2.get_data()
    assert (d1[-1][-1, -1] < 0) and (d2[-1][-1, -1] > 0)
    assert np.all(np.abs(d1[-1]) - np.abs(d2[-1])) < 1e-08

    # Interactive series produces the same numerical data as LineOver1DRangeSeries.
    # NOTE: InteractiveSeries doesn't support adaptive algorithm!
    s1 = LineOver1DRangeSeries(im(sqrt(-x)), (x, -10, 10), "s1",
            adaptive=False, is_complex=True, modules=None, n=10)
    s2 = InteractiveSeries([im(sqrt(-x))], [(x, -10, 10)], "s2",
            is_complex=True, modules=None, n1=10)
    s3 = InteractiveSeries([im(sqrt(-x))], [(x, -10, 10)], "s3",
            is_complex=True, modules="mpmath", n1=10)
    d1, d2, d3 = s1.get_data(), s2.get_data(), s3.get_data()
    do_test(d1, d2)
    assert np.all(np.abs(d1[-1]) - np.abs(d3[-1])) < 1e-08

    expr = sqrt(-x)
    s1 = LineOver1DRangeSeries(expr, (x, -10, 10), "s1",
            adaptive=False, is_complex=True, modules=None, n=10, absarg=expr)
    s2 = InteractiveSeries([expr], [(x, -10, 10)], "s2",
            is_complex=True, modules=None, n1=10, absarg=expr)
    s3 = InteractiveSeries([expr], [(x, -10, 10)], "s3",
            is_complex=True, modules="mpmath", n1=10, absarg=expr)
    d1, d2, d3 = s1.get_data(), s2.get_data(), s3.get_data()
    do_test(d1, d2)
    assert np.all(np.abs(d1[-1]) - np.abs(d3[-1])) < 1e-08

    # Interactive series produces the same numerical data as SurfaceOver2DRangeSeries
    s1 = SurfaceOver2DRangeSeries(im(sqrt(-x)), (x, -3, 3), (y, -3, 3),
            is_complex=True, modules="mpmath", n1=20, n2=20)
    s2 = InteractiveSeries([im(sqrt(-x))], [(x, -3, 3), (y, -3, 3)], "s2",
            is_complex=True, modules=None, n1=20, n2=20)
    s3 = InteractiveSeries([im(sqrt(-x))], [(x, -3, 3), (y, -3, 3)], "s3",
            is_complex=True, modules="mpmath", n1=20, n2=20)
    do_test(d1, d2)
    assert np.all(np.abs(d1[-1]) - np.abs(d3[-1])) < 1e-08
