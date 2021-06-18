from sympy import (
    symbols, cos, sin, log, Eq, I, Abs, exp, pi, gamma, Matrix, Tuple
)
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries, InteractiveSeries,
    ImplicitSeries, Vector2DSeries, Vector3DSeries, ComplexSeries,
    ComplexInteractiveSeries
)
import numpy as np

def test_interactive():
    u, x, y, z = symbols("u, x:z")

    # verify that InteractiveSeries produces the same numerical data as their
    # corresponding non-interactive series.
    def do_test(data1, data2):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert np.array_equal(d1, d2)

    s1 = InteractiveSeries([u * cos(x)], [(x, -5, 5)], "", params={u: 1}, n1=50)
    s2 = LineOver1DRangeSeries(cos(x), (x, -5, 5), "", adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())
    
    s1 = InteractiveSeries([u * cos(x), u* sin(x)], [(x, -5, 5)], "", params={u: 1}, n1=50)
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5), "", adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())
    
    s1 = InteractiveSeries([u * cos(x), u* sin(x), u * x], [(x, -5, 5)], "", 
            params={u: 1}, n1=50)
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5), "", 
            adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())
    
    s1 = InteractiveSeries([cos(x**2 + y**2)], [(x, -3, 3), (y, -3, 3)], "", 
            params={u: 1}, n1=50, n2=50)
    s2 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), "", 
            adaptive=False, n1=50, n2=50)
    do_test(s1.get_data(), s2.get_data())
    
    s1 = InteractiveSeries([cos(x + y), sin(x + y), x - y], 
            [(x, -3, 3), (y, -3, 3)], "", 
            params={u: 1}, n1=50, n2=50)
    s2 = ParametricSurfaceSeries(cos(x + y), sin(x + y), x - y, 
            (x, -3, 3), (y, -3, 3), "", 
            adaptive=False, n1=50, n2=50)
    do_test(s1.get_data(), s2.get_data())
    
    s1 = InteractiveSeries([-u * y, u * x], [(x, -3, 3), (y, -2, 2)], 
            "", params={u: 1}, n1=15, n2=15)
    s2 = Vector2DSeries(-y, x, (x, -3, 3), (y, -2, 2), "", 
            n1=15, n2=15)
    do_test(s1.get_data(), s2.get_data())
    
    s1 = InteractiveSeries([u * z, -u * y, u * x], [(x, -3, 3), (y, -2, 2),
            (z, -1, 1)], "", params={u: 1}, n1=15, n2=15, n3=15)
    s2 = Vector3DSeries(z, -y, x, (x, -3, 3), (y, -2, 2), (z, -1, 1), "", 
            n1=15, n2=15, n3=15)
    do_test(s1.get_data(), s2.get_data())

    
    ### Test ComplexInteractiveSeries

    # real and imag
    s1 = ComplexInteractiveSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50)
    s2 = ComplexSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50)
    do_test(s1.get_data(), s2.get_data())

    # only real
    s1 = ComplexInteractiveSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50, imag=False)
    s2 = ComplexSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50, imag=False)
    do_test(s1.get_data(), s2.get_data())

    # only imag
    s1 = ComplexInteractiveSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50, real=False)
    s2 = ComplexSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50, real=False)
    do_test(s1.get_data(), s2.get_data())

    # magnitude and argument
    s1 = ComplexInteractiveSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50, absarg=True)
    s2 = ComplexSeries((z**2 + 1) / (z**2 - 1), (z, -3, 3), 
            "", n1=50, absarg=True)
    do_test(s1.get_data(), s2.get_data())

    # domai coloring or 3D
    s1 = ComplexInteractiveSeries((z**2 + 1) / (z**2 - 1), 
            (z, -3 - 4 * I, 3 + 4 * I), "", n1=50)
    s2 = ComplexSeries((z**2 + 1) / (z**2 - 1),
            (z, -3 - 4 * I, 3 + 4 * I), "", n1=50)
    do_test(s1.get_data(), s2.get_data())


