
from spb.plot_functions.control import  (
    plot_pole_zero, plot_step_response, plot_impulse_response,
    plot_ramp_response, plot_bode_magnitude,
    plot_bode_phase, plot_bode, plot_nyquist, plot_nichols)
from spb.interactive import IPlot
from spb.series import HVLineSeries
from spb.backends.matplotlib import unset_show
from sympy import Dummy, I, Abs, arg, log, symbols, exp, latex
from sympy.abc import s, p, a, b
from sympy.external import import_module
from sympy.physics.control.lti import (TransferFunction,
    Series, Parallel, TransferFunctionMatrix)
from sympy.testing.pytest import raises, skip
import numpy as np
import pytest

unset_show()

tf1 = TransferFunction(1, p**2 + 0.5*p + 2, p)
tf2 = TransferFunction(p, 6*p**2 + 3*p + 1, p)
tf3 = TransferFunction(p, p**3 - 1, p)
tf4 = TransferFunction(10, p**3, p)
tf5 = TransferFunction(5, s**2 + 2*s + 10, s)
tf6 = TransferFunction(1, 1, s)
tf7 = TransferFunction(4*s*3 + 9*s**2 + 0.1*s + 11, 8*s**6 + 9*s**4 + 11, s)
tf8 = TransferFunction(5, s**2 + (2+I)*s + 10, s)

ser1 = Series(tf4, TransferFunction(1, p - 5, p))
ser2 = Series(tf3, TransferFunction(p, p + 2, p))

par1 = Parallel(tf1, tf2)

num1, den1 = p**2 + 1, p**4 + 4*p**3 + 6*p**2 + 5*p + 2
tf_test1 = TransferFunction(num1, den1, p)
num2, den2 = p, (p + a) * (p + b)
tf_test2 = TransferFunction(num2, den2, p)
test_params = {
    a: (3, 0, 5),
    b: (5, 0, 10),
}

def _to_tuple(a, b):
    return tuple(a), tuple(b)

def _trim_tuple(a, b):
    a, b = _to_tuple(a, b)
    return tuple(a[0: 2] + a[len(a)//2 : len(a)//2 + 1] + a[-2:]), \
        tuple(b[0: 2] + b[len(b)//2 : len(b)//2 + 1] + b[-2:])

def y_coordinate_equality(plot_data_func, evalf_func, system):
    """Checks whether the y-coordinate value of the plotted
    data point is equal to the value of the function at a
    particular x."""
    p = plot_data_func(system, show=False, show_axes=False, n=10)
    x, y = p[0].get_data()
    x, y = _trim_tuple(x, y)
    y_exp = tuple(evalf_func(system, x_i) for x_i in x)
    return all(Abs(y_exp_i - y_i) < 1e-8 for y_exp_i, y_i in zip(y_exp, y))


def test_errors():

    # Invalid `system` check
    tfm = TransferFunctionMatrix([[tf6, tf5], [tf5, tf6]])
    expr = 1/(s**2 - 1)
    raises(NotImplementedError, lambda: plot_pole_zero(tfm))
    raises(NotImplementedError, lambda: plot_step_response(tfm))
    raises(NotImplementedError, lambda: plot_bode(tfm))

    # More than 1 variables: raise error because `params` is missing
    tf_a = TransferFunction(a, s + a, s)
    raises(ValueError, lambda: plot_pole_zero(tf_a))
    raises(ValueError, lambda: plot_impulse_response(tf_a))
    raises(ValueError, lambda: plot_step_response(tf_a))
    raises(ValueError, lambda: plot_ramp_response(tf_a))
    raises(ValueError, lambda: plot_bode(tf_a))

    # lower_limit > 0 for response plots
    raises(ValueError, lambda: plot_impulse_response(tf1, lower_limit=-1))
    raises(ValueError, lambda: plot_step_response(tf1, lower_limit=-0.1))
    raises(ValueError, lambda: plot_ramp_response(tf1, lower_limit=-4/3))

    # slope in plot_ramp_response() is negative
    raises(ValueError, lambda: plot_ramp_response(tf1, slope=-0.1))

    # incorrect frequency or phase unit
    raises(ValueError, lambda: plot_bode(tf1,freq_unit = 'hz'))
    raises(ValueError, lambda: plot_bode(tf1,phase_unit = 'degree'))


def test_pole_zero():

    def pz_tester(sys, expected_value):
        plot = plot_pole_zero(sys, show_axes=False, show=False)
        xxp, yyp = plot[0].get_data()
        xxz, yyz = plot[1].get_data()
        p = xxp + 1j * yyp
        z = xxz + 1j * yyz
        z_check = np.allclose(z, expected_value[0])
        p_check = np.allclose(p, expected_value[1])
        return p_check and z_check

    exp1 = [[], [-0.24999999999999994+1.3919410907075054j, -0.24999999999999994-1.3919410907075054j]]
    exp2 = [[0.0], [-0.25+0.3227486121839514j, -0.25-0.3227486121839514j]]
    exp3 = [[0.0], [-0.5000000000000004+0.8660254037844395j,
        -0.5000000000000004-0.8660254037844395j, 0.9999999999999998+0j]]
    exp4 = [[], [5.0, 0.0, 0.0, 0.0]]
    exp5 = [[-5.645751311064592, -0.5000000000000008, -0.3542486889354093],
        [-0.24999999999999986+1.3919410907075052j,
        -0.24999999999999986-1.3919410907075052j, -0.2499999999999998+0.32274861218395134j,
        -0.2499999999999998-0.32274861218395134j]]
    exp6 = [[], [-1.1641600331447917-3.545808351896439j,
          -0.8358399668552097+2.5458083518964383j]]

    assert pz_tester(tf1, exp1)
    assert pz_tester(tf2, exp2)
    assert pz_tester(tf3, exp3)
    assert pz_tester(ser1, exp4)
    assert pz_tester(par1, exp5)
    assert pz_tester(tf8, exp6)


def test_bode():

    def bode_phase_evalf(system, point):
        expr = system.to_expr()
        _w = Dummy("w", real=True)
        w_expr = expr.subs({system.var: I*_w})
        return arg(w_expr).subs({_w: point}).evalf()

    def bode_mag_evalf(system, point):
        expr = system.to_expr()
        _w = Dummy("w", real=True)
        w_expr = expr.subs({system.var: I*_w})
        return 20*log(Abs(w_expr), 10).subs({_w: point}).evalf()

    def test_bode_data(sys):
        return y_coordinate_equality(plot_bode_magnitude, bode_mag_evalf, sys) \
            and y_coordinate_equality(plot_bode_phase, bode_phase_evalf, sys)

    assert test_bode_data(tf1)
    assert test_bode_data(tf2)
    assert test_bode_data(tf3)
    assert test_bode_data(tf4)
    assert test_bode_data(tf5)


def test_plot_bode_phase_unwrap_1():
    s = symbols("s")
    G = 1 / (s * (s + 1) * (s + 10))
    p1 = plot_bode_phase(G, phase_unit="deg", initial_exp=-2, final_exp=1,
        n=1000, show=False)
    p2 = plot_bode_phase(G, phase_unit="deg", initial_exp=-2, final_exp=1,
        n=1000, show=False, unwrap=True)
    p3 = plot_bode_phase(G, phase_unit="deg", initial_exp=-2, final_exp=1,
        n=1000, show=False, unwrap=False)
    s1 = p1.series[0]
    s2 = p2.series[0]
    s3 = p3.series[0]
    _, y1 = s1.get_data()
    _, y2 = s2.get_data()
    _, y3 = s3.get_data()
    assert np.allclose(y1, y2)
    assert not np.allclose(y1, y3)
    assert y1[0] < 0 and y1[-1] < y1[1]
    assert y3[0] < 0 and y3[-1] > 0


def test_plot_bode_phase_unwrap_2():
    # verify that unwrap produces the correct results
    s = symbols("s")
    G = 1 / (s+5) / (s+20) / (s+50)
    p1 = plot_bode_phase(G, phase_unit="rad", initial_exp=0, final_exp=3,
        unwrap=False, n=10)
    p2 = plot_bode_phase(G, phase_unit="rad", initial_exp=0, final_exp=3,
        unwrap=True, n=10)
    p3 = plot_bode_phase(G, phase_unit="deg", initial_exp=0, final_exp=3,
        unwrap=False, n=10)
    p4 = plot_bode_phase(G, phase_unit="deg", initial_exp=0, final_exp=3,
        unwrap=True, n=10)
    x, y1 = p1[0].get_data()
    _, y2 = p2[0].get_data()
    _, y3 = p3[0].get_data()
    _, y4 = p4[0].get_data()
    assert np.allclose(
        x,
        [
            1., 2.15443469, 4.64158883, 10., 21.5443469, 46.41588834, 100.,
            215.443469, 464.15888336, 1000.
        ]
    )
    assert np.allclose(
        y1,
        [
            -0.26735129, -0.55721635, -1.06885075, -1.76819189, -2.57215473,
            2.90750513, 2.28179789, 1.91460904, 1.73193809, 1.64575201
        ]
    )
    assert np.allclose(
        y2,
        [
            -0.26735129, -0.55721635, -1.06885075, -1.76819189, -2.57215473,
            -3.37568017, -4.00138742, -4.36857627, -4.55124722, -4.63743329
        ]
    )
    assert np.allclose(
        y3,
        [
            -15.31810054,  -31.92614531,  -61.24063705, -101.30993247,
            -147.37361045, 166.58777301, 130.73738888, 109.69901745,
            99.23274267, 94.29464457
        ]
    )
    assert np.allclose(
        y4,
        [
            -15.31810054, -31.92614531, -61.24063705, -101.30993247,
            -147.37361045, -193.41222699, -229.26261112, -250.30098255,
            -260.76725733, -265.70535543
        ]
    )



def test_bode_plot_delay():
    s = symbols("s")
    G1 = 1 / (s * (s + 1) * (s + 10))
    G2 = G1 * exp(-5*s)
    p1 = plot_bode_magnitude(G1, G2, initial_exp=-2, final_exp=1,
        n=1000, show=False)
    p2 = plot_bode_phase(G1, G2, phase_unit="deg",
        initial_exp=-2, final_exp=1, n=1000, show=False)
    _, y1 = p1[0].get_data()
    _, y2 = p1[1].get_data()
    assert np.allclose(y1, y2)
    _, y3 = p2[0].get_data()
    _, y4 = p2[1].get_data()
    assert not np.allclose(y3, y4)


def check_point_accuracy(a, b):
    return all(np.isclose(a_i, b_i) for a_i, b_i in zip(a, b))


def test_impulse_response():

    def impulse_res_tester(sys, expected_value):
        p = plot_impulse_response(sys,
            control=False, show_axes=False, show=False, n=10, prec=16)
        x, y = p[0].get_data()
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.544019738507865, 0.01993849743234938, -0.31140243360893216, -0.022852779906491996, 0.1778306498155759,
        0.01962941084328499, -0.1013115194573652, -0.014975541213105696, 0.0575789724730714))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.1666666675, 0.08389223412935855,
        0.02338051973475047, -0.014966807776379383, -0.034645954223054234, -0.040560075735512804,
        -0.037658628907103885, -0.030149507719590022, -0.021162090730736834, -0.012721292737437523))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (4.369893391586999e-09, 1.1750333000630964,
        3.2922404058312473, 9.432290008148343, 28.37098083007151, 86.18577464367974, 261.90356653762115,
        795.6538758627842, 2416.9920942096983, 7342.159505206647))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 6.17283950617284, 24.69135802469136,
        55.555555555555564, 98.76543209876544, 154.320987654321, 222.22222222222226, 302.46913580246917,
        395.0617283950618, 500.0))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, -0.10455606138085417,
        0.06757671513476461, -0.03234567568833768, 0.013582514927757873, -0.005273419510705473,
        0.0019364083003354075, -0.000680070134067832, 0.00022969845960406913, -7.476094359583917e-05))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-6.016699583000218e-09, 0.35039802056107394, 3.3728423827689884, 12.119846079276684,
        25.86101014293389, 29.352480635282088, -30.49475907497664, -273.8717189554019, -863.2381702029659,
        -1747.0262164682233))
    exp7 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335,
        4.444444444444445, 5.555555555555555, 6.666666666666667, 7.777777777777779,
        8.88888888888889, 10.0), (0.0, 18.934638095560974, 5346.93244680907, 1384609.8718249386,
        358161126.65801865, 92645770015.70108, 23964739753087.42, 6198974342083139.0, 1.603492601616059e+18,
        4.147764422869658e+20))

    assert impulse_res_tester(tf1, exp1)
    assert impulse_res_tester(tf2, exp2)
    assert impulse_res_tester(tf3, exp3)
    assert impulse_res_tester(tf4, exp4)
    assert impulse_res_tester(tf5, exp5)
    assert impulse_res_tester(tf7, exp6)
    assert impulse_res_tester(ser1, exp7)


def test_step_response():

    def step_res_tester(sys, expected_value):
        p = plot_step_response(sys,
            control=False, show_axes=False, show=False, n=10)
        x, y = p[0].get_data()
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-1.9193285738516863e-08, 0.42283495488246126, 0.7840485977945262, 0.5546841805655717,
        0.33903033806932087, 0.4627251747410237, 0.5909907598988051, 0.5247213989553071,
        0.4486997874319281, 0.4839358435839171))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.13728409095645816, 0.19474559355325086, 0.1974909129243011, 0.16841657696573073,
        0.12559777736159378, 0.08153828016664713, 0.04360471317348958, 0.015072994568868221,
        -0.003636420058445484))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.6314542141914303, 2.9356520038101035, 9.37731009663807, 28.452300356688376,
        86.25721933273988, 261.9236645044672, 795.6435410577224, 2416.9786984578764, 7342.154119725917))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 2.286236899862826, 18.28989519890261, 61.72839629629631, 146.31916159122088, 285.7796124828532,
        493.8271703703705, 784.1792566529494, 1170.553292729767, 1666.6667))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-3.999999997894577e-09, 0.6720357068882895, 0.4429938256137113, 0.5182010838004518,
        0.4944139147159695, 0.5016379853883338, 0.4995466896527733, 0.5001154784851325,
        0.49997448824584123, 0.5000039745919259))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-1.5433688493882158e-09, 0.3428705539937336, 1.1253619102202777, 3.1849962651016517,
        9.47532757182671, 28.727231099148135, 87.29426924860557, 265.2138681048606, 805.6636260007757,
        2447.387582370878))

    assert step_res_tester(tf1, exp1)
    assert step_res_tester(tf2, exp2)
    assert step_res_tester(tf3, exp3)
    assert step_res_tester(tf4, exp4)
    assert step_res_tester(tf5, exp5)
    assert step_res_tester(ser2, exp6)


def test_ramp_response():

    def ramp_res_tester(sys, num_points, expected_value, slope=1):
        p = plot_ramp_response(sys,
            control=False, show_axes=False, show=False,
            slope=slope, n=num_points)
        x, y = p[0].get_data()
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 2.0, 4.0, 6.0, 8.0, 10.0), (0.0, 0.7324667795033895, 1.9909720978650398,
        2.7956587704217783, 3.9224897567931514, 4.85022655284895))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (2.4360213402019326e-08, 0.10175320182493253, 0.33057612497658406, 0.5967937263298935,
        0.8431511866718248, 1.0398805391471613, 1.1776043125035738, 1.2600994825747305, 1.2981042689274653,
        1.304684417610106))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (-3.9329040468771836e-08,
        0.34686634635794555, 2.9998828170537903, 12.33303690737476, 40.993913948137795, 127.84145222317912,
        391.41713691996, 1192.0006858708389, 3623.9808672503405, 11011.728034546572))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 1.9051973784484078, 30.483158055174524,
        154.32098765432104, 487.7305288827924, 1190.7483615302544, 2469.1358024691367, 4574.3789056546275,
        7803.688462124678, 12500.0))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 3.8844361856975635, 9.141792069209865,
        14.096349157657231, 19.09783068994694, 24.10179770390321, 29.09907319114121, 34.10040420185154,
        39.09983919254265, 44.10006013058409))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 1.1111111111111112, 2.2222222222222223,
        3.3333333333333335, 4.444444444444445, 5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0))

    assert ramp_res_tester(tf1, 6, exp1)
    assert ramp_res_tester(tf2, 10, exp2, 1.2)
    assert ramp_res_tester(tf3, 10, exp3, 1.5)
    assert ramp_res_tester(tf4, 10, exp4, 3)
    assert ramp_res_tester(tf5, 10, exp5, 9)
    assert ramp_res_tester(tf6, 10, exp6)


def test_show_axes():
    def do_test(plot_func, sys, expected_num_series):
        p1 = plot_func(sys, show=False, show_axes=False)
        p2 = plot_func(sys, show=False, show_axes=True)
        assert len(p1.series) == expected_num_series
        assert len(p2.series) == expected_num_series + 2
        assert all(not isinstance(t, HVLineSeries) for t in p1.series)
        assert all(isinstance(t, HVLineSeries) for t in p2.series[:2])
        assert all(t.show_in_legend is False for t in p2.series[:2])
        p1.draw()
        p2.draw()
        assert len(p1.ax.lines) == expected_num_series
        assert len(p2.ax.lines) == expected_num_series + 2
        # axes lines should not be visibile on legend
        handles1, _ = p1.ax.get_legend_handles_labels()
        handles2, _ = p2.ax.get_legend_handles_labels()
        assert len(handles1) == len(handles2) == expected_num_series

    do_test(plot_pole_zero, tf1, 2)
    do_test(plot_impulse_response, tf1, 1)
    do_test(plot_step_response, tf1, 1)
    do_test(plot_ramp_response, tf1, 1)
    do_test(plot_bode_magnitude, tf1, 1)
    do_test(plot_bode_phase, tf1, 1)


def test_interactive_plots():
    tf_a = TransferFunction(a, s + a, s)
    def do_test(plot_func):
        p = plot_func(tf_a, show=False, params={a: (1, 0, 2)})
        assert isinstance(p, IPlot)

    do_test(plot_pole_zero)
    do_test(plot_impulse_response)
    do_test(plot_step_response)
    do_test(plot_ramp_response)
    do_test(plot_bode_magnitude)
    do_test(plot_bode_phase)


# xfail because tf7... who knows?!?!? locally works fine, on github it's
# random success
@pytest.mark.xfail
def test_plot_nyquist():
    exp1 = (
        [ 5.00000000e-01,  5.00169418e-01,  5.01314254e-01,  5.10328032e-01,
         5.89837668e-01, -5.98269135e-01, -5.04542656e-02, -6.05828968e-03,
        -7.75313970e-04, -1.00017503e-04],
        [-0.00000000e+00, -3.48072529e-03, -9.73290914e-03, -2.81397767e-02,
        -1.07763738e-01, -6.37634065e-01, -5.99119421e-03, -2.37381817e-04,
        -1.08035154e-05, -5.00187550e-07],
        [0.+0.00000000e+00j, 0.+2.78255940e-02j, 0.+7.74263683e-02j,
        0.+2.15443469e-01j, 0.+5.99484250e-01j, 0.+1.66810054e+00j,
        0.+4.64158883e+00j, 0.+1.29154967e+01j, 0.+3.59381366e+01j,
        0.+1.00000000e+02j]
    )
    exp2 = (
        [0.00000000e+00, 2.32814861e-03, 1.82897980e-02, 1.48402309e-01,
        2.35843706e-01, 3.07592869e-02, 3.88277687e-03, 4.99819405e-04,
        6.45261358e-05, 8.33340278e-06],
        [ 0  ,  0.02776021,  0.07590838,  0.16566288, -0.15163217,
        -0.0964726 , -0.03576567, -0.01289793, -0.0046373 , -0.00166665],
        [0.+0.00000000e+00j, 0.+2.78255940e-02j, 0.+7.74263683e-02j,
        0.+2.15443469e-01j, 0.+5.99484250e-01j, 0.+1.66810054e+00j,
        0.+4.64158883e+00j, 0.+1.29154967e+01j, 0.+3.59381366e+01j,
        0.+1.00000000e+02j]
    )
    exp3 = (
        [-0.00000000e+00, -5.99484250e-07, -3.59381289e-05, -2.15421927e-03,
        -1.23426037e-01, -3.43440281e-01, -4.64112472e-02, -5.99484121e-03,
        -7.74263682e-04, -1.00000000e-04],
        [-0.00000000e+00, -2.78255940e-02, -7.74263516e-02, -2.15421927e-01,
        -5.72892917e-01, -7.39919655e-02, -4.64112472e-04, -2.78255880e-06,
        -1.66810054e-08, -1.00000000e-10],
        [0.+0.00000000e+00j, 0.+2.78255940e-02j, 0.+7.74263683e-02j,
        0.+2.15443469e-01j, 0.+5.99484250e-01j, 0.+1.66810054e+00j,
        0.+4.64158883e+00j, 0.+1.29154967e+01j, 0.+3.59381366e+01j,
        0.+1.00000000e+02j]
    )
    # for particular cases, the function injects further points to compute
    # the correct result
    exp4 = (
        [ 1.00000000e+13,  1.00000000e+13,  9.98126106e+12,  9.92508330e+12,
         9.83158397e+12,  9.70095889e+12,  9.53348301e+12,  9.32951122e+12,
         9.08947946e+12,  8.81390596e+12,  8.50339295e+12,  8.15862850e+12,
         7.78038873e+12,  7.36954047e+12,  6.92704413e+12,  6.45395714e+12,
         5.95143778e+12,  5.42074962e+12,  4.86326649e+12,  4.28047817e+12,
         3.67399687e+12,  3.04556463e+12,  2.39706178e+12,  1.73051660e+12,
         1.04811646e+12,  3.52220743e+11, -3.54624254e+11, -1.06966784e+12,
        -1.78993444e+12, -2.51219941e+12, -3.23296011e+12, -3.94840133e+12,
        -4.65435320e+12, -5.34623977e+12, -6.01901519e+12, -6.66708356e+12,
        -7.28419690e+12, -7.86332282e+12, -8.39646977e+12, -8.87445121e+12,
        -9.28655884e+12, -9.62009636e+12, -9.85968962e+12, -9.98622073e+12,
        -9.97508928e+12, -9.79317264e+12, -9.39300037e+12, -8.70005019e+12,
        -7.57898287e+12, -5.70510297e+12,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00, -6.11904904e+11, -1.22176984e+12,
        -1.82755485e+12, -2.42721995e+12, -3.01872519e+12, -3.60003060e+12,
        -4.16909621e+12, -4.72388206e+12, -5.26234817e+12, -5.78245459e+12,
        -6.28216134e+12, -6.75942847e+12, -7.21221600e+12, -7.63848397e+12,
        -8.03619240e+12, -8.40330135e+12, -8.73777083e+12, -9.03756088e+12,
        -9.30063154e+12, -9.52494284e+12, -9.70845481e+12, -9.84912749e+12,
        -9.94492091e+12, -9.99379510e+12, -9.99371010e+12, -9.94262595e+12,
        -9.83850266e+12, -9.67930029e+12, -9.46297886e+12, -9.18749841e+12,
        -8.85081896e+12, -8.45090056e+12, -7.98570324e+12, -7.45318702e+12,
        -6.85131195e+12, -6.17803806e+12, -5.43132538e+12, -4.60913395e+12,
        -3.70942379e+12, -2.73015495e+12, -1.66928746e+12, -5.24781341e+11,
         7.05403361e+11,  2.02330662e+12,  3.43096839e+12,  4.93042865e+12,
         6.52372736e+12,  8.21290449e+12,  1.00000000e+13,  4.64158883e+05,
         2.15443469e+04,  1.00000000e+03,  4.64158883e+01,  2.15443469e+00,
         1.00000000e-01,  4.64158883e-03,  2.15443469e-04,  1.00000000e-05],
        [1.00000000e-04+0.00000000e+00j, 1.00000000e-04+0.00000000e+00j,
        9.99791732e-05+2.04081633e-06j, 9.99166667e-05+4.08163265e-06j,
        9.98124021e-05+6.12244898e-06j, 9.96662485e-05+8.16326531e-06j,
        9.94780213e-05+1.02040816e-05j, 9.92474809e-05+1.22448980e-05j,
        9.89743319e-05+1.42857143e-05j, 9.86582201e-05+1.63265306e-05j,
        9.82987313e-05+1.83673469e-05j, 9.78953874e-05+2.04081633e-05j,
        9.74476440e-05+2.24489796e-05j, 9.69548859e-05+2.44897959e-05j,
        9.64164229e-05+2.65306122e-05j, 9.58314847e-05+2.85714286e-05j,
        9.51992146e-05+3.06122449e-05j, 9.45186627e-05+3.26530612e-05j,
        9.37887779e-05+3.46938776e-05j, 9.30083989e-05+3.67346939e-05j,
        9.21762432e-05+3.87755102e-05j, 9.12908949e-05+4.08163265e-05j,
        9.03507903e-05+4.28571429e-05j, 8.93542011e-05+4.48979592e-05j,
        8.82992149e-05+4.69387755e-05j, 8.71837117e-05+4.89795918e-05j,
        8.60053368e-05+5.10204082e-05j, 8.47614680e-05+5.30612245e-05j,
        8.34491767e-05+5.51020408e-05j, 8.20651807e-05+5.71428571e-05j,
        8.06057864e-05+5.91836735e-05j, 7.90668189e-05+6.12244898e-05j,
        7.74435345e-05+6.32653061e-05j, 7.57305115e-05+6.53061224e-05j,
        7.39215113e-05+6.73469388e-05j, 7.20093011e-05+6.93877551e-05j,
        6.99854212e-05+7.14285714e-05j, 6.78398781e-05+7.34693878e-05j,
        6.55607282e-05+7.55102041e-05j, 6.31335033e-05+7.75510204e-05j,
        6.05403958e-05+7.95918367e-05j, 5.77590682e-05+8.16326531e-05j,
        5.47608484e-05+8.36734694e-05j, 5.15078754e-05+8.57142857e-05j,
        4.79483270e-05+8.77551020e-05j, 4.40078748e-05+8.97959184e-05j,
        3.95728968e-05+9.18367347e-05j, 3.44529449e-05+9.38775510e-05j,
        2.82783805e-05+9.59183673e-05j, 2.00997098e-05+9.79591837e-05j,
        0.00000000e+00+1.00000000e-04j, 0.00000000e+00+2.78255940e-02j,
        0.00000000e+00+7.74263683e-02j, 0.00000000e+00+2.15443469e-01j,
        0.00000000e+00+5.99484250e-01j, 0.00000000e+00+1.66810054e+00j,
        0.00000000e+00+4.64158883e+00j, 0.00000000e+00+1.29154967e+01j,
        0.00000000e+00+3.59381366e+01j, 0.00000000e+00+1.00000000e+02j]
    )
    exp5 = (
        [ 5.00000000e-01,  5.02321436e-01,  5.17834133e-01,  5.66680725e-01,
        -1.58829654e-01, -1.83550194e-02, -2.32725557e-03, -2.99849935e-04,
        -3.87149826e-05, -5.00003000e-06],
        [-0.00000000e+00, -2.81729174e-02, -8.53017374e-02, -4.55686051e-01,
        -7.34176688e-02, -2.28274667e-03, -1.00746024e-04, -4.64604353e-06,
        -2.15470161e-07, -1.00001600e-08],
        [0.+0.00000000e+00j, 0.+2.78255940e-01j, 0.+7.74263683e-01j,
        0.+2.15443469e+00j, 0.+5.99484250e+00j, 0.+1.66810054e+01j,
        0.+4.64158883e+01j, 0.+1.29154967e+02j, 0.+3.59381366e+02j,
        0.+1.00000000e+03j]
    )
    exp6 = (
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0.+0.00000000e+00j, 0.+2.78255940e-02j, 0.+7.74263683e-02j,
        0.+2.15443469e-01j, 0.+5.99484250e-01j, 0.+1.66810054e+00j,
        0.+4.64158883e+00j, 0.+1.29154967e+01j, 0.+3.59381366e+01j,
        0.+1.00000000e+02j]
    )
    exp7 = (
        [ 1.00000000e+00,  9.99366022e-01,  9.95066026e-01,  9.60400295e-01,
         6.58597505e-01, -4.18576362e+02, -8.91843768e+02, -1.07060465e+03,
        -1.19874735e+03, -1.29978855e+03, -1.38291755e+03, -1.45288145e+03,
        -1.51251671e+03, -1.56367938e+03, -1.60766197e+03, -1.64540596e+03,
        -1.67762077e+03, -1.70485488e+03, -1.72754091e+03, -1.74602531e+03,
        -1.76058866e+03, -1.77145991e+03, -1.77882670e+03, -1.78284282e+03,
        -1.78363385e+03, -1.78130128e+03, -1.77592570e+03, -1.76756911e+03,
        -1.75627664e+03, -1.74207774e+03, -1.72498696e+03, -1.70500430e+03,
        -1.68211522e+03, -1.65629026e+03, -1.62748428e+03, -1.59563529e+03,
        -1.56066271e+03, -1.52246503e+03, -1.48091668e+03, -1.43586388e+03,
        -1.38711898e+03, -1.33445302e+03, -1.27758543e+03, -1.21616977e+03,
        -1.14977340e+03, -1.07784785e+03, -9.99683712e+02, -9.14339478e+02,
        -8.20522664e+02, -7.16377214e+02, -5.99066656e+02, -4.63843896e+02,
        -3.01519639e+02, -8.85772045e+01,  4.18871757e+02,  1.53189426e-01,
         2.41254893e-03,  4.04066916e-05,  6.74368970e-07,  1.12498906e-08],
        [ 0.00000000e+00,  3.06081384e-02,  8.51665142e-02,  2.36587984e-01,
         6.15191065e-01,  1.73435960e+03,  1.54517417e+03,  1.42710407e+03,
         1.32125690e+03,  1.22195400e+03,  1.12697641e+03,  1.03517784e+03,
         9.45873355e+02,  8.58614768e+02,  7.73089980e+02,  6.89071641e+02,
         6.06388452e+02,  5.24907966e+02,  4.44525717e+02,  3.65158049e+02,
         2.86737209e+02,  2.09207910e+02,  1.32524844e+02,  5.66508721e+01,
        -1.84443263e+01, -9.27852613e+01, -1.66391391e+02, -2.39277687e+02,
        -3.11455048e+02, -3.82930585e+02, -4.53707812e+02, -5.23786729e+02,
        -5.93163822e+02, -6.61831980e+02, -7.29780304e+02, -7.96993823e+02,
        -8.63453079e+02, -9.29133561e+02, -9.94004949e+02, -1.05803010e+03,
        -1.12116372e+03, -1.18335050e+03, -1.24452271e+03, -1.30459668e+03,
        -1.36346794e+03, -1.42100406e+03, -1.47703372e+03, -1.53132959e+03,
        -1.58357956e+03, -1.63333544e+03, -1.67991231e+03, -1.72216386e+03,
        -1.75787114e+03, -1.78135550e+03, -1.73372452e+03, -2.20178757e-01,
        -7.40826563e-04, -4.23719886e-06, -2.52520163e-08, -1.51267018e-10],
        [ 0.00000000e+00+0.00000000e+00j,  0.00000000e+00+2.78255940e-02j,
         0.00000000e+00+7.74263683e-02j,  0.00000000e+00+2.15443469e-01j,
         0.00000000e+00+5.99484250e-01j, -4.69427265e-11+1.27961822e+00j,
        -2.82783805e-05+1.27962230e+00j, -3.95728968e-05+1.27962638e+00j,
        -4.79483270e-05+1.27963046e+00j, -5.47608484e-05+1.27963454e+00j,
        -6.05403958e-05+1.27963863e+00j, -6.55607282e-05+1.27964271e+00j,
        -6.99854212e-05+1.27964679e+00j, -7.39215113e-05+1.27965087e+00j,
        -7.74435345e-05+1.27965495e+00j, -8.06057864e-05+1.27965903e+00j,
        -8.34491767e-05+1.27966312e+00j, -8.60053368e-05+1.27966720e+00j,
        -8.82992149e-05+1.27967128e+00j, -9.03507903e-05+1.27967536e+00j,
        -9.21762432e-05+1.27967944e+00j, -9.37887779e-05+1.27968352e+00j,
        -9.51992146e-05+1.27968761e+00j, -9.64164229e-05+1.27969169e+00j,
        -9.74476440e-05+1.27969577e+00j, -9.82987313e-05+1.27969985e+00j,
        -9.89743319e-05+1.27970393e+00j, -9.94780213e-05+1.27970801e+00j,
        -9.98124021e-05+1.27971210e+00j, -9.99791732e-05+1.27971618e+00j,
        -9.99791732e-05+1.27972026e+00j, -9.98124021e-05+1.27972434e+00j,
        -9.94780213e-05+1.27972842e+00j, -9.89743319e-05+1.27973250e+00j,
        -9.82987313e-05+1.27973658e+00j, -9.74476440e-05+1.27974067e+00j,
        -9.64164229e-05+1.27974475e+00j, -9.51992146e-05+1.27974883e+00j,
        -9.37887779e-05+1.27975291e+00j, -9.21762432e-05+1.27975699e+00j,
        -9.03507903e-05+1.27976107e+00j, -8.82992149e-05+1.27976516e+00j,
        -8.60053368e-05+1.27976924e+00j, -8.34491767e-05+1.27977332e+00j,
        -8.06057864e-05+1.27977740e+00j, -7.74435345e-05+1.27978148e+00j,
        -7.39215113e-05+1.27978556e+00j, -6.99854212e-05+1.27978965e+00j,
        -6.55607282e-05+1.27979373e+00j, -6.05403958e-05+1.27979781e+00j,
        -5.47608484e-05+1.27980189e+00j, -4.79483270e-05+1.27980597e+00j,
        -3.95728968e-05+1.27981005e+00j, -2.82783805e-05+1.27981414e+00j,
        -4.69427265e-11+1.27981822e+00j,  0.00000000e+00+1.66810054e+00j,
         0.00000000e+00+4.64158883e+00j,  0.00000000e+00+1.29154967e+01j,
         0.00000000e+00+3.59381366e+01j,  0.00000000e+00+1.00000000e+02j]
    )
    exp8 = (
        [ 5.00000000e-01,  5.01842320e-01,  5.07051300e-01,  5.31188256e-01,
         6.61811024e-01, -1.37231910e-01, -1.04252206e-02, -8.24235914e-04,
        -6.43512152e-05, -4.99503496e-06],
        [-0.00000000e+00, -3.62053471e-03, -1.32914763e-02, -5.29067014e-02,
        -3.97873373e-01, -5.15256460e-02, -9.44305419e-04, -2.10540294e-05,
        -4.60935569e-07, -9.98018952e-09],
        [0.+0.00000000e+00j, 0.+3.59381366e-02j, 0.+1.29154967e-01j,
        0.+4.64158883e-01j, 0.+1.66810054e+00j, 0.+5.99484250e+00j,
        0.+2.15443469e+01j, 0.+7.74263683e+01j, 0.+2.78255940e+02j,
        0.+1.00000000e+03j]
    )

    def nyquist_res_tester(sys, num_points, expected_value):
        p = plot_nyquist(sys, show=False, n=num_points)
        x, y, param = p[0].get_data()
        x_check = np.allclose(x, expected_value[0])
        y_check = np.allclose(y, expected_value[1])
        param_check = np.allclose(param, expected_value[2], )
        return x_check and y_check and param_check

    assert nyquist_res_tester(tf1, 10, exp1)
    assert nyquist_res_tester(tf2, 10, exp2)
    assert nyquist_res_tester(tf3, 10, exp3)
    assert nyquist_res_tester(tf4, 10, exp4)
    assert nyquist_res_tester(tf5, 10, exp5)
    assert nyquist_res_tester(tf6, 10, exp6)
    assert nyquist_res_tester(tf7, 10, exp7)
    assert nyquist_res_tester(tf8, 10, exp8)


def test_plot_nyquist_matplotlib():
    # verify that plot_nyquist adds the necessary objects to the plot

    # standard plot, no m-circles
    p = plot_nyquist(tf1, show=False, n=10)
    ax = p.ax
    assert len(ax.lines) == 6
    assert len(ax.collections) == 0
    assert len(ax.patches) == 4
    assert len(ax.texts) == 0

    # standard plot, no m-circles, no mirror image
    p = plot_nyquist(tf1, show=False, n=10, mirror_style=False)
    ax = p.ax
    assert len(ax.lines) == 4
    assert len(ax.collections) == 0
    assert len(ax.patches) == 2
    assert len(ax.texts) == 0

    # m-circles + custom number of arrows
    p = plot_nyquist(tf1, show=False, n=10, arrows=3, m_circles=True)
    ax = p.ax
    assert len(ax.lines) == 6
    assert len(ax.collections) > 0
    assert len(ax.patches) == 6
    assert len(ax.texts) > 0

    # standard plot but no start marker, no m-circles
    p = plot_nyquist(tf1, show=False, n=10, start_marker=False)
    ax = p.ax
    assert len(ax.lines) == 5
    assert len(ax.collections) == 0
    assert len(ax.patches) == 4
    assert len(ax.texts) == 0


def test_plot_nyquist_matplotlib_linestyles():

    # standard plot, custom line styles. Verify that no errors are raised
    p = plot_nyquist(tf1, show=False, n=10,
        primary_style="-", mirror_style=":")
    ax = p.ax

    p = plot_nyquist(tf1, show=False, n=10,
        primary_style=["-", "-."], mirror_style=["--", ":"])
    ax = p.ax

    p = plot_nyquist(tf1, show=False, n=10,
        primary_style={"linestyle": "-"},
        mirror_style={"linestyle": ":"})
    ax = p.ax

    p = plot_nyquist(tf1, show=False, n=10,
        primary_style=[{"linestyle": "-"}, {"linestyle": ":"}],
        mirror_style=[{"linestyle": "--"}, {"linestyle": "-."}])
    ax = p.ax

    # unrecognized line styles
    p = plot_nyquist(tf1, show=False, n=10,
        primary_style=2,
        mirror_style=2)
    raises(ValueError, lambda: p.ax)


def test_plot_nyquist_matplotlib_interactive():
    # verify that interactive update doesn't raise errors

    tf = TransferFunction(1, s + a, s)
    pl = plot_nyquist(
        tf, xlim=(-2, 1), ylim=(-1, 1),
        aspect="equal", m_circles=True,
        params={a: (1, 0, 2)},
        arrows=4, n=10, show=False
    )
    ax = pl.backend.ax # force first draw
    pl.backend.update_interactive({a: 2}) # update with new value


def test_plot_nyquist_omega_limits():
    # verify that `omega_limits` works as expected

    p1 = plot_nyquist(tf1, show=False)
    p2 = plot_nyquist(tf1, omega_limits=(1e-05, 1e-02), show=False)
    assert p2[0].ranges[0][1] == 1e-05
    assert p2[0].ranges[0][2] == 1e-02
    assert p1[0].ranges[0][1:] != p2[0].ranges[0][1:]


def test_plot_nichols():
    # verify that NicholsLineSeries produces correct results

    def nichols_res_tester(sys, omega_limits, num_points, expected_value):
        p = plot_nichols(sys, omega_limits=omega_limits,
            show=False, n=num_points)
        x, y, param = p[0].get_data()
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        param_check = check_point_accuracy(param, expected_value[2])
        return x_check and y_check and param_check

    tf1 = TransferFunction(5 * (s - 1), s**2 * (s**2 + s + 4), s)
    exp1 = (
        [  -7.14627703,  -11.87501843,  -19.6071407 ,  -31.89094174,
         -50.57631953,  -81.231069  , -171.68469677, -232.49622166,
        -249.89886567, -258.34254381],
        [ 42.0004288 ,  33.22148705,  24.63227702,  16.53494133,
          9.61324014,   5.28081374,   1.12439465, -16.46462887,
        -31.69352926, -45.66968097],
        [ 0.1       ,  0.16681005,  0.27825594,  0.46415888,  0.77426368,
         1.29154967,  2.15443469,  3.59381366,  5.9948425 , 10.        ]
    )
    assert nichols_res_tester(tf1, [1e-01, 1e01], 10, exp1)

    tf2 = TransferFunction(-4*s**4 + 48*s**3 - 18*s**2 + 250*s + 600, s**4 + 30*s**3 + 282*s**2 + 525*s + 60, s)
    exp2 = (
        [  -4.7642299 ,  -10.74556702,  -23.41579813,  -44.99062736,
         -67.91516888,  -85.00078123, -107.10140914, -227.43935552,
        -331.72108713, -412.10280752, -477.24930481, -511.78850082,
        -527.54912472, -534.52446663, -537.59367177],
        [ 19.97103009,  19.85200215,  19.28289443,  17.14539872,
         12.36002147,   5.86393272,  -1.78976124, -10.51502076,
          2.44764247,   9.06567945,  11.41302756,  11.91973828,
         12.01777399,  12.0366782 ,  12.04032688],
        [1.00000000e-02, 2.27584593e-02, 5.17947468e-02, 1.17876863e-01,
        2.68269580e-01, 6.10540230e-01, 1.38949549e+00, 3.16227766e+00,
        7.19685673e+00, 1.63789371e+01, 3.72759372e+01, 8.48342898e+01,
        1.93069773e+02, 4.39397056e+02, 1.00000000e+03]
    )
    assert nichols_res_tester(tf2, [1e-02, 1e03], 15, exp2)


def test_plot_nichols_matplotlib():
    tf = TransferFunction(5 * (s - 1), s**2 * (s**2 + s + 4), s)

    # with nichols grid lines
    p = plot_nichols(tf, ngrid=True, show=False, n=10)
    ax = p.ax
    assert len(ax.lines) > 2
    assert len(ax.texts) > 0

    # no nichols grid lines
    p = plot_nichols(tf, ngrid=False, show=False, n=10)
    ax = p.ax
    assert len(ax.lines) == 1
    assert len(ax.texts) == 0


@pytest.mark.parametrize(
    "func, params",
    [
        (plot_pole_zero, test_params),
        (plot_impulse_response, test_params),
        (plot_step_response, test_params),
        (plot_ramp_response, test_params),
        (plot_bode_magnitude, test_params),
        (plot_bode_phase, test_params),
        (plot_nyquist, test_params),
        (plot_nichols, test_params),
    ]
)
def test_new_ways_of_providing_transfer_function(func, params):
    kwargs = {"show": False, "n": 10}

    p1 = func(tf_test1, **kwargs)
    p2 = func((num1, den1), **kwargs)
    p3 = func((num1, den1, p), **kwargs)
    d1, d2, d3 = [t[0].get_data() for t in [p1, p2, p3]]
    assert np.allclose(d1, d2) and np.allclose(d2, d3)

    kwargs["params"] = params
    p4 = func(tf_test2, **kwargs)
    p5 = func((num2, den2, p), **kwargs)
    d4, d5 = [t.backend[0].get_data() for t in [p4, p5]]
    assert np.allclose(d4, d5)


def test_plot_bode_title():
    G1 = (s+5)/(s+2)**2
    G2 = 1/s**2

    p = plot_bode(G1, show=False)
    assert p.args[0].title == f"Bode Plot of ${latex(G1)}$"
    assert p.args[1].title == ""

    p = plot_bode(G1, show=False, title="Test")
    assert p.args[0].title == "Test"
    assert p.args[1].title == ""

    p = plot_bode(G1, G2, show=False)
    assert p.args[0].title == "Bode Plot"
    assert p.args[1].title == ""

    p = plot_bode(G1, G2, show=False, title="Test")
    assert p.args[0].title == "Test"
    assert p.args[1].title == ""
