import numpy as np
import pytest
from pytest import warns
from spb import (
    control_axis, pole_zero, step_response, impulse_response, ramp_response,
    bode_magnitude, bode_phase, nyquist, nichols, sgrid, root_locus, zgrid
)
from spb.series import (
    LineOver1DRangeSeries, HVLineSeries, List2DSeries, NyquistLineSeries,
    NicholsLineSeries, SGridLineSeries, RootLocusSeries, ZGridLineSeries,
    SystemResponseSeries
)
from sympy.abc import a, b, c, d, e, s
from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
import control as ct
import scipy.signal as signal


n1, d1 = s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2
n2, d2 = s**2 + e, s**4 + a*s**3 + b*s**2 + c*s + d
n3, d3 = 4 * s**2 + 5 * s + 1, 3 * s**2 + 2 * s + 5
n4, d4 = s + 1, (s + a) * (s + b)
tf1 = TransferFunction(n1, d1, s)
tf2 = TransferFunction(n2, d2, s)
tf3 = TransferFunction(n3, d3, s)
tf4 = TransferFunction(n4, d4, s)
# expected expressions
ee1 = tf1.to_expr()
ee2 = tf2.to_expr()
mod_params = {
    a: (1, 0, 8),
    b: (6, 0, 8),
}

tf_1 = TransferFunction(1, s + 2, s)
tf_2 = TransferFunction(s + 1, s**2 + s + 1, s)
tf_3 = TransferFunction(s + 1, s**2 + s + 1.5, s)
tf_mimo_sympy = TransferFunctionMatrix([
    [tf_1, -tf_1],
    [tf_2, -tf_2],
    [tf_3, -tf_3]
])
tf_mimo_control = ct.TransferFunction(
    [[[1], [-1]], [[1, 1], [-1, -1]], [[1, 1], [-1, -1]]],
    [[[1, 2], [1, 2]], [[1, 1, 1], [1, 1, 1]], [[1, 1, 1.5], [1, 1, 1.5]]]
)
tf_siso_sympy = TransferFunction(s + 1, s**2 + s + 1, s)
tf_siso_control = ct.tf([1, 1], [1, 1, 1])
tf_siso_scipy = signal.TransferFunction([1, 1], [1, 1, 1])


@pytest.mark.parametrize(
    "hor, ver, rkw", [
        (True, True, None),
        (True, False, {"color": "r"}),
        (False, True, None),
        (False, False, {"color": "r"}),
])
def test_control_axis(hor, ver, rkw):
    series = control_axis(hor, ver, rendering_kw=rkw)
    assert len(series) == hor + ver
    assert all(s.rendering_kw == {} if not rkw else rkw for s in series)
    assert all(isinstance(s, HVLineSeries) for s in series)
    if len(series) == 2:
        assert series[0].is_horizontal is True
        assert series[1].is_horizontal is False


@pytest.mark.parametrize(
    "tf, label, pkw, zkw, params",
    [
        (tf1, None, None, None, None),
        ((n1, d1), None, None, None, None),
        (n1 / d1, None, None, None, None),
        (tf1, "test", {"color": "r"}, {"color": "k"}, None),
        ((n1, d1), "test", {"color": "r"}, {"color": "k"}, None),
        (n1 / d1, "test", {"color": "r"}, {"color": "k"}, None),
        (tf1, "test", {"color": "r"}, {"color": "k"}, mod_params),
        ((n1, d1), "test", {"color": "r"}, {"color": "k"}, mod_params),
        (n1 / d1, "test", {"color": "r"}, {"color": "k"}, mod_params),
    ]
)
def test_pole_zero(tf, label, pkw, zkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params
    if pkw:
        kwargs["p_rendering_kw"] = pkw
    if zkw:
        kwargs["z_rendering_kw"] = zkw

    series = pole_zero(tf, label=label, **kwargs)
    assert len(series) == 2
    assert all(isinstance(s, List2DSeries) for s in series)
    d1 = series[0].get_data()
    assert np.allclose(
        d1,
        [
            np.array([-2. , -0.5, -1. , -0.5]),
            np.array([
                4.44089210e-16,  8.66025404e-01,
                -6.76185784e-16, -8.66025404e-01
            ])
        ]
    )
    d2 = series[1].get_data()
    assert np.allclose(
        d2,
        [
            np.array([0.00000000e+00, 2.77555756e-17]),
            np.array([ 1., -1.])
        ]
    )
    if pkw:
        assert series[0].rendering_kw.get("color", None) == pkw["color"]
    if zkw:
        assert series[1].rendering_kw.get("color", None) == zkw["color"]
    assert all(s.is_interactive == (len(s.params) > 0) for s in series)
    assert all(s.params == {} if not params else params for s in series)


@pytest.mark.parametrize(
    "tf, label, rkw, params",
    [
        (tf1, None, None, None),
        ((n1, d1), None, None, None),
        (n1 / d1, None, None, None),
        (tf1, "test", {"color": "r"}, None),
        ((n1, d1), "test", {"color": "r"}, None),
        (n1 / d1, "test", {"color": "r"}, None),
        (tf4, "test", {"color": "r"}, mod_params),
        ((n4, d4, s), "test", {"color": "r"}, mod_params),
        (n4 / d4, "test", {"color": "r"}, mod_params),
    ]
)
def test_step_response(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    # evaluate with sympy
    series = step_response(tf, label=label, rendering_kw=rkw,
        control=False, prec=16, **kwargs.copy())
    assert len(series) == 1
    s1 = series[0]
    assert isinstance(s1, LineOver1DRangeSeries)
    d1 = s1.get_data()
    assert s1.rendering_kw == {} if not rkw else rkw
    assert s1.is_interactive == (len(s1.params) > 0)
    assert s1.params == {} if not params else params

    # evaluate with control
    series = step_response(tf, label=label, rendering_kw=rkw,
        control=True, prec=16, **kwargs.copy())
    assert len(series) == 1
    s2 = series[0]
    assert isinstance(s2, SystemResponseSeries)
    d2 = s2.get_data()
    assert s2.rendering_kw == {} if not rkw else rkw
    assert s2.is_interactive == (len(s2.params) > 0)
    assert s2.params == {} if not params else params

    assert np.allclose(d1, d2)


@pytest.mark.parametrize(
    "func, lower_limit, params",
    [
        (step_response, 1, {}),
        (step_response, a, {a: 2}),
        (impulse_response, 1, {}),
        (impulse_response, a, {a: 2}),
        (ramp_response, 1, {}),
        (ramp_response, a, {a: 2}),
    ]
)
def test_lower_limit_user_warning(func, lower_limit, params):
    # verify that a UserWarning is emitted when ``control=True`` and
    # ``lower_limit != 0`` is detected.

    with warns(
        UserWarning,
        match="You are evaluating a transfer function using the ``control``",
    ):
        func(tf1, lower_limit=lower_limit, params=params)


@pytest.mark.parametrize(
    "tf, label, rkw, params",
    [
        (tf1, None, None, None),
        ((n1, d1), None, None, None),
        (n1 / d1, None, None, None),
        (tf1, "test", {"color": "r"}, None),
        ((n1, d1), "test", {"color": "r"}, None),
        (n1 / d1, "test", {"color": "r"}, None),
        (tf4, "test", {"color": "r"}, mod_params),
        ((n4, d4, s), "test", {"color": "r"}, mod_params),
        (n4 / d4, "test", {"color": "r"}, mod_params),
    ]
)
def test_impulse_response(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    # evaluate with sympy
    series = impulse_response(tf, label=label, rendering_kw=rkw,
        control=False, prec=16, **kwargs.copy())
    assert len(series) == 1
    s1 = series[0]
    assert isinstance(s1, LineOver1DRangeSeries)
    d1 = s1.get_data()
    assert s1.rendering_kw == {} if not rkw else rkw
    assert s1.is_interactive == (len(s1.params) > 0)
    assert s1.params == {} if not params else params

    # evaluate with control
    series = impulse_response(tf, label=label, rendering_kw=rkw,
        control=True, prec=16, **kwargs)
    assert len(series) == 1
    s2 = series[0]
    assert isinstance(s2, SystemResponseSeries)
    d2 = s2.get_data()
    assert s2.rendering_kw == {} if not rkw else rkw
    assert s2.is_interactive == (len(s2.params) > 0)
    assert s2.params == {} if not params else params

    assert np.allclose(d1, d2)


@pytest.mark.parametrize(
    "tf, label, rkw, params",
    [
        (tf1, None, None, None),
        ((n1, d1), None, None, None),
        (n1 / d1, None, None, None),
        (tf1, "test", {"color": "r"}, None),
        ((n1, d1), "test", {"color": "r"}, None),
        (n1 / d1, "test", {"color": "r"}, None),
        (tf4, "test", {"color": "r"}, mod_params),
        ((n4, d4, s), "test", {"color": "r"}, mod_params),
        (n4 / d4, "test", {"color": "r"}, mod_params),
    ]
)
def test_ramp_response(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    # evaluate with sympy
    series = ramp_response(tf, label=label, rendering_kw=rkw,
        control=False, prec=16, **kwargs)
    assert len(series) == 1
    s1 = series[0]
    assert isinstance(s1, LineOver1DRangeSeries)
    d1 = s1.get_data()
    assert s1.rendering_kw == {} if not rkw else rkw
    assert s1.is_interactive == (len(s1.params) > 0)
    assert s1.params == {} if not params else params

    # evaluate with control
    series = ramp_response(tf, label=label, rendering_kw=rkw,
        control=True, prec=16, **kwargs)
    assert len(series) == 1
    s2 = series[0]
    assert isinstance(s2, SystemResponseSeries)
    d2 = s2.get_data()
    assert s2.rendering_kw == {} if not rkw else rkw
    assert s2.is_interactive == (len(s2.params) > 0)
    assert s2.params == {} if not params else params

    assert np.allclose(d1, d2)


@pytest.mark.parametrize(
    "tf, label, rkw, params",
    [
        (tf1, None, None, None),
        ((n1, d1), None, None, None),
        (n1 / d1, None, None, None),
        (tf1, "test", {"color": "r"}, None),
        ((n1, d1), "test", {"color": "r"}, None),
        (n1 / d1, "test", {"color": "r"}, None),
        (tf4, "test", {"color": "r"}, mod_params),
        ((n4, d4, s), "test", {"color": "r"}, mod_params),
        (n4 / d4, "test", {"color": "r"}, mod_params),
    ]
)
def test_bode_magnitude(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    series = bode_magnitude(tf, label=label, rendering_kw=rkw, **kwargs.copy())
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, LineOver1DRangeSeries)
    s.get_data()
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize(
    "tf, label, rkw, params",
    [
        (tf1, None, None, None),
        ((n1, d1), None, None, None),
        (n1 / d1, None, None, None),
        (tf1, "test", {"color": "r"}, None),
        ((n1, d1), "test", {"color": "r"}, None),
        (n1 / d1, "test", {"color": "r"}, None),
        (tf4, "test", {"color": "r"}, mod_params),
        ((n4, d4, s), "test", {"color": "r"}, mod_params),
        (n4 / d4, "test", {"color": "r"}, mod_params),
    ]
)
def test_bode_phase(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    series = bode_phase(tf, label=label, rendering_kw=rkw, **kwargs)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, LineOver1DRangeSeries)
    s.get_data()
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize(
    "tf, label, rkw, params",
    [
        (tf3, None, None, None),
        ((n3, d3), None, None, None),
        (n3 / d3, None, None, None),
        (tf3, "test", None, None),
        ((n3, d3), "test", None, None),
        (n3 / d3, "test", None, None),
        # TODO: these fails...
        # (tf3, "test", None, mod_params),
        # ((n3, d3), "test", None, mod_params),
        # (n3 / d3, "test", None, mod_params),
    ]
)
def test_nyquist(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    print(type(tf), tf)

    series = nyquist(tf, label=label, rendering_kw=rkw, **kwargs)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, NyquistLineSeries)
    s.get_data()
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize(
    "tf, label, rkw, params",
    [
        (tf1, None, None, None),
        ((n1, d1), None, None, None),
        (n1 / d1, None, None, None),
        (tf1, "test", {"color": "r"}, None),
        ((n1, d1), "test", {"color": "r"}, None),
        (n1 / d1, "test", {"color": "r"}, None),
        (tf4, "test", {"color": "r"}, mod_params),
        ((n4, d4, s), "test", {"color": "r"}, mod_params),
        (n4 / d4, "test", {"color": "r"}, mod_params),
    ]
)
def test_nichols(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    series = nichols(tf, label=label, rendering_kw=rkw, **kwargs)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, NicholsLineSeries)
    s.get_data()
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


def test_sgrid():
    series = sgrid()
    assert len(series) == 1
    assert isinstance(series[0], SGridLineSeries)
    assert series[0].show_control_axis
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    xi_ret = [k[0] for k in xi_dict.keys()]
    wn_ret = list(wn_dict.keys())
    assert np.allclose(xi_ret, [.1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99])
    assert np.allclose(wn_ret, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert (len(tp_dict) == 0) and (len(ts_dict) == 0)

    series = sgrid(show_control_axis=False)
    assert len(series) == 1
    assert not series[0].show_control_axis
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    xi_ret = [k[0] for k in xi_dict.keys()]
    wn_ret = list(wn_dict.keys())
    assert np.allclose(xi_ret, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99, 1])
    assert np.allclose(wn_ret, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    series = sgrid(show_control_axis=True, rendering_kw={"color": "r"})
    assert len(series) == 1
    assert series[0].show_control_axis
    assert series[0].rendering_kw == {"color": "r"}

    series = sgrid(xi=False, show_control_axis=False)
    assert len(series) == 1
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    wn_ret = list(wn_dict.keys())
    assert len(xi_dict) == 0
    assert np.allclose(wn_ret, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    series = sgrid(wn=False, show_control_axis=False)
    assert len(series) == 1
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    xi_ret = [k[0] for k in xi_dict.keys()]
    assert np.allclose(xi_ret, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99, 1])
    assert len(wn_dict) == 0

    # verify that by setting axis limits, the series computes appropriate
    # values to distribute the grid evenly over the specified space.
    series = sgrid(xlim=(-11, 1), ylim=(-10, 10), show_control_axis=False)
    assert len(series) == 1
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    xi_ret = [k[0] for k in xi_dict.keys()]
    wn_ret = list(wn_dict.keys())
    assert np.allclose(xi_ret, [
        0.26515648302104233, 0.48191874977215593, 0.6363829547955636,
        0.855197831554018, 0.9570244044334736, 0, 1])
    assert np.allclose(wn_ret, [1.83333333, 3.66666667, 5.5, 7.33333333, 9.16666667])

    series = sgrid(xlim=(-11, 1), ylim=(-10, 10), show_control_axis=True)
    assert len(series) == 1
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    xi_ret = [k[0] for k in xi_dict.keys()]
    wn_ret = list(wn_dict.keys())
    assert np.allclose(xi_ret, [
        0.26515648302104233, 0.48191874977215593, 0.6363829547955636,
        0.855197831554018, 0.9570244044334736])
    assert np.allclose(wn_ret, [1.83333333, 3.66666667, 5.5, 7.33333333, 9.16666667])


def test_zgrid():
    series = zgrid()
    assert len(series) == 1
    assert isinstance(series[0], ZGridLineSeries)
    assert series[0].show_control_axis
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    xi_ret = list(xi_dict.keys())
    wn_ret = list(wn_dict.keys())
    assert np.allclose(xi_ret, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    assert np.allclose(wn_ret, [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    assert (len(tp_dict) == 0) and (len(ts_dict) == 0)

    series = sgrid(show_control_axis=False, rendering_kw={"color": "r"})
    assert len(series) == 1
    assert not series[0].show_control_axis
    assert series[0].rendering_kw == {"color": "r"}


def test_root_locus():
    G1 = (s**2 + 1) / (s**3 + 2*s**2 + 3*s + 4)

    series = root_locus(G1)
    assert len(series) == 2
    assert isinstance(series[0], SGridLineSeries)
    assert isinstance(series[1], RootLocusSeries)
    assert series[0].get_label() == ""
    assert series[0].rendering_kw == {}

    series = root_locus(G1, label="a", rendering_kw={"color": "k"})
    assert len(series) == 2
    assert isinstance(series[0], SGridLineSeries)
    assert isinstance(series[1], RootLocusSeries)
    assert series[1].get_label() == "a"
    assert series[1].rendering_kw == {"color": "k"}

    series = root_locus(G1, sgrid=True)
    assert len(series) == 2
    assert isinstance(series[0], SGridLineSeries)
    assert isinstance(series[1], RootLocusSeries)

    series1 = root_locus(G1, sgrid=False)
    assert len(series1) == 1
    assert isinstance(series1[0], RootLocusSeries)
    data1 = series1[0].get_data()

    series2 = root_locus(
        G1, rl_kw={
            "kvect": np.linspace(1e-03, 1000, 10)
        }, sgrid=False)
    assert len(series2) == 1
    data2 = series2[0].get_data()
    assert data1[0].shape != data2[0].shape

    series = root_locus(G1, sgrid=False)
    assert len(series) == 1
    assert isinstance(series[0], RootLocusSeries)

    series = root_locus(G1, sgrid=False, zgrid=True)
    assert len(series) == 2
    assert isinstance(series[0], ZGridLineSeries)
    assert isinstance(series[1], RootLocusSeries)


@pytest.mark.parametrize(
    "tf_siso, func",
    [
        (tf_siso_sympy, step_response),
        (tf_siso_sympy, impulse_response),
        (tf_siso_sympy, ramp_response),
        (tf_siso_control, step_response),
        (tf_siso_control, impulse_response),
        (tf_siso_control, ramp_response),
        (tf_siso_scipy, step_response),
        (tf_siso_scipy, impulse_response),
        (tf_siso_scipy, ramp_response)
    ]
)
def test_siso_responses(tf_siso, func):
    # verify that the module is able to deal with SISO systems from the
    # sympy, control and scipy, both when inverse laplace transform is to
    # be used, as well as when numerical integration is to be used.

    series = func(tf_siso, prec=16, n=10, control=True)
    assert len(series) == 1
    s1 = series[0]
    d1 = s1.get_data()

    series = func(tf_siso, prec=16, n=10, control=False)
    assert len(series) == 1
    s2 = series[0]
    d2 = s2.get_data()

    assert np.allclose(d1, d2)


@pytest.mark.parametrize(
    "tf_mimo, func, use_control",
    [
        (tf_mimo_sympy, step_response, True),
        (tf_mimo_sympy, impulse_response, True),
        (tf_mimo_sympy, ramp_response, True),
        (tf_mimo_sympy, step_response, False),
        (tf_mimo_sympy, impulse_response, False),
        (tf_mimo_sympy, ramp_response, False),
        (tf_mimo_control, step_response, True),
        (tf_mimo_control, impulse_response, True),
        (tf_mimo_control, ramp_response, True),
        (tf_mimo_control, step_response, False),
        (tf_mimo_control, impulse_response, False),
        (tf_mimo_control, ramp_response, False),
    ]
)
def test_mimo_responses(tf_mimo, func, use_control):
    # verify that a MIMO system gets unpacked into several SISO systems

    series = func(tf_mimo, prec=16, n=10, control=use_control)
    assert len(series) == 6
    test_series = SystemResponseSeries if use_control else LineOver1DRangeSeries
    assert all(isinstance(s, test_series) for s in series)

    s1 = func(tf_1, prec=16, n=10, control=use_control)[0]
    s2 = func(tf_2, prec=16, n=10, control=use_control)[0]
    s3 = func(tf_3, prec=16, n=10, control=use_control)[0]
    s4 = func(-tf_1, prec=16, n=10, control=use_control)[0]
    s5 = func(-tf_2, prec=16, n=10, control=use_control)[0]
    s6 = func(-tf_3, prec=16, n=10, control=use_control)[0]
    assert np.allclose(s1.get_data(), series[0].get_data())
    assert np.allclose(s2.get_data(), series[1].get_data())
    assert np.allclose(s3.get_data(), series[2].get_data())
    assert np.allclose(s4.get_data(), series[3].get_data())
    assert np.allclose(s5.get_data(), series[4].get_data())
    assert np.allclose(s6.get_data(), series[5].get_data())


@pytest.mark.parametrize(
    "tf_mimo, func, inp, out",
    [
        (tf_mimo_sympy, step_response, 1, 2),
        (tf_mimo_sympy, impulse_response, 1, 2),
        (tf_mimo_sympy, ramp_response, 1, 2),
        (tf_mimo_control, step_response, 1, 2),
        (tf_mimo_control, impulse_response, 1, 2),
        (tf_mimo_control, ramp_response, 1, 2),
    ]
)
def test_mimo_responses_2(tf_mimo, func, inp, out):
    # verify that input-output keyword arguments work as expected

    series = func(tf_mimo, prec=16, n=10, control=True, input=inp, output=out)
    assert len(series) == 1
    d1 = series[0].get_data()

    s6 = func(-tf_3, prec=16, n=10, control=True)[0]
    d6 = s6.get_data()

    assert np.allclose(d1, d6)
