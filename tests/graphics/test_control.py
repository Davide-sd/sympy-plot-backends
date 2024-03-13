import numpy as np
import pytest
from pytest import warns, raises
from spb import (
    control_axis, pole_zero, step_response, impulse_response, ramp_response,
    bode_magnitude, bode_phase, nyquist, nichols, sgrid, root_locus, zgrid,
    mcircles
)
from spb.series import (
    LineOver1DRangeSeries, HVLineSeries, List2DSeries, NyquistLineSeries,
    NicholsLineSeries, SGridLineSeries, RootLocusSeries, ZGridLineSeries,
    SystemResponseSeries, PoleZeroSeries, NGridLineSeries, MCirclesSeries
)
from sympy.abc import a, b, c, d, e, s
from sympy import exp
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

tf_dt_control = ct.tf([1], [1, 2, 3], dt=0.05)
tf_dt_scipy = signal.TransferFunction([1], [1, 2, 3], dt=0.05)

tf_cont_control_2 = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236])
tf_cont_scipy_2 = signal.TransferFunction(
    [0.0244, 0.0236], [1.1052, -2.0807, 1.0236])
tf_dt_control_2 = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)
tf_dt_scipy_2 = signal.TransferFunction(
    [0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)


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
    "tf, label, pkw, zkw, params, use_control",
    [
        (tf1, None, None, None, None, False),
        ((n1, d1), None, None, None, None, False),
        (n1 / d1, None, None, None, None, False),
        (tf1, "test", {"color": "r"}, {"color": "k"}, None, False),
        ((n1, d1), "test", {"color": "r"}, {"color": "k"}, None, False),
        (n1 / d1, "test", {"color": "r"}, {"color": "k"}, None, False),
        (tf1, "test", {"color": "r"}, {"color": "k"}, mod_params, False),
        ((n1, d1), "test", {"color": "r"}, {"color": "k"}, mod_params, False),
        (n1 / d1, "test", {"color": "r"}, {"color": "k"}, mod_params, False),
        (tf1, None, None, None, None, True),
        ((n1, d1), None, None, None, None, True),
        (n1 / d1, None, None, None, None, True),
        (tf1, "test", {"color": "r"}, {"color": "k"}, None, True),
        ((n1, d1), "test", {"color": "r"}, {"color": "k"}, None, True),
        (n1 / d1, "test", {"color": "r"}, {"color": "k"}, None, True),
        (tf1, "test", {"color": "r"}, {"color": "k"}, mod_params, True),
        ((n1, d1), "test", {"color": "r"}, {"color": "k"}, mod_params, True),
        (n1 / d1, "test", {"color": "r"}, {"color": "k"}, mod_params, True),
    ]
)
def test_pole_zero(tf, label, pkw, zkw, params, use_control):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params
    if pkw:
        kwargs["p_rendering_kw"] = pkw
    if zkw:
        kwargs["z_rendering_kw"] = zkw

    series = pole_zero(tf, label=label, control=use_control, **kwargs)
    assert len(series) == 2
    test_series = List2DSeries if not use_control else PoleZeroSeries
    assert all(isinstance(s, test_series) for s in series)
    assert "poles" in series[0].get_label(True)
    assert "zeros" in series[1].get_label(True)
    d1 = series[0].get_data()
    assert len(d1) == 2
    # use np.sort because sympy and numpy results of roots are
    # sorted differently
    assert np.allclose(np.sort(d1[0]), np.array([-2., -1., -0.5, -0.5]))
    assert np.allclose(np.sort(d1[1]), np.array([
        -8.66025404e-01, -6.76185784e-16, 4.44089210e-16, 8.66025404e-01
    ]))
    d2 = series[1].get_data()
    assert len(d1) == 2
    assert np.allclose(np.sort(d2[0]), np.array([0, 0]))
    assert np.allclose(np.sort(d2[1]), np.array([
        -1, 1
    ]))
    if pkw:
        assert series[0].rendering_kw.get("color", None) == pkw["color"]
    if zkw:
        assert series[1].rendering_kw.get("color", None) == zkw["color"]
    assert all(s.is_interactive == (len(s.params) > 0) for s in series)
    assert all(s.params == {} if not params else params for s in series)


@pytest.mark.parametrize(
    "tf, use_control",
    [
        (tf_mimo_sympy, True),
        (tf_mimo_sympy, False),
        (tf_mimo_control, True),
        (tf_mimo_control, False),
    ]
)
def test_pole_zero_mimo_1(tf, use_control):
    series = pole_zero(tf, control=use_control)
    assert len(series) == 12
    test_series = List2DSeries if not use_control else PoleZeroSeries
    assert all(isinstance(s, test_series) for s in series)


@pytest.mark.parametrize(
    "use_control", [True, False]
)
def test_pole_zero_grids(use_control):
    # verify that grid line series works with pole_zero

    tf = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    test_series = List2DSeries if not use_control else PoleZeroSeries

    series = pole_zero(tf, sgrid=False, zgrid=False, control=use_control)
    assert len(series) == 2
    assert all(isinstance(t, test_series) for t in series)

    series = pole_zero(tf, sgrid=True, zgrid=False, control=use_control)
    assert len(series) == 3
    assert isinstance(series[0], SGridLineSeries)
    assert all(isinstance(t, test_series) for t in series[1:])

    series = pole_zero(tf, sgrid=False, zgrid=True, control=use_control)
    assert len(series) == 3
    assert isinstance(series[0], ZGridLineSeries)
    assert all(isinstance(t, test_series) for t in series[1:])


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


def test_ramp_response_symbolic_slope_non_symbolic_tf():
    # with a symbolic transfer function and a symbolic slope,
    # everything works fine.
    G1 = TransferFunction(s, (s+4)*(s+8), s)
    series = ramp_response(G1, slope=a, params={a: 1})
    assert len(series) == 1
    assert isinstance(series[0], SystemResponseSeries)

    G2 = ct.tf([1, 0], [1, 12, 32])
    raises(ValueError, lambda: ramp_response(G2, slope=a, params={a: 1}))


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
    "tf", [tf_dt_control, tf_dt_scipy]
)
def test_bode_magnitude_discrete_time(tf):
    series = bode_magnitude(tf, n=20)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, LineOver1DRangeSeries)
    omega, mag = s.get_data()
    assert np.allclose(
        omega,
        [
            0.1, 0.14036945853331978, 0.1970358488893738, 0.27657815420254417,
            0.3882312574755614, 0.5449581139755442, 0.7649547538208629,
            1.0737628459632347, 1.507235092810557, 2.115697738602358,
            2.969793459877822, 4.168682999188442, 5.8515577539313695,
            8.213799934957954, 11.529666493710657, 16.18413042791924,
            22.717576249996483, 31.88853877401411, 44.76176921127132,
            62.83185307179586
        ]
    )
    assert np.allclose(
        mag,
        [
            -15.562964688927812, -15.562906157993359, -15.562790830700457,
            -15.562563593155973, -15.56211584816428, -15.561233608406162,
            -15.559495195899126, -15.55606957427346, -15.54931860667977,
            -15.536011868372333, -15.509773745388479, -15.458001107539891,
            -15.35570220427156, -15.153014496678725, -14.74927194903147,
            -13.936730078489857, -12.271576540423327, -8.824474065020741,
            -4.304122587254962, -6.020599913279624
        ]
    )


@pytest.mark.parametrize(
    "tf, input, output, n_series, func",
    [
        (tf_mimo_sympy, None, None, 6, bode_magnitude),
        (tf_mimo_sympy, 1, None, 3, bode_magnitude),
        (tf_mimo_sympy, None, 1, 2, bode_magnitude),
        (tf_mimo_sympy, 1, 1, 1, bode_magnitude),
        (tf_mimo_control, None, None, 6, bode_magnitude),
        (tf_mimo_control, 1, None, 3, bode_magnitude),
        (tf_mimo_control, None, 1, 2, bode_magnitude),
        (tf_mimo_control, 1, 1, 1, bode_magnitude),
        (tf_mimo_sympy, None, None, 6, bode_phase),
        (tf_mimo_sympy, 1, None, 3, bode_phase),
        (tf_mimo_sympy, None, 1, 2, bode_phase),
        (tf_mimo_sympy, 1, 1, 1, bode_phase),
        (tf_mimo_control, None, None, 6, bode_phase),
        (tf_mimo_control, 1, None, 3, bode_phase),
        (tf_mimo_control, None, 1, 2, bode_phase),
        (tf_mimo_control, 1, 1, 1, bode_phase),
    ]
)
def test_bode_magnitude_phase_mimo(tf, input, output, n_series, func):
    series = func(tf, input=input, output=output)
    assert len(series) == n_series
    assert all(isinstance(t, LineOver1DRangeSeries) for t in series)


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
    "tf", [tf_dt_control, tf_dt_scipy]
)
def test_bode_phase_discrete_time(tf):
    series = bode_phase(tf, n=20)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, LineOver1DRangeSeries)
    omega, phase = s.get_data()
    assert np.allclose(
        omega,
        [
            0.1, 0.14036945853331978, 0.1970358488893738, 0.27657815420254417,
            0.3882312574755614, 0.5449581139755442, 0.7649547538208629,
            1.0737628459632347, 1.507235092810557, 2.115697738602358,
            2.969793459877822, 4.168682999188442, 5.8515577539313695,
            8.213799934957954, 11.529666493710657, 16.18413042791924,
            22.717576249996483, 31.88853877401411, 44.76176921127132,
            62.83185307179586
        ]
    )
    assert np.allclose(
        phase,
        [
            -0.003333327932077868, -0.004678967012361904, -0.006567820312024576,
            -0.009219157529571666, -0.01294072584105985, -0.01816439622255212,
            -0.02549607355839771, -0.03578540508403212, -0.050222659270990996,
            -0.07047201825044286, -0.09885115861230633, -0.1385621707492224,
            -0.19395520911160205, -0.27072012891745806, -0.3755968033794409,
            -0.5140165555432296, -0.6785159725865778, -0.7849727242513563,
            -0.37341667572023984, -0.0
        ]
    )


def test_bode_time_delay():
    G1 = 1 / (s * (s + 1) * (s + 10))
    G2 = G1 * exp(-5*s)
    s1 = bode_magnitude(G1, n=10)[0]
    s2 = bode_magnitude(G2, n=10)[0]
    s3 = bode_phase(G1, n=10)[0]
    s4 = bode_phase(G2, n=10)[0]
    d1, d2, d3, d4 = [t.get_data() for t in [s1, s2, s3, s4]]
    assert np.allclose(d1, d2)
    assert not np.allclose(d3, d4)


@pytest.mark.parametrize(
    "tf, label, rkw, params, mcircles",
    [
        (tf3, None, None, None, False),
        ((n3, d3), None, None, None, False),
        (n3 / d3, None, None, None, False),
        (tf3, None, None, None, True),
        ((n3, d3), None, None, None, True),
        (n3 / d3, None, None, None, True),
        (tf3, "test", None, None, False),
        ((n3, d3), "test", None, None, False),
        (n3 / d3, "test", None, None, False),
        (tf3, "test", None, mod_params, False),
        ((n3, d3), "test", None, mod_params, False),
        (n3 / d3, "test", None, mod_params, False),
    ]
)
def test_nyquist(tf, label, rkw, params, mcircles):
    kwargs = {"n": 10}
    if params:
        # params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    series = nyquist(tf, label=label, rendering_kw=rkw,
        m_circles=mcircles, **kwargs)
    assert len(series) == 1 if not mcircles else 2
    s = series[0]
    if not mcircles:
        assert isinstance(s, NyquistLineSeries)
    else:
        assert isinstance(s, MCirclesSeries)
    s.get_data()
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize(
    "mcircles", [False, True]
)
def test_nyquist_mimo(mcircles):
    series = nyquist(tf_mimo_sympy, m_circles=mcircles)
    assert len(series) == 6 if not mcircles else 7
    if not mcircles:
        assert all(isinstance(s, NyquistLineSeries) for s in series)
    else:
        assert isinstance(series[0], MCirclesSeries)
        assert all(isinstance(s, NyquistLineSeries) for s in series[1:])


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
    assert len(series) == 2
    assert isinstance(series[0], NGridLineSeries)
    assert isinstance(series[1], NicholsLineSeries)
    s = series[1]
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

    # verify that by setting axis limits and `auto=True`, the series
    # computes appropriate values to distribute the grid evenly over
    # the specified space.
    series = sgrid(
        xlim=(-11, 1), ylim=(-10, 10), auto=True, show_control_axis=False)
    assert len(series) == 1
    xi_dict, wn_dict, tp_dict, ts_dict = series[0].get_data()
    xi_ret = [k[0] for k in xi_dict.keys()]
    wn_ret = list(wn_dict.keys())
    assert np.allclose(xi_ret, [
        0.26515648302104233, 0.48191874977215593, 0.6363829547955636,
        0.855197831554018, 0.9570244044334736, 0, 1])
    assert np.allclose(wn_ret, [1.83333333, 3.66666667, 5.5, 7.33333333, 9.16666667])

    series = sgrid(
        xlim=(-11, 1), ylim=(-10, 10), auto=True, show_control_axis=True)
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


@pytest.mark.parametrize(
    "tf_cont, tf_dt, func",
    [
        (tf_cont_control_2, tf_dt_control_2, impulse_response),
        (tf_cont_scipy_2, tf_dt_scipy_2, impulse_response),
        (tf_cont_control_2, tf_dt_control_2, step_response),
        (tf_cont_scipy_2, tf_dt_scipy_2, step_response),
        (tf_cont_control_2, tf_dt_control_2, ramp_response),
        (tf_cont_scipy_2, tf_dt_scipy_2, ramp_response),
    ]
)
def test_discrete_responses(tf_cont, tf_dt, func):
    # verify that discrete-time systems produce different data than
    # continuous time systems.

    series1 = func(tf_cont, upper_limit=10, n=10, control=True)
    assert len(series1) == 1
    assert isinstance(series1[0], SystemResponseSeries)
    d1 = series1[0].get_data()
    assert (len(d1) == 2) and all(len(t) == 10 for t in d1)

    series2 = func(tf_dt, upper_limit=10, n=10, control=True)
    assert len(series2) == 1
    assert isinstance(series2[0], SystemResponseSeries)
    d2 = series2[0].get_data()
    assert (len(d2) == 2) and all(len(t) != 10 for t in d2)


@pytest.mark.parametrize(
    "arg, n_series, params",
    [
        (None, 11, None),
        (-20, 1, None),
        ([-20, -10, -6, -4, -2, 0], 6, None),
        (a, 1, {a: 1}),
        ([a, b, c], 3, {a: 1, b: 2, c: 3}),
    ]
)
def test_mcircles(arg, n_series, params):
    kwargs = {}
    if params:
        kwargs["params"] = params

    series = mcircles(arg, **kwargs)
    assert len(series) == 1
    assert isinstance(series[0], MCirclesSeries)
    d = series[0].get_data()
    assert len(d) == n_series
    assert all(len(t) == 3 for t in d)
    assert series[0].is_interactive == (True if params else False)
