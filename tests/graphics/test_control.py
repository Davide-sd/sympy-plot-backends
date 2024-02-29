import numpy as np
import pytest
from spb import (
    control_axis, pole_zero, step_response, impulse_response, ramp_response,
    bode_magnitude, bode_phase, nyquist, nichols, sgrid, root_locus
)
from spb.series import (
    LineOver1DRangeSeries, HVLineSeries, List2DSeries, NyquistLineSeries,
    NicholsLineSeries, SGridLineSeries, RootLocusSeries
)
from sympy.abc import a, b, c, d, e, s
from sympy.physics.control.lti import TransferFunction


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
    c: (5, 0, 8),
    d: (2, 0, 8),
    e: (1, 0, 8),
}
mod_params = {
    a: (1, 0, 8),
    b: (6, 0, 8),
}


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

    series = step_response(tf, label=label, rendering_kw=rkw, **kwargs)
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
def test_impulse_response(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    series = impulse_response(tf, label=label, rendering_kw=rkw, **kwargs)
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
def test_ramp_response(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    series = ramp_response(tf, label=label, rendering_kw=rkw, **kwargs)
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
def test_bode_magnitude(tf, label, rkw, params):
    kwargs = {"n": 10}
    if params:
        params = {k: v[0] for k, v in params.items()}
        kwargs["params"] = params

    series = bode_magnitude(tf, label=label, rendering_kw=rkw, **kwargs)
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
    d = series[0].get_data()
    assert np.allclose(d[0], [.1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99])
    assert np.allclose(d[1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    series = sgrid(show_control_axis=False)
    assert len(series) == 1
    d = series[0].get_data()
    assert np.allclose(d[0], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99, 1])
    assert np.allclose(d[1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    series = sgrid(show_control_axis=True)
    assert len(series) == 1
    d = series[0].get_data()
    assert np.allclose(d[0], [.1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99])
    assert np.allclose(d[1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    series = sgrid(xi=False, show_control_axis=False)
    assert len(series) == 1
    d = series[0].get_data()
    assert len(d[0]) == 0
    assert np.allclose(d[1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    series = sgrid(wn=False, show_control_axis=False)
    assert len(series) == 1
    d = series[0].get_data()
    assert np.allclose(d[0], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99, 1])
    assert len(d[1]) == 0

    series = sgrid(xlim=(-11, 1), ylim=(-10, 10), show_control_axis=False)
    assert len(series) == 1
    d = series[0].get_data()
    assert np.allclose(d[0], [
        0.26515648302104233, 0.48191874977215593, 0.6363829547955636,
        0.855197831554018, 0.9570244044334736, 0, 1])
    assert np.allclose(d[1], [1.83333333, 3.66666667, 5.5, 7.33333333, 9.16666667])

    series = sgrid(xlim=(-11, 1), ylim=(-10, 10), show_control_axis=True)
    assert len(series) == 1
    d = series[0].get_data()
    assert np.allclose(d[0], [
        0.26515648302104233, 0.48191874977215593, 0.6363829547955636,
        0.855197831554018, 0.9570244044334736])
    assert np.allclose(d[1], [1.83333333, 3.66666667, 5.5, 7.33333333, 9.16666667])


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
