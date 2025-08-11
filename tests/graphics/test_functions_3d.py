import numpy as np
import pytest
from spb.graphics import (
    line_parametric_3d, list_3d,
    surface, surface_parametric, surface_spherical,
    surface_revolution, implicit_3d, wireframe, plane
)
from spb.series import (
    Parametric3DLineSeries, SurfaceOver2DRangeSeries, ParametricSurfaceSeries,
    Implicit3DSeries, List3DSeries
)
from sympy import (
    symbols, cos, sin, pi, sqrt, exp, Plane, atan2
)


p1, p2 = symbols("p1 p2")


@pytest.mark.parametrize("rang, label, rkw, n, params", [
    (None, None, None, None, None),
    ((-2, 3), "test", {"color": "r"}, None, None),
    ((-2, 3), "test", {"color": "r"}, 10, None),
    ((-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}),
])
def test_line_parametric_3d(default_range, rang, label, rkw, n, params):
    x = symbols("x")

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    kwargs = {}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = line_parametric_3d(
        cos(x), sin(x), x, range=r, label=label,
        rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, Parametric3DLineSeries)
    assert s.expr == (cos(x), sin(x), x)
    assert s.ranges[0] == (default_range(x) if not rang else r)
    assert s.get_label(False) == ("x" if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert s.n[0] == (1000 if not n else n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.parametrize(
    "range1, range2, label, rkw, n, params, wf, wf_n1, wf_n2", [
        (None, None, None, None, None, None, False, None, None),
        ((-2, 3), None, None, None, None, None, False, None, None),
        (None, (-2, 3), None, None, None, None, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, True, 10, 10),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, True, 10, 10),
])
def test_surface(default_range, range1, range2, label, rkw, n, params,
    wf, wf_n1, wf_n2):
    x, y = symbols("x, y")

    r1 = (x, *range1) if isinstance(range1, (list, tuple)) else None
    r2 = (y, *range2) if isinstance(range2, (list, tuple)) else None
    kwargs = {"wireframe": wf}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = surface(
        cos(x*y), range1=r1, range2=r2, label=label,
        rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1 + ((wf_n1 + wf_n2) if wf else 0)
    s = series[0]
    assert isinstance(s, SurfaceOver2DRangeSeries)
    assert s.expr == cos(x*y)
    assert (s.ranges[0] == (default_range(x) if not range1 else r1)) or \
        (s.ranges[0] == (default_range(y) if not range1 else r1))
    assert (s.ranges[1] == (default_range(y) if not range2 else r2)) or \
        (s.ranges[1] == (default_range(x) if not range2 else r2))
    assert s.get_label(False) == ("cos(x*y)" if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert all(t == 100 if not n else n for t in s.n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)
    assert all(isinstance(t, Parametric3DLineSeries) for t in series[1:])


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.parametrize(
    "range1, range2, label, rkw, n, params, wf, wf_n1, wf_n2", [
        (None, None, None, None, None, None, False, None, None),
        ((-2, 3), None, None, None, None, None, False, None, None),
        (None, (-2, 3), None, None, None, None, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, True, 10, 10),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, True, 10, 10),
])
def test_surface_parametric(
    default_range, range1, range2, label, rkw, n,
    params, wf, wf_n1, wf_n2
):
    x, y = symbols("x, y")

    r1 = (x, *range1) if isinstance(range1, (list, tuple)) else None
    r2 = (y, *range2) if isinstance(range2, (list, tuple)) else None
    kwargs = {"wireframe": wf}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = surface_parametric(
        cos(x*y), sin(x*y), x*y, range1=r1, range2=r2, label=label,
        rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1 + ((wf_n1 + wf_n2) if wf else 0)
    s = series[0]
    assert isinstance(s, ParametricSurfaceSeries)
    assert s.expr == (cos(x*y), sin(x*y), x*y)
    assert (s.ranges[0] == (default_range(x) if not range1 else r1)) or \
        (s.ranges[0] == (default_range(y) if not range1 else r1))
    assert (s.ranges[1] == (default_range(y) if not range2 else r2)) or \
        (s.ranges[1] == (default_range(x) if not range2 else r2))
    assert s.get_label(False) == ("(cos(x*y), sin(x*y), x*y)" if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert all(t == 100 if not n else n for t in s.n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)
    assert all(isinstance(t, Parametric3DLineSeries) for t in series[1:])


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.parametrize(
    "range1, range2, label, rkw, n, params, wf, wf_n1, wf_n2", [
        (None, None, None, None, None, None, False, None, None),
        ((-4, 6), None, None, None, None, None, False, None, None),
        (None, (-2, 3), None, None, None, None, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, True, 10, 10),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, False, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, True, 10, 10),
])
def test_surface_spherical(
    range1, range2, label, rkw, n, params, wf, wf_n1, wf_n2
):
    theta, phi = symbols("theta, phi")
    # after enforcing the polar and azimuthal conditions, this is what
    # should be expected
    expected_range_theta = (theta, 0, pi)
    expected_range_phi = (phi, 0, 3)

    r1 = (theta, *range1) if isinstance(range1, (list, tuple)) else None
    r2 = (phi, *range2) if isinstance(range2, (list, tuple)) else None
    kwargs = {"wireframe": wf}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = surface_spherical(
        1, range_theta=r1, range_phi=r2, label=label,
        rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1 + ((wf_n1 + wf_n2) if wf else 0)
    s = series[0]
    assert isinstance(s, ParametricSurfaceSeries)
    if r1 and r2:
        assert s.expr == (sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta))
        assert (s.ranges[0] == expected_range_theta)
        assert (s.ranges[1] == expected_range_phi)
    elif r1:
        assert (s.ranges[0] == expected_range_theta)
    elif r2:
        assert (s.ranges[1] == expected_range_phi)

    if not label:
        assert "Dummy" in s.get_label(False)
    else:
        assert s.get_label(False) == label
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert all(t == 100 if not n else n for t in s.n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)
    assert all(isinstance(t, Parametric3DLineSeries) for t in series[1:])


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.parametrize(
    "range1, range2, label, rkw, n, params, wf, wf_n1, wf_n2", [
        ((0, pi), None, None, None, None, None, False, None, None),
        ((0, pi), (0, 2*pi), "test", {"color": "r"}, 10, None, False, None, None),
        ((0, pi), (0, 2*pi), "test", {"color": "r"}, 10, None, True, 10, 10),
        ((0, pi), (0, 2*pi), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, False, None, None),
        ((0, pi), (0, 2*pi), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}, True, 10, 10),
])
def test_surface_revolution(default_range, range1, range2, label, rkw, n, params,
    wf, wf_n1, wf_n2):
    t, phi = symbols("t, phi")

    r1 = (t, *range1) if isinstance(range1, (list, tuple)) else None
    r2 = (phi, *range2) if isinstance(range2, (list, tuple)) else None
    kwargs = {"wireframe": wf}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = surface_revolution(
        cos(t), r1, r2, label=label, rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1 + ((wf_n1 + wf_n2) if wf else 0)
    s = series[0]
    assert isinstance(s, ParametricSurfaceSeries)
    assert s.expr == (sqrt(t**2)*cos(phi + atan2(0, t)), sqrt(t**2)*sin(phi + atan2(0, t)), cos(t))
    if r1:
        assert (s.ranges[0] == (default_range(t) if not range1 else r1)) or \
            (s.ranges[0] == (default_range(phi) if not range1 else r1))
    if r2:
        assert (s.ranges[1] == (default_range(t) if not range2 else r2)) or \
            (s.ranges[1] == (default_range(phi) if not range2 else r2))

    lbl = "(sqrt(t**2)*cos(phi + atan2(0, t)), sqrt(t**2)*sin(phi + atan2(0, t)), cos(t))"
    assert s.get_label(False) == (lbl if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert all(t == 100 if not n else n for t in s.n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)
    assert all(isinstance(t, Parametric3DLineSeries) for t in series[1:])


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.filterwarnings("ignore:Not enough ranges were provided.")
@pytest.mark.parametrize(
    "range1, range2, range3, label, rkw, n", [
        (None, None, None, None, None, None),
        ((-2, 3), None, None, None, None, None),
        (None, (-2, 3), None, None, None, None),
        (None, None, (-2, 3), None, None, None),
        ((-4, 6), (-2, 3), (1, 5), "test", {"color": "r"}, 10),
        ((-4, 6), (-2, 3), (1, 5),  "test", {"color": "r"}, 10),
        ((-4, 6), (-2, 3), (1, 5),  "test", {"color": "r"}, 10),
        ((-4, 6), (-2, 3), (1, 5),  "test", {"color": "r"}, 10),
])
def test_implicit_3d(default_range, range1, range2, range3, label, rkw, n):
    x, y, z = symbols("x, y, z")

    r1 = (x, *range1) if isinstance(range1, (list, tuple)) else None
    r2 = (y, *range2) if isinstance(range2, (list, tuple)) else None
    r3 = (z, *range3) if isinstance(range3, (list, tuple)) else None
    kwargs = {}
    if n:
        kwargs["n"] = n

    series = implicit_3d(
        x**2 + y**3 - z**2, range1=r1, range2=r2, range3=r3,
        label=label, rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, Implicit3DSeries)
    assert s.expr == x**2 + y**3 - z**2
    assert (s.ranges[0] == (default_range(x) if not range1 else r1)) or \
        (s.ranges[0] == (default_range(y) if not range1 else r1)) or \
        (s.ranges[0] == (default_range(z) if not range1 else r1))
    assert (s.ranges[1] == (default_range(y) if not range2 else r2)) or \
        (s.ranges[1] == (default_range(x) if not range2 else r2)) or \
        (s.ranges[1] == (default_range(z) if not range2 else r2))
    assert (s.ranges[2] == (default_range(y) if not range3 else r3)) or \
        (s.ranges[2] == (default_range(x) if not range3 else r3)) or \
        (s.ranges[2] == (default_range(z) if not range3 else r3))
    assert s.get_label(False) == ("x**2 + y**3 - z**2" if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert s.n[0] == (60 if not n else n)


@pytest.mark.parametrize(
    "label, rkw", [
        (None, None),
        ("test", {"color": "r"})
])
def test_list_3d(label, rkw):
    zz = np.linspace(0, 6*np.pi, 100)
    xx = zz * np.cos(zz)
    yy = zz * np.sin(zz)
    series = list_3d(xx, yy, zz, label=label, rendering_kw=rkw)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, List3DSeries)
    assert s.get_label(False) == label
    assert s.rendering_kw == ({} if not rkw else {"color": "r"})


@pytest.mark.parametrize(
    "n1, n2, n, rkw, params", [
        (10, 10, 1000, None, None),
        (20, 10, 100, None, None),
        (10, 25, 10, None, None),
        (5, 25, 1000, {"color": "r"}, {p1: (1, 0, 2)}),
])
def test_wireframe(n1, n2, n, rkw, params):
    x, y = symbols("x y")
    if not params:
        params = {}
        expr = cos(x*y) * exp(-sqrt(x**2 + y**2) / 3)
    else:
        expr = cos(x*y) * exp(-sqrt(x**2 + p1**2) / 3)
    s = surface(
        expr, (x, -pi, pi), (y, -2*pi, 2*pi), params=params
    )[0]

    series = wireframe(s, n1=n1, n2=n2, n=n, rendering_kw=rkw, params=params)
    assert len(series) == n1 + n2
    assert all(isinstance(t, Parametric3DLineSeries) for t in series)
    s_along_x = [t for t in series if (t.ranges[0][1] == -2*pi)
        and (t.ranges[0][2] == 2*pi)]
    assert len(s_along_x) == n1
    s_along_y = [
        t for t in series if (t.ranges[0][1] == -pi)
        and (t.ranges[0][2] == pi)
    ]
    assert len(s_along_y) == n2
    assert all(t.n[0] == n for t in series)
    assert all(t.rendering_kw == ({} if not rkw else rkw) for t in series)
    assert all(t.is_interactive == (len(t.params) > 0) for t in series)
    assert all(t.params == ({} if not params else params) for t in series)


@pytest.mark.parametrize(
    "label, n1, n2, n3, rkw, params", [
        (None, None, None, None, None, None),
        (None, 10, None, None, None, None),
        (None, None, 10, None, None, None),
        (None, None, None, 10, None, None),
        (None, 10, 15, 20, None, None),
        ("test", 10, 15, 25, {"color": "r"}, None),
        ("test", 10, 15, 25, {"color": "r"}, {p1: (1, 0, 2)}),
])
def test_plane(label, n1, n2, n3, rkw, params):
    x, y, z = symbols("x:z")
    ranges = [(x, -5, 5), (y, -5, 5), (z, -5, 5)]
    default_n = 20
    p = Plane(
        (p1 if params else 0, 0, 0),
        (1, 0, 0)
    )
    kwargs = dict(rendering_kw=rkw,)
    if params: kwargs["params"] = params
    if n1:
        kwargs["n1"] = n1
    if n2:
        kwargs["n2"] = n2
    if n3:
        kwargs["n3"] = n3

    series = plane(p, *ranges, label=label, **kwargs)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s.expr, Plane)
    assert s.n[0] == (default_n if not n1 else n1)
    assert s.n[1] == (default_n if not n2 else n2)
    assert s.n[2] == (default_n if not n3 else n3)
    assert s.get_label(False) == (str(p) if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    is_interactive = (False if params is None else (len(params) > 0))
    assert s.is_interactive == is_interactive
    assert s.params == ({} if not params else params)
