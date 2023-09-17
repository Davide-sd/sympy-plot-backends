import numpy as np
import pytest
from pytest import raises
from spb.graphics import (
    complex_points, line_abs_arg_colored, line_abs_arg, line_real_imag,
    surface_abs_arg, surface_real_imag, domain_coloring, analytic_landscape,
    riemann_sphere_2d, riemann_sphere_3d
)
from spb.series import (
    ComplexPointSeries, AbsArgLineSeries, LineOver1DRangeSeries,
    ComplexSurfaceSeries, ComplexDomainColoringSeries,
    ComplexParametric3DLineSeries, RiemannSphereSeries, Parametric2DLineSeries
)
from sympy import exp, pi, I, symbols, cos, sin, sqrt, sympify, Dummy


p1 = symbols("p1")


@pytest.mark.parametrize("label, rkw, line, params", [
    (None, None, False, None),
    (None, None, True, None),
    ("test", {"color": "r"}, True, {p1: (1, 0, 2)}),
])
def test_complex_points(label, rkw, line, params):
    series = complex_points(
        3 + 2 * I, 4 * I, 2,
        line=line, rendering_kw=rkw, label=label
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, ComplexPointSeries)
    assert s.get_label(False) == label
    assert s.is_point == (not line)
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize("label, rkw, line, params", [
    (None, None, False, None),
    (None, None, True, None),
    ("test", {"color": "r"}, True, {p1: (1, 0, 2)}),
])
def test_complex_points_2(label, rkw, line, params):
    z = symbols("z")
    expr1 = z * exp(2 * pi * I * z)
    expr2 = 2 * expr1
    n = 15
    l1 = [expr1.subs(z, t / n) for t in range(n)]
    l2 = [expr2.subs(z, t / n) for t in range(n)]
    series = complex_points(l1, label=label, line=line, rendering_kw=rkw)
    s = series[0]
    assert len(series) == 1
    assert isinstance(s, ComplexPointSeries)
    assert s.get_label(False) == label
    assert s.is_point == (not line)
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params
    raises(TypeError, lambda: complex_points(l1, l2))


@pytest.mark.parametrize("rang, label, rkw, params", [
    (None, None, None, None),
    ((-2, 3), "test", {"cmap": "hsv"}, None),
    (None, None, None, {p1: (1, 0, 2)}),
    ((-2, 3), "test", {"cmap": "hsv"}, {p1: (1, 0, 2)}),
])
def test_line_abs_arg_colored(default_range, rang, label, rkw, params):
    x = symbols("x")
    expr = cos(x) + sin(I * x)
    kwargs = {}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = line_abs_arg_colored(
        expr, range=r, label=label, rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, AbsArgLineSeries)
    assert s.expr == expr
    assert s.ranges[0] == default_range(x) if not rang else r
    assert s.get_label(False) == "Arg" if not label else label
    assert s.rendering_kw == {} if not rkw else rkw
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize("rang, label, rkw, abs, arg, params", [
    (None, None, None, True, True, None),
    (None, None, None, True, False, None),
    (None, None, None, False, True, None),
    (None, None, None, False, False, None),
    ((-2, 3), "test", {"linestyle": "--"}, True, True, None),
    ((-2, 3), "test", {"linestyle": "--"}, True, True, {p1: (1, 0, 2)}),
])
def test_line_abs_arg(default_range, rang, label, rkw, abs, arg, params):
    x = symbols("x")
    expr = sqrt(x)
    kwargs = {"n": 10}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = line_abs_arg(
        expr, range=r, label=label,
        rendering_kw=rkw, abs=abs, arg=arg, **kwargs
    )
    ref = LineOver1DRangeSeries(
        expr, r if r else default_range(x),
        label, rendering_kw=rkw, **kwargs
    )
    assert len(series) == sum([abs, arg])
    for s in series:
        assert isinstance(s, LineOver1DRangeSeries)
        assert s.expr == expr
        assert s.ranges[0] == default_range(x) if not rang else r
        assert s.rendering_kw == {} if not rkw else rkw
        assert s.is_interactive == (len(s.params) > 0)
        assert s.params == {} if not params else params
        assert not np.allclose(ref.get_data(), s.get_data())

    if abs and arg:
        assert series[0].get_label(False) == "Abs" if not label else label
        assert series[1].get_label(False) == "Arg" if not label else label
    elif abs:
        assert series[0].get_label(False) == "Abs" if not label else label
    elif arg:
        assert series[0].get_label(False) == "Arg" if not label else label


@pytest.mark.parametrize("rang, label, rkw, real, imag, params", [
    (None, None, None, True, True, None),
    (None, None, None, True, False, None),
    (None, None, None, False, True, None),
    (None, None, None, False, False, None),
    ((-2, 3), "test", {"linestyle": "--"}, True, True, None),
    ((-2, 3), "test", {"linestyle": "--"}, True, True, {p1: (1, 0, 2)}),
])
def test_line_real_imag(default_range, rang, label, rkw, real, imag, params):
    x = symbols("x")
    expr = sqrt(x)
    kwargs = {"n": 10}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = line_real_imag(
        expr, range=r, label=label,
        rendering_kw=rkw, real=real, imag=imag, **kwargs
    )
    ref = LineOver1DRangeSeries(
        expr, r if r else default_range(x),
        label, rendering_kw=rkw, **kwargs
    )
    assert len(series) == sum([real, imag])
    for s in series:
        assert isinstance(s, LineOver1DRangeSeries)
        assert s.expr == expr
        assert s.ranges[0] == default_range(x) if not rang else r
        assert s.rendering_kw == {} if not rkw else rkw
        assert s.is_interactive == (len(s.params) > 0)
        assert s.params == {} if not params else params
        assert not np.allclose(ref.get_data(), s.get_data())

    if real and imag:
        assert series[0].get_label(False) == "Re" if not label else label
        assert series[1].get_label(False) == "Im" if not label else label
    elif real:
        assert series[0].get_label(False) == "Re" if not label else label
    elif imag:
        assert series[0].get_label(False) == "Im" if not label else label


@pytest.mark.parametrize("rang, label, rkw, abs, arg, params, wf, wf_n1, wf_n2", [
    (None, None, None, True, True, None, False, None, None),
    (None, None, None, True, False, None, False, None, None),
    (None, None, None, False, True, None, False, None, None),
    (None, None, None, False, False, None, False, None, None),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, None, None, None, None),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, None, True, 10, 10),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, {p1: (1, 0, 2)}, False, None, None),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, {p1: (1, 0, 2)}, True, 10, 10),
])
def test_surface_abs_arg(
    default_complex_range, rang, label, rkw,
    abs, arg, params, wf, wf_n1, wf_n2
):
    x = symbols("x")
    expr = sqrt(x)
    kwargs = {"wireframe": wf, "wf_n1": wf_n1, "wf_n2": wf_n2}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = surface_abs_arg(
        expr, range=r, label=label,
        rendering_kw=rkw, abs=abs, arg=arg, **kwargs
    )
    assert len(series) == sum([abs, arg]) * (1 + ((wf_n1 + wf_n2) if wf else 0))

    surface_series = [s for s in series if isinstance(s, ComplexSurfaceSeries)]
    for s in surface_series:
        assert s.expr == expr
        dcr = default_complex_range(x)
        assert s.ranges[0][0] == x
        assert s.ranges[0][1] == sympify(dcr[1]) if not rang else r[1]
        assert s.ranges[0][2] == sympify(dcr[2]) if not rang else r[2]
        assert s.rendering_kw == {} if not rkw else rkw
        assert s.is_interactive == (len(s.params) > 0)
        assert s.params == {} if not params else params

    if abs and arg:
        lbl = surface_series[0].get_label(False)
        assert lbl == "Abs" if not label else label
        lbl = surface_series[1].get_label(False)
        assert lbl == "Arg" if not label else label
    elif abs:
        lbl = surface_series[0].get_label(False)
        assert lbl == "Abs" if not label else label
    elif arg:
        lbl = surface_series[0].get_label(False)
        assert lbl == "Arg" if not label else label

    wf_series = [s for s in series if isinstance(s, ComplexParametric3DLineSeries)]
    assert len(wf_series) == sum([abs, arg]) * ((wf_n1 + wf_n2) if wf else 0)


@pytest.mark.parametrize("rang, label, rkw, real, imag, params, wf, wf_n1, wf_n2", [
    (None, None, None, True, True, None, False, None, None),
    (None, None, None, True, False, None, False, None, None),
    (None, None, None, False, True, None, False, None, None),
    (None, None, None, False, False, None, False, None, None),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, None, None, None, None),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, None, True, 10, 10),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, {p1: (1, 0, 2)}, False, None, None),
    ((-2-2j, 3+3j), "test", {"opacity": "0.5"}, True, True, {p1: (1, 0, 2)}, True, 10, 10),
])
def test_surface_real_imag(
    default_complex_range, rang, label, rkw,
    real, imag, params, wf, wf_n1, wf_n2
):
    x = symbols("x")
    expr = sqrt(x)
    kwargs = {"wireframe": wf, "wf_n1": wf_n1, "wf_n2": wf_n2}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = surface_real_imag(
        expr, range=r, label=label,
        rendering_kw=rkw, real=real, imag=imag, **kwargs
    )
    assert len(series) == sum([real, imag]) * (1 + ((wf_n1 + wf_n2) if wf else 0))

    surface_series = [
        s for s in series if isinstance(s, ComplexSurfaceSeries)
    ]
    for s in surface_series:
        assert s.expr == expr
        dcr = default_complex_range(x)
        assert s.ranges[0][0] == x
        assert s.ranges[0][1] == sympify(dcr[1]) if not rang else r[1]
        assert s.ranges[0][2] == sympify(dcr[2]) if not rang else r[2]
        assert s.rendering_kw == {} if not rkw else rkw
        assert s.is_interactive == (len(s.params) > 0)
        assert s.params == {} if not params else params

    if real and imag:
        lbl = surface_series[0].get_label(False)
        assert lbl == "Re" if not label else label
        lbl = surface_series[1].get_label(False)
        assert lbl == "Im" if not label else label
    elif real:
        lbl = surface_series[0].get_label(False)
        assert lbl == "Re" if not label else label
    elif imag:
        lbl = surface_series[0].get_label(False)
        assert lbl == "Im" if not label else label

    wf_series = [
        s for s in series
        if isinstance(s, ComplexParametric3DLineSeries)
    ]
    assert len(wf_series) == sum([real, imag]) * ((wf_n1 + wf_n2) if wf else 0)


@pytest.mark.parametrize("rang, coloring, params", [
    (None, "a", None),
    ((-2-2j, 3+3j), "b", None),
    (None, "c", {p1: (1, 0, 2)}),
    ((-2-2j, 3+3j), "d", {p1: (1, 0, 2)}),
])
def test_domain_coloring(default_complex_range, rang, coloring, params):
    x = symbols("x")
    expr = sin(x)
    kwargs = {}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = domain_coloring(expr, range=r, coloring=coloring, **kwargs)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, ComplexDomainColoringSeries)
    assert not s.is_3Dsurface
    assert s.expr == expr
    dcr = default_complex_range(x)
    assert s.ranges[0][0] == x
    assert s.ranges[0][1] == sympify(dcr[1]) if not rang else r[1]
    assert s.ranges[0][2] == sympify(dcr[2]) if not rang else r[2]
    assert s.coloring == coloring
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize("rang, coloring, params", [
    (None, "a", None),
    ((-2-2j, 3+3j), "b", None),
    (None, "c", {p1: (1, 0, 2)}),
    ((-2-2j, 3+3j), "d", {p1: (1, 0, 2)}),
])
def test_analytic_landscape(default_complex_range, rang, coloring, params):
    x = symbols("x")
    expr = sin(x)
    kwargs = {}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = analytic_landscape(expr, range=r, coloring=coloring, **kwargs)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, ComplexDomainColoringSeries)
    assert s.is_3Dsurface
    assert s.expr == expr
    dcr = default_complex_range(x)
    assert s.ranges[0][0] == x
    assert s.ranges[0][1] == sympify(dcr[1]) if not rang else r[1]
    assert s.ranges[0][2] == sympify(dcr[2]) if not rang else r[2]
    assert s.coloring == coloring
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params


@pytest.mark.parametrize("rang, coloring, at_infinity, riemann_mask, params", [
    (None, "a", True, True, None),
    ((-2-2j, 3+3j), "b", False, True, None),
    ((-2-2j, 3+3j), "c", True, True, {p1: (1, 0, 2)}),
    ((-2-2j, 3+3j), "c", False, True, {p1: (1, 0, 2)}),
    (None, "a", True, False, None),
    ((-2-2j, 3+3j), "b", False, False, None),
    ((-2-2j, 3+3j), "c", True, False, {p1: (1, 0, 2)}),
    ((-2-2j, 3+3j), "c", False, False, {p1: (1, 0, 2)}),
])
def test_riemann_sphere_2d(rang, coloring, at_infinity, riemann_mask, params):
    x = symbols("x")
    expr = (x - 1) / (x**2 + x + 2)

    kwargs = {}
    if params:
        kwargs["params"] = params

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    series = riemann_sphere_2d(
        expr, range=r, coloring=coloring,
        riemann_mask=riemann_mask, at_infinity=at_infinity, **kwargs
    )
    assert len(series) == 1 + (1 if riemann_mask else 0)
    s = series[0]
    assert isinstance(s, ComplexDomainColoringSeries)
    assert not s.is_3Dsurface
    if not at_infinity:
        assert s.expr == expr
    dcr = (x, -1.25 - 1.25 * I, 1.25 + 1.25 * I)
    assert s.ranges[0][0] == x
    assert s.ranges[0][1] == sympify(dcr[1]) if not rang else r[1]
    assert s.ranges[0][2] == sympify(dcr[2]) if not rang else r[2]
    assert s.coloring == coloring
    assert s.at_infinity == at_infinity
    assert s.riemann_mask == riemann_mask
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == {} if not params else params
    if riemann_mask:
        assert isinstance(series[1], Parametric2DLineSeries)
        a = Dummy()
        fs = series[1].ranges[0][0]
        e1, e2 = series[1].expr
        assert e1.subs(fs, a) == cos(a)
        assert e2.subs(fs, a) == sin(a)


@pytest.mark.parametrize("coloring", ["a", "b"])
def test_riemann_sphere_3d(coloring):
    x = symbols("x")
    expr = (x - 1) / (x**2 + x + 2)

    series = riemann_sphere_3d(expr, coloring=coloring)
    assert len(series) == 2
    assert all(isinstance(t, RiemannSphereSeries) for t in series)
    assert all(t.coloring == coloring for t in series)
