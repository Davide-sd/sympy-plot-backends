import pytest
from pytest import raises, warns
from spb import (
    plot, plot3d, plot_implicit, plot_vector,
    plot_complex, plot_complex_list, plot_geometry,
    MB,
)
from spb.utils import (
    _create_missing_ranges, _plot_sympify,
    _validate_kwargs, prange, extract_solution, tf_to_control
)
from sympy import (
    symbols, Expr, Tuple, Integer, sin, cos, Matrix,
    I, Polygon, solveset, FiniteSet, ImageSet,
)
from sympy.physics.control import TransferFunction
import numpy as np
import control as ct
import scipy.signal as signal

x, a, b = symbols("x a b")


def test_plot_sympify():
    x, y = symbols("x, y")

    # argument is already sympified
    args = x + y
    r = _plot_sympify(args)
    assert r == args

    # one argument needs to be sympified
    args = (x + y, 1)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Integer)

    # string and dict should not be sympified
    args = (x + y, (x, 0, 1), "str", 1, {1: 1, 2: 2.0})
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 5
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[2], str)
    assert isinstance(r[3], Integer)
    assert (
        isinstance(r[4], dict)
        and isinstance(r[4][1], int)
        and isinstance(r[4][2], float)
    )

    # nested arguments containing strings
    args = ((x + y, (y, 0, 1), "a"), (x + 1, (x, 0, 1), "$f_{1}$"))
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Tuple)
    assert isinstance(r[0][1], Tuple)
    assert isinstance(r[0][1][1], Integer)
    assert isinstance(r[0][2], str)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[1][1], Tuple)
    assert isinstance(r[1][1][1], Integer)
    assert isinstance(r[1][2], str)


def test_create_missing_ranges():
    x, y = symbols("x, y")

    # user don't provide any range -> return a default range
    r = _create_missing_ranges({x}, [], 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert isinstance(r[0], (Tuple, tuple))
    assert r[0] == (x, -10, 10)

    r = _create_missing_ranges({x, y}, [], 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, -10, 10) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, -10, 10)
    assert r[0] != r[1]

    # not enough ranges provided by the user -> create default ranges
    r = _create_missing_ranges(
        {x, y},
        [
            (x, 0, 1),
        ],
        2,
    )
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, 0, 1) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, 0, 1)
    assert r[0] != r[1]

    # too many free symbols
    raises(ValueError, lambda: _create_missing_ranges({x, y}, [], 1))
    raises(
        ValueError,
        lambda: _create_missing_ranges({x, y}, [(x, 0, 5), (y, 0, 1)], 1)
    )


def test_raise_warning_keyword_validation():
    # verify that plotting functions raise warning when a mispelled keyword
    # argument is provided.
    # NOTE: there is pytest.warn, however I can't get it to work here. I don't
    # understand its error message :|
    # Hence, I'm going to do it my own way: execute the _validate_kwargs
    # function and check that the warning message contains the expected
    # misspelled keywords.

    x, y, z = symbols("x:z")

    def do_test(p, kw, keys):
        msg = _validate_kwargs(p, **kw)
        assert all(k in msg for k in keys)

    # x_label should be xlabel: this is a Backend-related keyword
    kw = dict(adaptive=False, x_label="a")

    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot(sin(x), backend=MB, show=False, **kw)
        do_test(p, kw, ["x_label", "xlabel"])

    # adapt should be adaptive: this is a LineOver1DRangeSeries keyword
    kw = dict(adapt=False)

    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot(sin(x), backend=MB, show=False, **kw)
        do_test(p, kw, ["adapt", "adaptive"])

    # surface_colors should be surface_color: this is a SurfaceBaseSeries
    # keyword
    kw = dict(surface_colors="r")
    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot3d(cos(x**2 + y**2), backend=MB, show=False, **kw)
        do_test(p, kw, ["surface_colors", "surface_color"])

    # deptt should be depth: this is a ImplicitSeries keyword
    kw = dict(deptt=2)
    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot_implicit(cos(x), backend=MB, show=False, **kw)
        do_test(p, kw, ["deptt", "depth"])

    # streamline should be streamlines: this is a VectorBase keyword
    kw = dict(streamline=True)
    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot_vector(Matrix([sin(y), cos(x)]), backend=MB, show=False, **kw)
        do_test(p, kw, ["streamline", "streamlines"])

    # phase_res should be phaseres
    kw = dict(phase_res=3)
    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot_complex(z, (z, -2 - 2j, 2 + 2j), backend=MB, show=False, **kw)
        do_test(p, kw, ["phase_res", "phaseres"])

    # render_kw should be rendering_kw
    kw = dict(render_kw={"color": "r"})
    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot_complex_list(3 + 2 * I, backend=MB, show=False, **kw)
        do_test(p, kw, ["render_kw", "rendering_kw"])

    # is_fille should be is_filled
    kw = dict(is_fille=False)
    with warns(
        UserWarning,
        match="The following keyword arguments are unused."
    ):
        p = plot_geometry(Polygon((4, 0), 4, n=5), backend=MB, show=False)
        do_test(p, kw, ["is_fille", "is_filled"])


@pytest.mark.parametrize("_sym, _min, _max", [
    (x, 0, 5),
    (x, a, 5),
    (x, 0, b),
    (x, a, b),
    (x, a * b, a + b)
])
def test_prange_instantiation(_sym, _min, _max):
    # verify that prange raises the necessary errors

    p = prange(_sym, _min, _max)
    assert isinstance(p, prange)


def test_prange_instantiation_errors():
    x, a, b = symbols("x a b")

    # too many elements
    raises(ValueError, lambda: prange(x, a, b, 5))
    # too few elements
    raises(ValueError, lambda: prange(x, a))
    # first element is not a symbol
    raises(TypeError, lambda: prange(5, a, b))
    # range symbols is present in the starting position
    raises(ValueError, lambda: prange(x, x * a, b))
    # range symbols is present in the ending position
    raises(ValueError, lambda: prange(x, a, x * b))


def test_extract_solution():
    x = symbols("x")

    sol = solveset(cos(10 * x))
    assert sol.has(ImageSet)
    res = extract_solution(sol)
    assert len(res) == 20
    assert isinstance(res, FiniteSet)

    res = extract_solution(sol, 20)
    assert len(res) == 40
    assert isinstance(res, FiniteSet)


def _is_control_tf_equals(A, B):
    assert isinstance(A, ct.TransferFunction)
    assert isinstance(B, ct.TransferFunction)
    assert A.ninputs == B.ninputs
    assert A.noutputs == B.noutputs
    for n1, n2 in zip(A.num, B.num):
        assert np.allclose(n1, n2)
    for d1, d2 in zip(A.den, B.den):
        assert np.allclose(d1, d2)
    assert A.dt == B.dt


def test_tf_to_control_1():
    # symbolic expressions to ct.TransferFunction
    s = symbols("s")

    G = s / (s+4) / (s+8)
    _is_control_tf_equals(
        tf_to_control(G),
        ct.tf([1., 0.], [1., 12., 32.])
    )

    _is_control_tf_equals(
        tf_to_control(2 * G),
        ct.tf([2., 0.], [1., 12., 32.])
    )


def test_tf_to_control_2():
    # sympy.physics.control.TransferFunction to ct.TransferFunction
    s = symbols("s")

    G = TransferFunction(25, s**2 + 10*s + 25, s)
    _is_control_tf_equals(
        tf_to_control(G),
        ct.tf([25.], [1., 10., 25.])
    )

    G = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
    _is_control_tf_equals(
        tf_to_control(G),
        ct.tf([8., 18., 32], [1., 6, 14., 24.])
    )


def test_tf_to_control_3():
    # scipy.signal.TransferFunction to ct.TransferFunction

    G = signal.TransferFunction([1, 3, 3], [1, 2, 1])
    _is_control_tf_equals(
        tf_to_control(G),
        ct.tf([1., 3., 3.], [1., 2., 1.])
    )
