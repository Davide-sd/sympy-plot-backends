from sympy.core.symbol import symbols
from sympy.core.expr import Expr
from sympy.core.containers import Tuple
from sympy.core.numbers import Integer
from sympy.functions.elementary.trigonometric import sin, cos
from spb.utils import _check_arguments, _create_ranges, _plot_sympify
from pytest import raises


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
    assert isinstance(r[4], dict) and isinstance(r[4][1], int) and isinstance(r[4][2], float)

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


def test_create_ranges():
    x, y = symbols("x, y")

    # user don't provide any range -> return a default range
    r = _create_ranges({x}, [], 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert isinstance(r[0], (Tuple, tuple))
    assert r[0] == (x, -10, 10)

    r = _create_ranges({x, y}, [], 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, -10, 10) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, -10, 10)
    assert r[0] != r[1]

    # not enough ranges provided by the user -> create default ranges
    r = _create_ranges(
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
    raises(ValueError, lambda: _create_ranges({x, y}, [], 1))
    raises(ValueError, lambda: _create_ranges({x, y}, [(x, 0, 5), (y, 0, 1)], 1))


def test_check_arguments_plot():
    ### Test arguments for plot()

    x, y = symbols("x, y")

    # single expressions
    args = _plot_sympify((x + 1,))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, (x, -10, 10), "x + 1", None)

    # single expressions with range
    args = _plot_sympify((x + 1, (x, -2, 2)))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, (x, -2, 2), "x + 1", None)

    # single expressions with range, label and rendering-kw dictionary
    args = _plot_sympify((x + 1, (x, -2, 2), "test", {0: 0, 1: 1}))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, (x, -2, 2), "test", {0: 0, 1: 1})

    # multiple expressions
    args = _plot_sympify((x + 1, x ** 2))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, (x, -10, 10), "x + 1", None)
    assert r[1] == (x ** 2, (x, -10, 10), "x**2", None)

    # multiple expressions over the same range
    args = _plot_sympify((x + 1, x ** 2, (x, 0, 5)))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, (x, 0, 5), "x + 1", None)
    assert r[1] == (x ** 2, (x, 0, 5), "x**2", None)

    # multiple expressions over the same range with the same rendering kws
    args = _plot_sympify((x + 1, x ** 2, (x, 0, 5), {0: 0, 1: 1}))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, (x, 0, 5), "x + 1", {0: 0, 1: 1})
    assert r[1] == (x ** 2, (x, 0, 5), "x**2", {0: 0, 1: 1})

    # multiple expressions with different ranges, labels and rendering kws
    args = _plot_sympify([(x + 1, (x, 0, 5)), (x ** 2, (x, -2, 2), "test", {0: 0, 1: 1})])
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, (x, 0, 5), "x + 1", None)
    assert r[1] == (x ** 2, (x, -2, 2), "test", {0: 0, 1: 1})


def test_check_arguments_plot_parametric():
    ### Test arguments for plot_parametric()

    x, y = symbols("x, y")

    # single parametric expression
    args = _plot_sympify((x + 1, x))
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, (x, -10, 10), "x", None)

    # single parametric expression with custom range, label and rendering kws
    args = _plot_sympify((x + 1, x, (x, -2, 2), "test", {0: 0, 1: 1}))
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, (x, -2, 2), "test", {0: 0, 1: 1})

    args = _plot_sympify(((x + 1, x), (x, -2, 2), "test"))
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, (x, -2, 2), "test", None)

    # multiple parametric expressions same symbol
    args = _plot_sympify([(x + 1, x), (x ** 2, x + 1)])
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, (x, -10, 10), "x", None)
    assert r[1] == (x ** 2, x + 1, (x, -10, 10), "x", None)

    # multiple parametric expressions different symbols
    args = _plot_sympify([(x + 1, x), (y ** 2, y + 1)])
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, (x, -10, 10), "x", None)
    assert r[1] == (y ** 2, y + 1, (y, -10, 10), "y", None)

    # multiple parametric expressions same range
    args = _plot_sympify([(x + 1, x), (x ** 2, x + 1), (x, -2, 2)])
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, (x, -2, 2), "x", None)
    assert r[1] == (x ** 2, x + 1, (x, -2, 2), "x", None)

    # multiple parametric expressions, custom ranges and labels
    args = _plot_sympify([(x + 1, x, (x, -2, 2)), (x ** 2, x + 1, (x, -3, 3), "test", {0: 0, 1: 1})])
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, (x, -2, 2), "x", None)
    assert r[1] == (x ** 2, x + 1, (x, -3, 3), "test", {0: 0, 1: 1})


def test_check_arguments_plot3d_parametric_line():
    ### Test arguments for plot3d_parametric_line()

    x, y = symbols("x, y")

    # single parametric expression
    args = _plot_sympify((x + 1, x, sin(x)))
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, sin(x), (x, -10, 10), "x", None)

    # single parametric expression with custom range, label and rendering kws
    args = _plot_sympify((x + 1, x, sin(x), (x, -2, 2), "test", {0: 0, 1: 1}))
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, sin(x), (x, -2, 2), "test", {0: 0, 1: 1})

    args = _plot_sympify(((x + 1, x, sin(x)), (x, -2, 2), "test"))
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, sin(x), (x, -2, 2), "test", None)

    # multiple parametric expression same symbol
    args = _plot_sympify([(x + 1, x, sin(x)), (x ** 2, 1, cos(x), {0: 0, 1: 1})])
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, sin(x), (x, -10, 10), "x", None)
    assert r[1] == (x ** 2, Integer(1), cos(x), (x, -10, 10), "x", {0: 0, 1: 1})

    # multiple parametric expression different symbols
    args = _plot_sympify([(x + 1, x, sin(x)), (y ** 2, 1, cos(y))])
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, sin(x), (x, -10, 10), "x", None)
    assert r[1] == (y ** 2, Integer(1), cos(y), (y, -10, 10), "y", None)

    # multiple parametric expression, custom ranges and labels
    args = _plot_sympify([(x + 1, x, sin(x)), (x ** 2, 1, cos(x), (x, -2, 2), "test", {0: 0, 1: 1})])
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, sin(x), (x, -10, 10), "x", None)
    assert r[1] == (x ** 2, Integer(1), cos(x), (x, -2, 2), "test", {0: 0, 1: 1})


def test_check_arguments_plot3d_plot_contour():
    ### Test arguments for plot3d() and plot_contour()

    x, y = symbols("x, y")

    # single expression
    args = _plot_sympify((x + y,))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 5
    assert r[0][0] == x + y
    assert r[0][1] == (x, -10, 10) or (y, -10, 10)
    assert r[0][2] == (y, -10, 10) or (x, -10, 10)
    assert r[0][1] != r[0][2]
    assert r[0][3] == "x + y"
    assert r[0][4] is None

    # single expression, custom range, label and rendering kws
    args = _plot_sympify((x + y, (x, -2, 2), "test", {0: 0, 1: 1}))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 5
    assert r[0][0] == x + y
    assert r[0][1] == (x, -2, 2) or (y, -10, 10)
    assert r[0][2] == (y, -10, 10) or (x, -2, 2)
    assert r[0][1] != r[0][2]
    assert r[0][3] == "test"
    assert r[0][4] == {0: 0, 1: 1}

    args = _plot_sympify((x + y, (x, -2, 2), (y, -4, 4), "test"))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + y, (x, -2, 2), (y, -4, 4), "test", None)

    # multiple expressions
    args = _plot_sympify((x + y, x * y))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 5
    assert len(r[1]) == 5
    assert r[0][0] == x + y
    assert r[0][1] == (x, -10, 10) or (y, -10, 10)
    assert r[0][2] == (y, -10, 10) or (x, -10, 10)
    assert r[0][1] != r[0][2]
    assert r[0][3] == "x + y"
    assert r[0][4] == None
    assert r[1][0] == x * y
    assert r[1][1] == (x, -10, 10) or (y, -10, 10)
    assert r[1][2] == (y, -10, 10) or (x, -10, 10)
    assert r[1][1] != r[0][2]
    assert r[1][3] == "x*y"
    assert r[1][4] == None

    # multiple expressions, same custom ranges
    args = _plot_sympify((x + y, x * y, (x, -2, 2), (y, -4, 4)))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + y, (x, -2, 2), (y, -4, 4), "x + y", None)
    assert r[1] == (x * y, (x, -2, 2), (y, -4, 4), "x*y", None)

    # multiple expressions, custom ranges, labels and rendering kws
    args = _plot_sympify(
        [(x + y, (x, -2, 2), (y, -4, 4)),
        (x * y, (x, -3, 3), (y, -6, 6), "test", {0: 0, 1: 1})]
    )
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + y, (x, -2, 2), (y, -4, 4), "x + y", None)
    assert r[1] == (x * y, (x, -3, 3), (y, -6, 6), "test", {0: 0, 1: 1})


def test_check_arguments_plot3d_parametric_surface():
    ### Test arguments for plot3d_parametric_surface()

    x, y = symbols("x, y")

    # single parametric expression
    args = _plot_sympify((x + y, cos(x + y), sin(x + y)))
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 7
    assert r[0][0] == x + y
    assert r[0][1] == cos(x + y)
    assert r[0][2] == sin(x + y)
    assert r[0][3] == (x, -10, 10) or (y, -10, 10)
    assert r[0][4] == (y, -10, 10) or (x, -10, 10)
    assert r[0][3] != r[0][4]
    assert r[0][5] == "(x + y, cos(x + y), sin(x + y))"
    assert r[0][6] == None

    # single parametric expression, custom ranges, labels and rendering kws
    args = _plot_sympify(
        (x + y, cos(x + y), sin(x + y), (x, -2, 2), (y, -4, 4), "test", {0: 0})
    )
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + y, cos(x + y), sin(x + y), (x, -2, 2), (y, -4, 4), "test", {0: 0})

    # multiple parametric expressions
    args = _plot_sympify(
        [(x + y, cos(x + y), sin(x + y)), (x - y, cos(x - y), sin(x - y))]
    )
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 7
    assert len(r[1]) == 7
    assert r[0][0] == x + y
    assert r[0][1] == cos(x + y)
    assert r[0][2] == sin(x + y)
    assert r[0][3] == (x, -10, 10) or (y, -10, 10)
    assert r[0][4] == (y, -10, 10) or (x, -10, 10)
    assert r[0][3] != r[0][4]
    assert r[0][5] == "(x + y, cos(x + y), sin(x + y))"
    assert r[0][6] == None
    assert r[1][0] == x - y
    assert r[1][1] == cos(x - y)
    assert r[1][2] == sin(x - y)
    assert r[1][3] == (x, -10, 10) or (y, -10, 10)
    assert r[1][4] == (y, -10, 10) or (x, -10, 10)
    assert r[1][3] != r[0][4]
    assert r[1][5] == "(x - y, cos(x - y), sin(x - y))"
    assert r[1][6] == None

    # multiple parametric expressions, custom ranges and labels
    args = _plot_sympify(
        [
            (x + y, cos(x + y), sin(x + y), (x, -2, 2), "test"),
            (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4), "test2", {0: 0, 1: 1}),
        ]
    )
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 7
    assert r[0][0] == x + y
    assert r[0][1] == cos(x + y)
    assert r[0][2] == sin(x + y)
    assert r[0][3] == (x, -2, 2) or (y, -10, 10)
    assert r[0][4] == (y, -10, 10) or (x, -2, 2)
    assert r[0][3] != r[0][4]
    assert r[0][5] == "test"
    assert r[0][6] == None
    assert r[1] == (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4), "test2", {0: 0, 1: 1})


def test_check_arguments_plot_implicit():
    ### Test arguments for plot_implicit

    x, y = symbols("x, y")

    # single expression with both ranges
    args = _plot_sympify((x > 0, (x, -2, 2), (y, -3, 3)))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 5
    assert r[0] == (x > 0, (x, -2, 2), (y, -3, 3), "x > 0", None)

    # single expression with one missing range
    args = _plot_sympify((x > 0, (x, -2, 2), "test", {0: 0, 1: 1}))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 5
    assert r[0][:2] == (x > 0, (x, -2, 2))
    assert r[0][-2] == "test"
    assert r[0][-1] == {0: 0, 1: 1}
    assert (r[0][2][1] == Integer(-10)) and (r[0][2][2] == Integer(10))

    # multiple expressions
    args = _plot_sympify([(x > 0, (x, -2, 2), (y, -3, 3)), ((x > 0) & (y < 0), "test", {0: 0, 1: 1})])
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 5
    assert r[0] == (x > 0, (x, -2, 2), (y, -3, 3), "x > 0", None)
    assert len(r[1]) == 5
    assert r[1][0] == ((x > 0) & (y < 0))
    assert (r[1][1] == Tuple(x, -10, 10)) or (r[1][1] == Tuple(y, -10, 10))
    assert (r[1][2] == Tuple(x, -10, 10)) or (r[1][2] == Tuple(y, -10, 10))
    assert r[1][-2] == "test"
    assert r[1][-1] == {0: 0, 1: 1}

    # incompatible free symbols between expression and ranges
    z = symbols("z")
    args = _plot_sympify((x * y > 0, (x, -2, 2), (z, -3, 3)))
    raises(ValueError, lambda: _check_arguments(args, 1, 2))
