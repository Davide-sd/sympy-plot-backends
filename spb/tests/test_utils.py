from sympy import Expr, Integer, Tuple, symbols, sin, cos
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

    # string should not be sympified
    args = (x + y, (x, 0, 1), "str", 1)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 4
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[2], str)
    assert isinstance(r[3], Integer)

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
    r = _create_ranges({x, y}, [(x, 0, 1), ], 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, 0, 1) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, 0, 1)
    assert r[0] != r[1]

    # too many free symbols
    raises(ValueError, lambda: _create_ranges({x, y}, [], 1))
    raises(ValueError,
        lambda: _create_ranges({x, y}, [(x, 0, 5), (y, 0, 1)], 1))

def test_check_arguments():
    x, y = symbols("x, y")

    ### Test arguments for plot()

    # single expressions
    args = _plot_sympify((x + 1, ))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, (x, -10, 10), "x + 1")

    # single expressions with range
    args = _plot_sympify((x + 1, (x, -2, 2)))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, (x, -2, 2), "x + 1")

    # single expressions with range and label
    args = _plot_sympify((x + 1, (x, -2, 2), "test"))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, (x, -2, 2), "test")

    # multiple expressions
    args = _plot_sympify((x + 1, x**2))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, (x, -10, 10), "x + 1")
    assert r[1] == (x**2, (x, -10, 10), "x**2")

    # multiple expressions over the same range
    args = _plot_sympify((x + 1, x**2, (x, 0, 5)))
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, (x, 0, 5), "x + 1")
    assert r[1] == (x**2, (x, 0, 5), "x**2")

    # multiple expressions with different ranges and labels
    args = _plot_sympify([(x + 1, (x, 0, 5)), (x**2, (x, -2, 2), "test")])
    r = _check_arguments(args, 1, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, (x, 0, 5), "x + 1")
    assert r[1] == (x**2, (x, -2, 2), "test")


    ### Test arguments for plot_parametric()

    # single parametric expression
    args = _plot_sympify((x + 1, x))
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, (x, -10, 10), "(x + 1, x)")

    # single parametric expression with custom range and label
    args = _plot_sympify((x + 1, x, (x, -2, 2), "test"))
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, (x, -2, 2), "test")

    args = _plot_sympify(((x + 1, x), (x, -2, 2), "test"))
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, (x, -2, 2), "test")

    # multiple parametric expressions
    args = _plot_sympify([(x + 1, x), (x**2, x + 1)])
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, (x, -10, 10), "(x + 1, x)")
    assert r[1] == (x**2, x + 1, (x, -10, 10), "(x**2, x + 1)")

    # multiple parametric expressions same range
    args = _plot_sympify([(x + 1, x), (x**2, x + 1), (x, -2, 2)])
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, (x, -2, 2), "(x + 1, x)")
    assert r[1] == (x**2, x + 1, (x, -2, 2), "(x**2, x + 1)")

    # multiple parametric expressions, custom ranges and labels
    args = _plot_sympify([(x + 1, x, (x, -2, 2)),
        (x**2, x + 1, (x, -3, 3), "test")])
    r = _check_arguments(args, 2, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, (x, -2, 2), "(x + 1, x)")
    assert r[1] == (x**2, x + 1, (x, -3, 3), "test")


    ### Test arguments for plot3d_parametric_line()

    # single parametric expression
    args = _plot_sympify((x + 1, x, sin(x)))
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, sin(x), (x, -10, 10), "(x + 1, x, sin(x))")

    # single parametric expression with custom range and label
    args = _plot_sympify((x + 1, x, sin(x), (x, -2, 2), "test"))
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, sin(x), (x, -2, 2), "test")

    args = _plot_sympify(((x + 1, x, sin(x)), (x, -2, 2), "test"))
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + 1, x, sin(x), (x, -2, 2), "test")

    # multiple parametric expression
    args = _plot_sympify([(x + 1, x, sin(x)), (x**2, 1, cos(x))])
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, sin(x), (x, -10, 10), "(x + 1, x, sin(x))")
    assert r[1] == (x**2, Integer(1), cos(x), (x, -10, 10), "(x**2, 1, cos(x))")

    # multiple parametric expression, custom ranges and labels
    args = _plot_sympify([(x + 1, x, sin(x)),
            (x**2, 1, cos(x), (x, -2, 2), "test")])
    r = _check_arguments(args, 3, 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + 1, x, sin(x), (x, -10, 10), "(x + 1, x, sin(x))")
    assert r[1] == (x**2, Integer(1), cos(x), (x, -2, 2), "test")


    ### Test arguments for plot3d() and plot_contour()

    # single expression
    args = _plot_sympify((x + y,))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 4
    assert r[0][0] == x + y
    assert r[0][1] == (x, -10, 10) or (y, -10, 10)
    assert r[0][2] == (y, -10, 10) or (x, -10, 10)
    assert r[0][1] != r[0][2]
    assert r[0][3] == "x + y"

    # single expression, custom range and label
    args = _plot_sympify((x + y, (x, -2, 2), "test"))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 4
    assert r[0][0] == x + y
    assert r[0][1] == (x, -2, 2) or (y, -10, 10)
    assert r[0][2] == (y, -10, 10) or (x, -2, 2)
    assert r[0][1] != r[0][2]
    assert r[0][3] == "test"

    args = _plot_sympify((x + y, (x, -2, 2), (y, -4, 4), "test"))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + y, (x, -2, 2), (y, -4, 4), "test")

    # multiple expressions
    args = _plot_sympify((x + y, x * y))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 4
    assert len(r[1]) == 4
    assert r[0][0] == x + y
    assert r[0][1] == (x, -10, 10) or (y, -10, 10)
    assert r[0][2] == (y, -10, 10) or (x, -10, 10)
    assert r[0][1] != r[0][2]
    assert r[0][3] == "x + y"
    assert r[1][0] == x * y
    assert r[1][1] == (x, -10, 10) or (y, -10, 10)
    assert r[1][2] == (y, -10, 10) or (x, -10, 10)
    assert r[1][1] != r[0][2]
    assert r[1][3] == "x*y"

    # multiple expressions, same custom ranges
    args = _plot_sympify((x + y, x * y, (x, -2, 2), (y, -4, 4)))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + y, (x, -2, 2), (y, -4, 4), "x + y")
    assert r[1] == (x * y, (x, -2, 2), (y, -4, 4), "x*y")

    # multiple expressions, custom ranges and labels
    args = _plot_sympify([(x + y, (x, -2, 2), (y, -4, 4)),
                (x * y, (x, -3, 3), (y, -6, 6), "test")])
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert r[0] == (x + y, (x, -2, 2), (y, -4, 4), "x + y")
    assert r[1] == (x * y, (x, -3, 3), (y, -6, 6), "test")


    ### Test arguments for plot3d_parametric_surface()

    # single parametric expression
    args = _plot_sympify((x + y, cos(x + y), sin(x + y)))
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 6
    assert r[0][0] == x + y
    assert r[0][1] == cos(x + y)
    assert r[0][2] == sin(x + y)
    assert r[0][3] == (x, -10, 10) or (y, -10, 10)
    assert r[0][4] == (y, -10, 10) or (x, -10, 10)
    assert r[0][3] != r[0][4]
    assert r[0][5] == "(x + y, cos(x + y), sin(x + y))"

    # single parametric expression, custom ranges and labels
    args = _plot_sympify((x + y, cos(x + y), sin(x + y), (x, -2, 2),
                (y, -4, 4), "test"))
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert r[0] == (x + y, cos(x + y), sin(x + y), (x, -2, 2), (y, -4, 4),
                "test")

    # multiple parametric expressions
    args = _plot_sympify([(x + y, cos(x + y), sin(x + y)),
                (x - y, cos(x - y), sin(x - y))])
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 6
    assert len(r[1]) == 6
    assert r[0][0] == x + y
    assert r[0][1] == cos(x + y)
    assert r[0][2] == sin(x + y)
    assert r[0][3] == (x, -10, 10) or (y, -10, 10)
    assert r[0][4] == (y, -10, 10) or (x, -10, 10)
    assert r[0][3] != r[0][4]
    assert r[0][5] == "(x + y, cos(x + y), sin(x + y))"
    assert r[1][0] == x - y
    assert r[1][1] == cos(x - y)
    assert r[1][2] == sin(x - y)
    assert r[1][3] == (x, -10, 10) or (y, -10, 10)
    assert r[1][4] == (y, -10, 10) or (x, -10, 10)
    assert r[1][3] != r[0][4]
    assert r[1][5] == "(x - y, cos(x - y), sin(x - y))"

    # multiple parametric expressions, custom ranges and labels
    args = _plot_sympify([(x + y, cos(x + y), sin(x + y), (x, -2, 2), "test"),
            (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4), "test2")])
    r = _check_arguments(args, 3, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 6
    assert r[0][0] == x + y
    assert r[0][1] == cos(x + y)
    assert r[0][2] == sin(x + y)
    assert r[0][3] == (x, -2, 2) or (y, -10, 10)
    assert r[0][4] == (y, -10, 10) or (x, -2, 2)
    assert r[0][3] != r[0][4]
    assert r[0][5] == "test"
    assert r[1] == (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4),
                "test2")
    

    ### Test arguments for plot_implicit

    # single expression with both ranges
    args = _plot_sympify((x > 0, (x, -2, 2), (y, -3, 3)))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 4
    assert r[0] == (x > 0, (x, -2, 2), (y, -3, 3), "x > 0")

    # single expression with one missing range
    args = _plot_sympify((x > 0, (x, -2, 2), "test"))
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert len(r[0]) == 4
    assert r[0][:2] == (x > 0, (x, -2, 2))
    assert r[0][-1] == "test"
    assert (r[0][2][1] == Integer(-10)) and (r[0][2][2] == Integer(10))

    # multiple expressions
    args = _plot_sympify([(x > 0, (x, -2, 2), (y, -3, 3)), 
        ((x > 0) & (y < 0), "test")])
    r = _check_arguments(args, 1, 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert len(r[0]) == 4
    assert r[0] == (x > 0, (x, -2, 2), (y, -3, 3), "x > 0")
    assert len(r[1]) == 4
    assert r[1][0] == ((x > 0) & (y < 0))
    assert (r[1][1] == Tuple(x, -10, 10)) or (r[1][1] == Tuple(y, -10, 10))
    assert (r[1][2] == Tuple(x, -10, 10)) or (r[1][2] == Tuple(y, -10, 10))
    assert r[1][-1] == "test"

    # incompatible free symbols between expression and ranges
    z = symbols("z")
    args = _plot_sympify((x * y > 0, (x, -2, 2), (z, -3, 3)))
    raises(ValueError, lambda: _check_arguments(args, 1, 2))