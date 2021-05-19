from sympy import symbols, Matrix, Tuple
from sympy.vector import CoordSys3D

from spb.vectors import _preprocess,_build_series, vector_plot
from spb.utils import _plot_sympify, _split_vector
from spb.series import Vector2DSeries, Vector3DSeries

from pytest import raises

def pw(*args):
    """ _preprocess wrapper. Only need it to sympify the arguments before
    calling _preprocess. """
    args = _plot_sympify(args)
    return _preprocess(*args)

def test_preprocess():
    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j + z * N.k
    v2 = z * N.i + x * N.j + y * N.k
    m1 = v1.to_matrix(N)
    m2 = v2.to_matrix(N)
    l1 = list(m1)
    l2 = list(m2)

    # passing in vectors
    r = pw(v1)[0]
    assert r[0] == v1
    assert r[1] == str(v1)

    r = pw(v1, (x, -5, 5), "test")[0]
    assert r == [v1, Tuple(x, -5, 5), "test"]

    r = pw((v1, (x, -5, 5), "test"))[0]
    assert r == [v1, Tuple(x, -5, 5), "test"]

    r = pw((v1, (x, -5, 5), "v1"), (v2, (x, -5, 5), (y, -2, 2)))
    assert r[0] == [v1, Tuple(x, -5, 5), "v1"]
    assert r[1] == [v2, Tuple(x, -5, 5), Tuple(y, -2, 2), str(v2)]

    r = pw(v1, v2, (x, -5, 5), (y, -2, 2), (z, -3, 3))
    assert r[0] == [v1, Tuple(x, -5, 5), Tuple(y, -2, 2),
            Tuple(z, -3, 3), str(v1)]
    assert r[1] == [v2, Tuple(x, -5, 5), Tuple(y, -2, 2),
            Tuple(z, -3, 3), str(v2)]
    
    # passing in matrices
    r = pw(m1, (x, -5, 5), "test")[0]
    assert r == [m1, Tuple(x, -5, 5), "test"]

    r = pw(m1, m2, (x, -5, 5), (y, -2, 2), (z, -3, 3))
    assert r[0] == [m1, Tuple(x, -5, 5), Tuple(y, -2, 2),
            Tuple(z, -3, 3), str(m1)]
    assert r[1] == [m2, Tuple(x, -5, 5), Tuple(y, -2, 2),
            Tuple(z, -3, 3), str(m2)]
    
    # passing in lists
    r = pw(l1, (x, -5, 5), "test")[0]
    assert r == [tuple(l1), Tuple(x, -5, 5), "test"]

    r = pw(l1, l2, (x, -5, 5), (y, -2, 2), (z, -3, 3))
    assert r[0] == [tuple(l1), Tuple(x, -5, 5), Tuple(y, -2, 2),
            Tuple(z, -3, 3), str(tuple(l1))]
    assert r[1] == [tuple(l2), Tuple(x, -5, 5), Tuple(y, -2, 2),
            Tuple(z, -3, 3), str(tuple(l2))]


def test_split_vector():
    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j + z * N.k
    v2 = z * N.i + x * N.j + y * N.k
    m1 = v1.to_matrix(N)
    m2 = v2.to_matrix(N)
    l1 = list(m1)
    l2 = list(m2)

    ranges = [Tuple(x, -5, 5)]
    assert _split_vector(v1, ranges) == ((x, y, z), ranges)
    assert _split_vector(m1, ranges) == ((x, y, z), ranges)
    assert _split_vector(l1, ranges) == ((x, y, z), ranges)
    assert _split_vector(v2, ranges) == ((z, x, y), ranges)
    assert _split_vector(m2, ranges) == ((z, x, y), ranges)
    assert _split_vector(l2, ranges) == ((z, x, y), ranges)

    # too few or too many elements
    raises(ValueError, lambda: _split_vector([x], ranges))
    raises(ValueError, lambda: _split_vector([x, x, x, x], ranges))

def test_build_series():
    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j
    v2 = z * N.i + x * N.j + y * N.k
    m1 = v1.to_matrix(N)
    m2 = v2.to_matrix(N)

    # Tests for 2D vectors
    args = pw(v1, "test")[0]
    _, _, s = _build_series(args[0], *args[1:-1], label=args[-1])
    assert isinstance(s, Vector2DSeries)
    # auto generate ranges
    t1 = (s.u.var_x, s.u.start_x, s.u.end_x)
    t2 = (s.u.var_y, s.u.start_y, s.u.end_y)
    assert (t1 == (x, -10.0, 10.0)) or (t1 == (y, -10.0, 10.0))
    assert (t2 == (x, -10.0, 10.0)) or (t2 == (y, -10.0, 10.0))

    args = pw(v1, (x, -5, 5), "test")[0]
    _, _, s = _build_series(args[0], *args[1:-1], label=args[-1])
    assert isinstance(s, Vector2DSeries)
    assert (s.u.var_x, s.u.start_x, s.u.end_x) == (x, -5.0, 5.0)
    # auto generate range
    assert (s.u.var_y, s.u.start_y, s.u.end_y) == (y, -10.0, 10.0)

    # vector doesn't contain free symbols, and not all ranges were provided.
    # raise error because the missing range could be any symbol.
    args = pw([1, 2], (x, -5, 5), "test")[0]
    raises(ValueError, 
        lambda: _build_series(args[0], *args[1:-1], label=args[-1]))
    
    # too many free symbols in the 2D vector
    args = pw([x + y, z], (x, -5, 5), "test")[0]
    raises(ValueError, 
        lambda: _build_series(args[0], *args[1:-1], label=args[-1]))
    

    # Tests for 3D vectors
    args = pw(v2, "test")[0]
    _, _, s = _build_series(args[0], *args[1:-1], label=args[-1])
    assert isinstance(s, Vector3DSeries)
    # auto generate ranges
    t1 = (s.var_x, s.start_x, s.end_x)
    t2 = (s.var_y, s.start_y, s.end_y)
    t3 = (s.var_z, s.start_z, s.end_z)
    assert ( (t1 == (x, -10.0, 10.0)) or (t1 == (y, -10.0, 10.0)) or
        (t1 == (z, -10.0, 10.0)) )
    assert ( (t2 == (x, -10.0, 10.0)) or (t2 == (y, -10.0, 10.0)) or
        (t2 == (z, -10.0, 10.0)) )
    assert ( (t3 == (x, -10.0, 10.0)) or (t3 == (y, -10.0, 10.0)) or
        (t3 == (z, -10.0, 10.0)) )

    args = pw(v2, (x, -5, 5), "test")[0]
    _, _, s = _build_series(args[0], *args[1:-1], label=args[-1])
    assert isinstance(s, Vector3DSeries)
    t1 = (s.var_x, s.start_x, s.end_x)
    t2 = (s.var_y, s.start_y, s.end_y)
    t3 = (s.var_z, s.start_z, s.end_z)
    assert t1 == (x, -5.0, 5.0)
    assert (t2 == (y, -10.0, 10.0)) or (t2 == (z, -10.0, 10.0))
    assert (t3 == (y, -10.0, 10.0)) or (t3 == (z, -10.0, 10.0))

    # vector doesn't contain free symbols, and not all ranges were provided.
    # raise error because the missing range could be any symbol.
    args = pw(Matrix([1, 2, 3]), (x, -5, 5), "test")[0]
    raises(ValueError, 
        lambda: _build_series(args[0], *args[1:-1], label=args[-1]))
    
    # too many free symbols in the 3D vector
    a = symbols("a")
    args = pw(Matrix([x + y, z, a + x]), (x, -5, 5), "test")[0]
    raises(ValueError, 
        lambda: _build_series(args[0], *args[1:-1], label=args[-1]))

def test_vector_plot():
    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j
    v2 = z * N.i + x * N.j + y * N.k

    # this will stop inside vector_plot, because we are mixing 2D and 3D vectors
    raises(ValueError, lambda: vector_plot(v1, v2))
    # this will stop inside _build_series, because 3 ranges have been provided
    # for a 2D vector plot (v1)
    raises(ValueError, lambda: vector_plot(v1, v2, (x, -5, 5), (y, -2, 2), 
        (z, -3, 3)))
    
    # scalar is not one of [None,True,False,Expr]
    raises(ValueError, lambda: vector_plot(v1, scalar="s"))

def test_vector_data():
    x, y, z = symbols("x:z")

    s = Vector2DSeries(x, y, (x, -5, 5), (y, -3, 3), "test", n1=10, n2=15)
    xx, yy, uu, vv = s.get_data()
    assert xx.shape == (15, 10)
    assert yy.shape == (15, 10)
    assert uu.shape == xx.shape
    assert vv.shape == yy.shape

    # at least one vector component is a scalar
    s = Vector2DSeries(1, y, (x, -5, 5), (y, -3, 3), "test", n1=10, n2=15)
    xx, yy, uu, vv = s.get_data()
    assert xx.shape == (15, 10)
    assert yy.shape == (15, 10)
    assert uu.shape == xx.shape
    assert vv.shape == yy.shape

    s = Vector3DSeries(x, y, z, (x, -5, 5), (y, -3, 3), (z, -2, 2), "test",
        n1=10, n2=15, n3=20)
    xx, yy, zz, uu, vv, ww = s.get_data()
    assert xx.shape == (15, 10, 20)
    assert yy.shape == (15, 10, 20)
    assert zz.shape == (15, 10, 20)
    assert uu.shape == xx.shape
    assert vv.shape == yy.shape
    assert ww.shape == zz.shape

    # at least one vector component is a scalar
    s = Vector3DSeries(x, 1, z, (x, -5, 5), (y, -3, 3), (z, -2, 2), "test",
        n1=10, n2=15, n3=20)
    xx, yy, zz, uu, vv, ww = s.get_data()
    assert xx.shape == (15, 10, 20)
    assert yy.shape == (15, 10, 20)
    assert zz.shape == (15, 10, 20)
    assert uu.shape == xx.shape
    assert vv.shape == yy.shape
    assert ww.shape == zz.shape
