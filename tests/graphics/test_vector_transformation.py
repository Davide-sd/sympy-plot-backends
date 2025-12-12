# test_vector_transforms.py
import math
import pytest
import sympy
from sympy import Matrix, pi, sin, cos, simplify, N, sqrt

# import your implementation (adjust module name/path if needed)
from spb.graphics.vector_transforms import (
    is_curvilinear,
    LocalTransform,
    express,
)
from sympy.vector import CoordSys3D
from sympy import symbols


@pytest.mark.parametrize("coord_sys, is_curvil",[
    (CoordSys3D("C1"), False),
    (CoordSys3D("S", transformation="spherical"), True),
    (CoordSys3D("C2", transformation="cylindrical"), True),
])
def test_is_curvilinear_detects_spherical(coord_sys, is_curvil):
    assert is_curvilinear(coord_sys) is is_curvil


def test_LocalTransform_instantiation_error():
    C = CoordSys3D("C")
    # error because `to_system` is not an instance of CoordSys3D
    pytest.raises(TypeError, lambda: LocalTransform(C, None))
    # error because `from_system` is not an instance of CoordSys3D
    pytest.raises(TypeError, lambda: LocalTransform(None, C))
    # error because both `to_system` and `from_system` are not an instances
    # of CoordSys3D
    pytest.raises(TypeError, lambda: LocalTransform(None, None))


def test_express_cartesian_to_cartesian():
    C1 = CoordSys3D("C1")
    C2 = C1.locate_new("C2", 2 * C1.i)
    a, b, c = symbols("a:c")

    v = a * C2.i + b * C2.j + c * C2.k
    assert express(v, C1) == a * C1.i + b * C1.j + c * C1.k


def test_express_cartesian_to_cartesian_rotated():
    alpha = symbols("alpha")
    C1 = CoordSys3D("C1")
    C2 = C1.orient_new_axis("C2", alpha, C1.k)
    a, b, c = symbols("a:c")

    v1 = C2.i
    assert express(v1, C1) == cos(alpha) * C1.i + sin(alpha) * C1.j

    v2 = C2.j
    assert express(v2, C1) == -sin(alpha) * C1.i + cos(alpha) * C1.j

    v3 = C2.k
    assert express(v3, C1) == C1.k

    v4 = a * C2.i + b * C2.j + c * C2.k
    assert express(v4, C1) == (
        (a * cos(alpha) - b * sin(alpha)) * C1.i +
        (a * sin(alpha) + b * cos(alpha)) * C1.j +
        c * C1.k
    )


def test_express_from_spherical_to_cartesian():
    C = CoordSys3D("C")
    S = C.create_new("S", transformation="spherical")
    r, theta, phi = S.base_scalars()
    a, b, c = symbols("a:c")

    # vector along the radial direction
    v1 = 1 * S.i
    assert express(v1, C) == (
        (sin(theta) * cos(phi)) * C.i +
        (sin(theta) * sin(phi)) * C.j +
        cos(theta) * C.k
    )
    # vector along the polar direction
    v2 = 1 * S.j
    assert express(v2, C) == (
        (cos(phi) * cos(theta)) * C.i +
        (sin(phi) * cos(theta)) * C.j +
        (-sin(theta)) * C.k
    )
    # vector along the azimuthal direction
    v3 = 1 * S.k
    assert express(v3, C) == (
        -sin(phi) * C.i +
        cos(phi) * C.j
    )
    v4 = a * S.i + b * S.j + c * S.k
    assert express(v4, C) == (
        (a*cos(phi)*sin(theta) + b*cos(phi)*cos(theta) - c*sin(phi)) * C.i +
        (a*sin(phi)*sin(theta) + b*sin(phi)*cos(theta) + c*cos(phi)) * C.j +
        (a*cos(theta) - b*sin(theta)) * C.k
    )


def test_express_from_cartesian_to_spherical():
    C = CoordSys3D("C")
    S = C.create_new("S", transformation="spherical")
    r, theta, phi = S.base_scalars()
    a, b, c = symbols("a:c")

    # vector along the x-axis
    v1 = 1 * C.i
    assert express(v1, S) == (
        (sin(theta) * cos(phi)) * S.i +
        (cos(theta) * cos(phi)) * S.j +
        (-sin(phi)) * S.k
    )
    # vector along the y-axis
    v2 = 1 * C.j
    assert express(v2, S) == (
        (sin(theta) * sin(phi)) * S.i +
        (cos(theta) * sin(phi)) * S.j +
        (cos(phi)) * S.k
    )
    # vector along the z-axis
    v3 = 1 * C.k
    assert express(v3, S) == (
        cos(theta) * S.i +
        (-sin(theta)) * S.j
    )
    # vector in an arbitrary direction
    v4 = a * C.i + b * C.j + c * C.k
    assert express(v4, S) == (
        (a*cos(phi)*sin(theta) + b*sin(phi)*sin(theta) + c*cos(theta)) * S.i +
        (a*cos(phi)*cos(theta) + b*sin(phi)*cos(theta) - c*sin(theta)) * S.j +
        (-a*sin(phi) + b*cos(phi)) * S.k
    )


def test_express_from_cylindrical_to_cartesian():
    Cart = CoordSys3D("Cart")
    Cyl = Cart.create_new("Cyl", transformation="cylindrical")
    r, theta, zeta = Cyl.base_scalars()
    a, b, c = symbols("a:c")

    # vector along the radial direction
    v1 = 1 * Cyl.i
    assert express(v1, Cart) == cos(theta) * Cart.i + sin(theta) * Cart.j
    # vector along the theta direction
    v2 = 1 * Cyl.j
    assert express(v2, Cart) == -sin(theta) * Cart.i + cos(theta) * Cart.j
    # vector along the z direction
    v3 = 1 * Cyl.k
    assert express(v3, Cart) == Cart.k
    # vector in an arbitrary direction
    v4 = a * Cyl.i + b * Cyl.j + c * Cyl.k
    assert express(v4, Cart) == (
        (a*cos(theta) - b*sin(theta)) * Cart.i +
        (a*sin(theta) + b*cos(theta)) * Cart.j +
        c * Cart.k
    )


def test_express_from_cartesian_to_cylindrical():
    Cart = CoordSys3D("Cart")
    Cyl = Cart.create_new("Cyl", transformation="cylindrical")
    r, theta, zeta = Cyl.base_scalars()
    a, b, c = symbols("a:c")

    # vector along the x-axis
    v1 = 1 * Cart.i
    assert express(v1, Cyl) == cos(theta) * Cyl.i - sin(theta) * Cyl.j
    # vector along the y-axis
    v2 = 1 * Cart.j
    assert express(v2, Cyl) == sin(theta) * Cyl.i + cos(theta) * Cyl.j
    # vector along the z-axis
    v3 = 1 * Cart.k
    assert express(v3, Cyl) == Cyl.k
    # vector in an arbitrary direction
    v4 = a * Cart.i + b * Cart.j + c * Cart.k
    assert express(v4, Cyl) == (
        (a*cos(theta) + b*sin(theta)) * Cyl.i +
        (-a*sin(theta) + b*cos(theta)) * Cyl.j +
        c * Cyl.k
    )


def test_expr_from_spherical_to_cylindrical():
    Cart = CoordSys3D("Cart")
    S = Cart.create_new("S", transformation="spherical")
    C = Cart.create_new("C", transformation="cylindrical")
    r_s, theta_s, phi_s = S.base_scalars()
    r_c, theta_c, z_c = C.base_scalars()
    a, b, c = symbols("a, b, c")
    # Note that phi_s = theta_c (azimuthal angle, [0, 2*pi[)

    # vector along the radial direction
    v1 = S.i
    r1 = express(v1, C)
    assert r1.equals(
        (sin(theta_s) * cos(theta_c - phi_s)) * C.i +
        (-sin(theta_s) * sin(theta_c - phi_s)) * C.j +
        (cos(theta_s)) * C.k
    )
    assert r1.subs(phi_s, theta_c).equals(
        sin(theta_s) * C.i + cos(theta_s) * C.k)

    # vector along the polar direction
    v2 = S.j
    r2 = express(v2, C)
    assert r2.equals(
        (cos(theta_s) * cos(theta_c - phi_s)) * C.i +
        (-cos(theta_s) * sin(theta_c - phi_s)) * C.j +
        (-sin(theta_s)) * C.k
    )
    assert r2.subs(phi_s, theta_c).equals(
        cos(theta_s) * C.i - sin(theta_s) * C.k)

    # vector along the azimuthal direction
    v3 = S.k
    r3 = express(v3, C)
    assert r3.equals(
        (sin(theta_c - phi_s)) * C.i +
        (cos(theta_c - phi_s)) * C.j
    )
    assert r3.subs(phi_s, theta_c).equals(C.j)

    # vector along arbitrary direction
    v4 = a * S.i + b * S.j + c * S.k
    r4 = express(v4, C)
    assert r4.equals(
        (
            (
                a*sin(phi_s)*sin(theta_s) + b*sin(phi_s)*cos(theta_s) +
                c*cos(phi_s)
            ) * sin(theta_c) +
            (a*sin(theta_s)*cos(phi_s) + b*cos(phi_s)*cos(theta_s) -
            c*sin(phi_s))*cos(theta_c)
        )*C.i +
        (
            (
                a*sin(phi_s)*sin(theta_s) + b*sin(phi_s)*cos(theta_s) +
                c*cos(phi_s)
            )*cos(theta_c) -
            (
                a*sin(theta_s)*cos(phi_s) + b*cos(phi_s)*cos(theta_s) -
                c*sin(phi_s)
            )*sin(theta_c)
        )*C.j +
        (a*cos(theta_s) - b*sin(theta_s))*C.k
    )
    assert r4.subs(phi_s, theta_c).equals(
        (a * sin(theta_s) + b * cos(theta_s)) * C.i +
        c * C.j +
        (a * cos(theta_s) - b * sin(theta_s)) * C.k
    )


def test_expr_from_cylindrical_to_spherical():
    Cart = CoordSys3D("Cart")
    S = Cart.create_new("S", transformation="spherical")
    C = Cart.create_new("C", transformation="cylindrical")
    r_s, theta_s, phi_s = S.base_scalars()
    r_c, theta_c, z_c = C.base_scalars()
    a, b, c = symbols("a, b, c")
    # Note that phi_s = theta_c (azimuthal angle, [0, 2*pi[)

    # vector along the radial direction
    v1 = C.i
    r1 = express(v1, S)
    assert r1.equals(
        (sin(theta_s) * cos(theta_c - phi_s)) * S.i +
        (cos(theta_s) * cos(theta_c - phi_s)) * S.j +
        (sin(theta_c - phi_s)) * C.k
    )
    assert r1.subs(phi_s, theta_c).equals(
        sin(theta_s) * S.i + cos(theta_s) * S.j)

    # vector along the azimuthal direction
    v2 = C.j
    r2 = express(v2, S)
    assert r2.equals(
        (-sin(theta_s) * sin(theta_c - phi_s)) * S.i +
        (-cos(theta_s) * sin(theta_c - phi_s)) * S.j +
        (cos(theta_c - phi_s)) * C.k
    )
    assert r2.subs(phi_s, theta_c).equals(C.k)

    # vector along the z-axis
    v3 = C.k
    r3 = express(v3, S)
    assert r3 == cos(theta_s) * S.i - sin(theta_s) * S.j
    assert r3.subs(phi_s, theta_c) == cos(theta_s) * S.i - sin(theta_s) * S.j

    # vector along arbitrary direction
    v4 = a * C.i + b * C.j + c * C.k
    r4 = express(v4, S)
    assert r4.equals(
        (
            c*cos(theta_s) +
            (a*sin(theta_c) + b*cos(theta_c))*sin(phi_s)*sin(theta_s) +
            (a*cos(theta_c) - b*sin(theta_c))*sin(theta_s)*cos(phi_s)
        )*S.i +
        (
            -c*sin(theta_s) +
            (a*sin(theta_c) + b*cos(theta_c))*sin(phi_s)*cos(theta_s) +
            (a*cos(theta_c) - b*sin(theta_c))*cos(phi_s)*cos(theta_s)
        )*S.j +
        (
            (a*sin(theta_c) + b*cos(theta_c))*cos(phi_s) - (a*cos(theta_c) -
            b*sin(theta_c))*sin(phi_s)
        )*S.k
    )
    assert r4.subs(phi_s, theta_c).equals(
        (a * sin(theta_s) + c * cos(theta_s)) * S.i +
        (a * cos(theta_s) - c * sin(theta_s)) * S.j +
        b * S.k
    )


def test_express_path_of_systems():
    # C1 -> C2 (translate) -> C3 (rotate about k by alpha) -> S (spherical)
    p, alpha = symbols("p, alpha")
    C1 = CoordSys3D("C1")
    C2 = C1.orient_new_axis("C2", alpha, C1.k)    # rotation about z by alpha
    C3 = C2.locate_new("C3", p * C2.i)            # translation of origin
    S = C3.create_new("S", transformation="spherical")
    r, theta, phi = S.base_scalars()

    # vector along the radial direction
    v1 = S.i
    assert express(v1, C1) == (
        (-sin(phi)*sin(theta)*sin(alpha) + sin(theta)*cos(phi)*cos(alpha)) * C1.i +
        (sin(phi)*sin(theta)*cos(alpha) + sin(theta)*sin(alpha)*cos(phi)) * C1.j +
        cos(theta) * C1.k
    )
    assert express(v1, C3) == (
        (sin(theta) * cos(phi)) * C3.i +
        (sin(theta) * sin(phi)) * C3.j +
        cos(theta) * C3.k
    )

    # vector along the polar direction
    v2 = 1 * S.j
    assert express(v2, C1) == (
        (-sin(phi)*sin(alpha)*cos(theta) + cos(phi)*cos(theta)*cos(alpha)) * C1.i +
        (sin(phi)*cos(theta)*cos(alpha) + sin(alpha)*cos(phi)*cos(theta)) * C1.j +
        (-sin(theta)) * C1.k
    )
    assert express(v2, C3) == (
        (cos(phi) * cos(theta)) * C3.i +
        (sin(phi) * cos(theta)) * C3.j +
        (-sin(theta)) * C3.k
    )

    # vector along the azimuthal direction
    v3 = 1 * S.k
    assert express(v3, C1) == (
        (-sin(phi) * cos(alpha) - sin(alpha) * cos(phi)) * C1.i +
        (-sin(phi) * sin(alpha) + cos(phi) * cos(alpha)) * C1.j
    )
    assert express(v3, C3) == (
        -sin(phi) * C3.i +
        cos(phi) * C3.j
    )
