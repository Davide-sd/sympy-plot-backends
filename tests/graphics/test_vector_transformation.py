# test_vector_transforms.py
import math
import pytest
import sympy
from sympy import Matrix, pi, sin, cos, simplify, N, sqrt

# import your implementation (adjust module name/path if needed)
from spb.graphics.vector_transforms import (
    is_curvilinear,
    LocalTransform,
    VectorTransformationChain,
    get_parent,
)

from sympy.vector import CoordSys3D
from sympy import symbols


# # -------------------------
# # small helpers for tests
# # -------------------------
# def mat_equal_sym(a: Matrix, b: Matrix) -> bool:
#     """Symbolic matrix equality (entrywise simplified to 0)."""
#     a = Matrix(a)
#     b = Matrix(b)
#     diff = simplify(a - b)
#     # all entries must simplify to 0
#     return all(simplify(diff[i, 0]) == 0 for i in range(diff.rows * diff.cols))


# def mat_allclose_numeric(a: Matrix, b: Matrix, tol=1e-12) -> bool:
#     """Numeric approximate equality for 3x1 matrices."""
#     a_n = [float(N(ai)) for ai in a]
#     b_n = [float(N(bi)) for bi in b]
#     return all(abs(x - y) <= tol for x, y in zip(a_n, b_n))


# -------------------------
# fixtures
# -------------------------


@pytest.fixture(scope="module")
def example_1():
    C = CoordSys3D("C")
    return dict(C=C)


@pytest.fixture(scope="module")
def example_2():
    C1 = CoordSys3D("C1")
    C2 = C1.locate_new("C2", 2 * C1.i)
    return dict(C1=C1, C2=C2)


@pytest.fixture(scope="module")
def example_3():
    C = CoordSys3D("C")
    S = C.create_new("S", transformation="spherical")
    r, th, ph = S.base_scalars()
    return dict(C=C, S=S, r=r, th=th, ph=ph)


@pytest.fixture(scope="module")
def example_4():
    # C1 -> C2 (translate) -> C3 (rotate about k by alpha) -> S (spherical)
    p, alpha = symbols("p, alpha")
    C1 = CoordSys3D("C1")
    # C2 = C1.locate_new("C2", p * C1.i)            # translation of origin
    # C3 = C2.orient_new_axis("C3", alpha, C2.k)    # rotation about z by alpha
    C2 = C1.orient_new_axis("C2", alpha, C1.k)    # rotation about z by alpha
    C3 = C2.locate_new("C3", p * C2.i)            # translation of origin
    S = C3.create_new("S", transformation="spherical")
    r, th, ph = S.base_scalars()
    return dict(C1=C1, C2=C2, C3=C3, S=S, r=r, th=th, ph=ph, alpha=alpha)


# -------------------------
# tests
# -------------------------


@pytest.mark.parametrize("coord_sys, is_curvil",[
    (CoordSys3D("C1"), False),
    (CoordSys3D("S", transformation="spherical"), True),
    (CoordSys3D("C2", transformation="cylindrical"), True),
])
def test_is_curvilinear_detects_spherical(coord_sys, is_curvil):
    assert is_curvilinear(coord_sys) is is_curvil


def test_LocalTransform_instantiation_error(example_1):
    C = example_1["C"]
    # error because parent is not an instance of CoordSys3D
    pytest.raises(TypeError, lambda: LocalTransform(child=C, parent=None))
    # error because child is not an instance of CoordSys3D
    pytest.raises(TypeError, lambda: LocalTransform(child=None, parent=C))
    # error because both parent and child are not an instances of CoordSys3D
    pytest.raises(TypeError, lambda: LocalTransform(child=None, parent=None))


def test_LocalTransform(example_2, example_3):
    C1 = example_2["C1"]
    C2 = example_2["C2"]
    lt = LocalTransform(C1, C2)
    g = lt._compute_covariant_basis()
    assert isinstance(g, list) and len(g) == 3
    assert g[0] == Matrix([1, 0, 0])
    assert g[1] == Matrix([0, 1, 0])
    assert g[2] == Matrix([0, 0, 1])
    h = lt._compute_scale_factors()
    assert h == [1, 1, 1]
    e_units = lt._compute_orthonormal_basis()
    assert isinstance(e_units, list) and len(e_units) == 3
    assert e_units[0] == Matrix([1, 0, 0])
    assert e_units[1] == Matrix([0, 1, 0])
    assert e_units[2] == Matrix([0, 0, 1])

    C = example_3["C"]
    S = example_3["S"]
    r = example_3["r"]
    theta = example_3["th"]
    phi = example_3["ph"]
    lt = LocalTransform(S, C)
    g = lt._compute_covariant_basis()
    assert isinstance(g, list) and len(g) == 3
    assert g[0] == Matrix([
        cos(phi) * sin(theta),
        sin(phi) * sin(theta),
        cos(theta)
    ])
    assert g[1] == Matrix([
        r * cos(phi) * cos(theta),
        r * sin(phi) * cos(theta),
        -r * sin(theta)
    ])
    assert g[2] == Matrix([
        -r * sin(phi) * sin(theta),
        r * cos(phi) * sin(theta),
        0
    ])
    h = lt._compute_scale_factors()
    assert h == [1, sqrt(r**2), sqrt(r**2 * sin(theta)**2)]
    e_units = lt._compute_orthonormal_basis()
    assert isinstance(e_units, list) and len(e_units) == 3
    assert e_units[0] == Matrix([
        cos(phi) * sin(theta),
        sin(phi) * sin(theta),
        cos(theta)
    ])
    assert e_units[1] == Matrix([
        r * cos(phi) * cos(theta) / sqrt(r**2),
        r * sin(phi) * cos(theta) / sqrt(r**2),
        -r * sin(theta) / sqrt(r**2)
    ])
    assert e_units[2] == Matrix([
        -r * sin(phi) * sin(theta) / sqrt(r**2 * sin(theta)**2),
        r * cos(phi) * sin(theta) / sqrt(r**2 * sin(theta)**2),
        0
    ])


def test_vector_transformation_instantion_errors():
    # wrong type
    pytest.raises(TypeError, lambda: VectorTransformationChain(1))
    pytest.raises(TypeError, lambda: VectorTransformationChain(None))


def test_vector_transformation_chain_length(
    example_1, example_2, example_3, example_4
):
    C = example_1["C"]
    vt = VectorTransformationChain(C)
    # no chain because C is the only reference system (unconnected)
    assert len(vt.links) == 0

    C1 = example_2["C1"]
    C2 = example_2["C2"]
    assert len(VectorTransformationChain(C1).links) == 0
    assert len(VectorTransformationChain(C2).links) == 1

    C = example_3["C"]
    S = example_3["S"]
    assert len(VectorTransformationChain(C).links) == 0
    assert len(VectorTransformationChain(S).links) == 1

    C1 = example_4["C1"]
    C2 = example_4["C2"]
    C3 = example_4["C3"]
    S = example_4["S"]
    assert len(VectorTransformationChain(C1).links) == 0
    assert len(VectorTransformationChain(C2).links) == 1
    assert len(VectorTransformationChain(C3).links) == 2
    assert len(VectorTransformationChain(S).links) == 3


def test_express_errors():
    C1 = CoordSys3D("C1")
    C2 = CoordSys3D("C2")
    t = VectorTransformationChain(C2)
    # vector ok because it's defined using the target system, C2
    v_ok = C2.i + 2 * C2.j + 3 * C2.k
    # vector wrong because it's mixing different systems
    v_wrong_1 = C1.i + 2 * C2.j
    # vector wrong because it's defined in a different system than the target
    v_wrong_2 = C1.i + 2 * C1.j + 2 * C1.k

    # errors about `vector` not being in proper form
    pytest.raises(ValueError, lambda: t.express("a"))
    pytest.raises(ValueError, lambda: t.express(v_wrong_1))
    pytest.raises(ValueError, lambda: t.express(v_wrong_2))
    # too few components
    pytest.raises(ValueError, lambda: t.express((1, 2)))
    # too many components
    pytest.raises(ValueError, lambda: t.express((1, 2, 2, 4)))

    # `vector` is in proper form
    t.express(v_ok)
    t.express((1, 2, 3))

    # wrong system type
    pytest.raises(TypeError, lambda: t.express(v_ok, "a"))
    # system is not in the path
    pytest.raises(ValueError, lambda: t.express(v_ok, C1))


def test_express_1(example_2):
    C1 = example_2["C1"]
    C2 = example_2["C2"]
    a, b, c = symbols("a:c")
    v = a * C2.i + b * C2.j + c * C2.k
    t = VectorTransformationChain(C2)
    assert t.express(v) == a * C1.i + b * C1.j + c * C1.k


def test_express_2():
    alpha = symbols("alpha")
    C1 = CoordSys3D("C1")
    C2 = C1.orient_new_axis("C2", alpha, C1.k)
    a, b, c = symbols("a:c")

    t = VectorTransformationChain(C2)

    v1 = C2.i
    assert t.express(v1) == (
        cos(alpha) * C1.i +
        sin(alpha) * C1.j
    )

    v2 = C2.j
    assert t.express(v2) == (
        -sin(alpha) * C1.i +
        cos(alpha) * C1.j
    )

    v3 = C2.k
    assert t.express(v3) == C1.k

    v4 = a * C2.i + b * C2.j + c * C2.k
    assert t.express(v4) == (
        (a * cos(alpha) - b * sin(alpha)) * C1.i +
        (a * sin(alpha) + b * cos(alpha)) * C1.j +
        c * C1.k
    )


def test_express_3(example_3):
    C = example_3["C"]
    S = example_3["S"]
    r, theta, phi = S.base_scalars()

    t = VectorTransformationChain(S)

    # vector along the radial direction
    v1 = 1 * S.i
    assert t.express(v1) == (
        (sin(theta) * cos(phi)) * C.i +
        (sin(theta) * sin(phi)) * C.j +
        cos(theta) * C.k
    )

    # vector along the polar direction
    v2 = 1 * S.j
    # NOTE: r / sqrt(r**2) = 1
    assert t.express(v2) == (
        (r * cos(phi) * cos(theta) / sqrt(r**2)) * C.i +
        (r * sin(phi) * cos(theta) / sqrt(r**2)) * C.j +
        (-r * sin(theta) / sqrt(r**2)) * C.k
    )

    # vector along the azimuthal direction
    v3 = 1 * S.k
    # NOTE: r * sin(theta) / sqrt(r**2 * sin(theta)**2) = 1
    assert t.express(v3) == (
        (-r * sin(theta) * sin(phi) / sqrt(r**2 * sin(theta)**2)) * C.i +
        (r * sin(theta) * cos(phi) / sqrt(r**2 * sin(theta)**2)) * C.j
    )


def test_express_4(example_4):
    C1 = example_4["C1"]
    C2 = example_4["C2"]
    C3 = example_4["C3"]
    S = example_4["S"]
    r, theta, phi = S.base_scalars()
    alpha = symbols("alpha")

    t = VectorTransformationChain(S)

    # vector along the radial direction
    v1 = S.i
    assert t.express(v1) == (
        (sin(theta) * cos(phi + alpha)) * C1.i +
        (sin(theta) * sin(phi + alpha)) * C1.j +
        cos(theta) * C1.k
    )
    assert t.express(v1, C3) == (
        (sin(theta) * cos(phi)) * C3.i +
        (sin(theta) * sin(phi)) * C3.j +
        cos(theta) * C3.k
    )

    # vector along the polar direction
    v2 = 1 * S.j
    # NOTE: r / sqrt(r**2) = 1
    assert t.express(v2) == (
        (r * cos(phi + alpha) * cos(theta) / sqrt(r**2)) * C1.i +
        (r * sin(phi + alpha) * cos(theta) / sqrt(r**2)) * C1.j +
        (-r * sin(theta) / sqrt(r**2)) * C1.k
    )
    assert t.express(v2, C3) == (
        (r * cos(phi) * cos(theta) / sqrt(r**2)) * C3.i +
        (r * sin(phi) * cos(theta) / sqrt(r**2)) * C3.j +
        (-r * sin(theta) / sqrt(r**2)) * C3.k
    )

    # vector along the azimuthal direction
    v3 = 1 * S.k
    # NOTE: r * sin(theta) / sqrt(r**2 * sin(theta)**2) = 1
    assert t.express(v3) == (
        (-r * sin(theta) * sin(phi + alpha) / sqrt(r**2 * sin(theta)**2)) * C1.i +
        (r * sin(theta) * cos(phi + alpha) / sqrt(r**2 * sin(theta)**2)) * C1.j
    )
    assert t.express(v3, C3) == (
        (-r * sin(theta) * sin(phi) / sqrt(r**2 * sin(theta)**2)) * C3.i +
        (r * sin(theta) * cos(phi) / sqrt(r**2 * sin(theta)**2)) * C3.j
    )




# def test_er_maps_to_expected_symbolic_expression(frame_chain):
#     S = frame_chain["S"]
#     chain = VectorTransformationChain(S)
#     # e_r in S -> rewrite to root in symbolic form
#     v_root = chain.rewrite_to_root((1, 0, 0), basis="orthonormal")
#     # expected: [ sin(theta)*cos(phi - alpha), sin(theta)*sin(phi - alpha), cos(theta) ]
#     r_sym, th_sym, ph_sym = list(S.base_scalars())
#     alpha = frame_chain["alpha"]

#     expected = Matrix([
#         [sin(th_sym) * cos(ph_sym - alpha)],
#         [sin(th_sym) * sin(ph_sym - alpha)],
#         [cos(th_sym)]
#     ])
#     assert mat_equal_sym(simplify(v_root), simplify(expected))


# def test_evaluate_numeric_point(frame_chain):
#     S = frame_chain["S"]
#     chain = VectorTransformationChain(S)
#     # pick numeric values
#     subs_map = {S.base_scalars()[0]: 2.0, S.base_scalars()[1]: pi/4, S.base_scalars()[2]: pi/3,
#                 frame_chain["alpha"]: pi / 6}
#     v_root_sym = chain.rewrite_to_root((1, 0, 0), basis="orthonormal")
#     v_root_num = v_root_sym.subs(subs_map)
#     # compute expected numerically: e_r(θ,φ-α)
#     thv = float(pi / 4)
#     phv = float(pi / 3)
#     av = float(pi / 6)
#     expected_num = Matrix([
#         [math.sin(thv) * math.cos(phv - av)],
#         [math.sin(thv) * math.sin(phv - av)],
#         [math.cos(thv)]
#     ])
#     assert mat_allclose_numeric(v_root_num, expected_num, tol=1e-12)


# def test_covariant_contravariant_consistency(frame_chain):
#     S = frame_chain["S"]
#     lt = LocalTransform(S, get_parent(S))
#     # pick symbolic components v^i in contravariant form then convert to parent and lower with metric
#     v_contrav = Matrix([sympy.symbols("a b c")])
#     v_contrav = v_contrav.T  # shape (3,1) with a,b,c as symbols
#     v_cart_from_contrav = lt.apply(v_contrav, basis="contravariant")
#     # Now compute covariant components by lowering with metric and reconstruct
#     g = lt._compute_covariant_basis()
#     G = Matrix([[g[i].dot(g[j]) for j in range(3)] for i in range(3)])
#     v_cov = G * Matrix(v_contrav)
#     # reconstruct from v_cov using basis='covariant'
#     v_cart_from_cov = lt.apply(v_cov, basis="covariant")
#     assert mat_equal_sym(simplify(v_cart_from_contrav), simplify(v_cart_from_cov))


# def test_translation_ignored(frame_chain):
#     # vector should be unaffected by translation of C2 relative to C1
#     C1 = frame_chain["C1"]
#     C2 = frame_chain["C2"]
#     C3 = frame_chain["C3"]
#     S = frame_chain["S"]

#     # make a chain S -> C3 -> C2 -> C1
#     chain = VectorTransformationChain(S)
#     # get vector in root for v = (0,1,0) (theta direction)
#     v_root_before = chain.rewrite_to_root((0, 1, 0), basis="orthonormal")

#     # create a second C2' translated differently
#     C2p = C1.locate_new("C2p", 10 * C1.i)
#     C3p = C2p.orient_new_axis("C3p", frame_chain["alpha"], C2p.k)
#     Sp = C3p.create_new("Sp", transformation=S.transformation_to_parent())
#     chainp = VectorTransformationChain(Sp)
#     v_root_after = chainp.rewrite_to_root((0, 1, 0), basis="orthonormal")

#     # rotations are the same, translations different — vectors must be equal
#     assert mat_equal_sym(simplify(v_root_before), simplify(v_root_after))


# def test_cartesian_child_rotation(frame_chain):
#     # Test a purely Cartesian child rotated relative to parent
#     C1 = frame_chain["C1"]
#     C2 = C1.orient_new_axis("C2r", frame_chain["alpha"], C1.k)
#     # create a simple chain (C2 rotated relative to C1, no further child)
#     chain = VectorTransformationChain(C2)
#     # a vector expressed in C2's Cartesian axes (1,0,0) should map to C1: rotated by alpha
#     v_root = chain.rewrite_to_root((1, 0, 0), basis="orthonormal")
#     expected = Matrix([[cos(frame_chain["alpha"])], [-sin(frame_chain["alpha"])], [0]])
#     assert mat_equal_sym(simplify(v_root), simplify(expected))
