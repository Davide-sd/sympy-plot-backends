from sympy import simplify, Matrix, diff, sqrt, Dummy
from sympy.vector import (
    CoordSys3D, BaseScalar, matrix_to_vector, express, Vector
)
from typing import Union


def _my_simplify(expr):
    # NOTE: this function is needed in order to workaround this bug:
    # https://github.com/sympy/sympy/issues/28727
    # TODO: remove this function once the bug has been solved.
    try:
        res = simplify(expr)
    except TypeError:
        d = {k: Dummy() for k in expr.atoms(BaseScalar)}
        rd = {v: k for k, v in d.items()}
        res = simplify(expr.subs(d)).subs(rd)
    return res


# NOTE: this function comes from sympy.vector.functions
def _path(from_object, to_object):
    """
    Calculates the 'path' of objects starting from 'from_object'
    to 'to_object', along with the index of the first common
    ancestor in the tree.

    Returns (index, list) tuple.
    """

    if from_object._root != to_object._root:
        raise ValueError(
            f"No connecting path found between {from_object} and {to_object}.")

    other_path = []
    obj = to_object
    while obj._parent is not None:
        other_path.append(obj)
        obj = obj._parent
    other_path.append(obj)
    object_set = set(other_path)
    from_path = []
    obj = from_object
    while obj not in object_set:
        from_path.append(obj)
        obj = obj._parent
    index = len(from_path)
    from_path.extend(other_path[other_path.index(obj)::-1])
    return index, from_path


def is_curvilinear(system: CoordSys3D) -> bool:
    """
    Return True if the coordinate system is curvilinear (non-Cartesian),
    determined by checking if Lamé coefficients != (1,1,1).
    """
    return not all(
        _my_simplify(coeff - 1) == 0
        for coeff in system.lame_coefficients()
    )


class LocalTransform:
    """
    Represents the local transform for one link in the system chain,
    consisting of two consecutive (connected) CoordSys3D.
    
    This allows to convert vectors specified in the `from_system` system
    to vectors specified in `to_system`.

    Notes
    -----

    Why is this class necessary? As of sympy 1.14, there is a bug in
    CoordSys3D, namely the wrong computation of the DCM matrix from
    curvilinear to cartesian (or viceversa). Consider this example:

    >>> C = CoordSys3D("C")
    >>> S = C.create_new("S", transformation="spherical")
    >>> R_fromS_toC = C.rotation_matrix(S)
    Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    This is wrong. The correct matrix should be:

    >>> X = Matrix(S.transformation_to_parent())
    >>> u = S.base_scalars()
    >>> J = X.jacobian(u)
    >>> h = S.lame_coefficients()
    >>> R_fromS_toC = Matrix([(J.col(i) / h[i]).T for i in range(3)]).T
    >>> R_fromS_toC
    Matrix([[sin(S.theta)*cos(S.phi), cos(S.phi)*cos(S.theta), -sin(S.phi)], [sin(S.phi)*sin(S.theta), sin(S.phi)*cos(S.theta), cos(S.phi)], [cos(S.theta), -sin(S.theta), 0]])

    This means that `CoordSys.scalar_map` and `express()` will compute wrong
    values when a curvilinear coordinate system is used.

    This class breaks down the path connecting two coordinates systems into
    links, where each link is represented by two consecutive systems. In doing
    so, the correct rotation matrix can be applied, and `express()` can be
    patched.
    """
    def __init__(self, from_system: CoordSys3D, to_system: CoordSys3D):
        if not all(isinstance(c, CoordSys3D) for c in [from_system, to_system]):
            raise TypeError(
                "`from_system` and `to_system` must both be instances of"
                " `CoordSys3D`. Instead, the following types were given:"
                f" type(from_system)={type(from_system).__name__},"
                f" type(to_system)={type(to_system).__name__}"
            )
        self.from_system = from_system
        self.to_system = to_system

        self._is_from_sys_curvilinear = is_curvilinear(from_system)
        self._is_to_sys_curvilinear = is_curvilinear(to_system)

    def apply(self, components: Matrix) -> Matrix:
        """
        Convert the components of a vector specified in `from_system` into a
        3x1 Matrix expressed in the `to_system`'s axes.
        """
        if self._is_from_sys_curvilinear:
            X = Matrix(self.from_system.transformation_to_parent())
            u = self.from_system.base_scalars()
            J = X.jacobian(u)
            # scale factors
            h = self.from_system.lame_coefficients()
            # here R_FromTo should be read as R_CurvCart: given a vector
            # expressed in Curvilinear frame, vCurv, then the following holds:
            # vCart = R_CurvCart * vCurv
            R_FromTo = Matrix([(J.col(i) / h[i]).T for i in range(3)]).T
        elif self._is_to_sys_curvilinear:
            X = Matrix(self.to_system.transformation_to_parent())
            u = self.to_system.base_scalars()
            J = X.jacobian(u)
            # scale factors
            h = self.to_system.lame_coefficients()
            # here R_ToFrom should be read as R_CurvCart: given a vector
            # expressed in Curvilinear frame, vCurv, then the following holds:
            # vCart = R_CurvCart * vCurv
            R_ToFrom = Matrix([(J.col(i) / h[i]).T for i in range(3)]).T
            R_FromTo = R_ToFrom.T
        else:
            R_FromTo = self.to_system.rotation_matrix(self.from_system)

        new_components = R_FromTo * components
        return new_components


def _build_chain(from_system, to_system):
    _, path = _path(from_system, to_system)
    links = []
    for i in range(1, len(path)):
        links.append(LocalTransform(path[i-1], path[i]))
    return links


def express(expr, system):
    """
    Express the provided vector to the specified system.

    Parameters
    ----------
    expr : Vector
        A Vector from the sympy.vector module.
    system : CoordSys3D
        The system where `expr` should be expressed.

    Returns
    -------

    vec : Vector
    """
    if not isinstance(system, CoordSys3D):
        raise TypeError("`system` must be an instance of CoordSys3D.")

    if expr in (0, Vector.zero):
        return expr

    if not isinstance(expr, Vector):
        # if dyadic or scalar field (symbolic expression)
        from sympy.vector import express
        return express(expr, system)

    sep = expr.separate()
    if len(sep) > 1:
        used_systems = ", ".join(str(t) for t in sep.keys())
        raise ValueError(
            "`expr` must be defined in one coordinate system,"
            f" {system}. Instead, it was defined using"
            f" multiple systems: {used_systems}"
        )

    curr_sys = set(sep).pop()
    links = _build_chain(curr_sys, system)
    if len(links) == 0:
        # empty chain => system is root
        return expr

    components = expr.to_matrix(curr_sys)

    v = links[0].apply(components)
    for lt in links[1:]:
        v = lt.apply(v)
    return matrix_to_vector(v, system)
