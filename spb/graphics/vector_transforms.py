from sympy import simplify, Matrix, diff, sqrt, Dummy
from sympy.vector import (
    CoordSys3D, BaseScalar, matrix_to_vector, express,
    VectorAdd, VectorMul, VectorZero, BaseVector
)
from typing import Union


def get_parent(system: CoordSys3D) -> Union[CoordSys3D, None]:
    """Return the parent reference system."""
    return getattr(
        system,
        "parent",
        getattr(system, "_parent", None)
    )


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


def is_curvilinear(system: CoordSys3D) -> bool:
    """
    Return True if the coordinate system is curvilinear (non-Cartesian),
    determined by checking if Lamé coefficients != (1,1,1).
    """
    return not all(
        _my_simplify(coeff - 1) == 0
        for coeff in system.lame_coefficients()
    )

# -----------------------
# LocalTransform: child -> parent
# -----------------------
class LocalTransform:
    """
    Represents the local transform for one link in the system chain: 
    child -> parent.
    
    Responsibilities:
      - Provide covariant basis g_i expressed in the PARENT's Cartesian axes
      - Provide orthonormal basis (unit) ê_i expressed in the PARENT's Cartesian axes
      - Convert vectors specified in the CHILD system to a parent-Cartesian 3x1 Matrix
    """
    def __init__(self, child: CoordSys3D, parent: CoordSys3D):
        if not all(isinstance(c, CoordSys3D) for c in [child, parent]):
            raise TypeError(
                "`child` and `parent` must both be instances of `CoordSys3D`."
                " Instead, the following types were given:"
                f" type(child)={type(child).__name__},"
                f" type(parent)={type(parent).__name__}"
            )
        self.child = child
        self.parent = parent
        self._is_curvilinear = is_curvilinear(child)

        # caches
        self.g_covariant = None         # list of 3 column Matrices (in parent axes)
        self.h = None                   # scale factors
        self.e_units = None             # orthonormal basis (in parent axes)

    def _compute_covariant_basis(self):
        """Compute covariant basis g_i (columns of Jacobian) expressed in parent axes."""
        if self.g_covariant:
            return self.g_covariant

        if self._is_curvilinear:
            # child.transformation_to_parent() returns (x(u), y(u), z(u)) expressed in child's base scalars
            X = Matrix(self.child.transformation_to_parent())
            u = self.child.base_scalars()
            g = []
            for i in range(3):
                gi = Matrix([diff(X[0], u[i]), diff(X[1], u[i]), diff(X[2], u[i])])
                g.append(_my_simplify(gi))

            self.g_covariant = g
            return g
        else:
            # Cartesian child: basis vectors are child axes expressed in parent axes via rotation
            # use child.rotation_matrix(parent)
            R = self.parent.rotation_matrix(self.child)
            e1 = R * Matrix([1, 0, 0])
            e2 = R * Matrix([0, 1, 0])
            e3 = R * Matrix([0, 0, 1])
            g = [_my_simplify(e1), _my_simplify(e2), _my_simplify(e3)]
            self.g_covariant = g
            return g

    def _compute_scale_factors(self):
        """Scale factors h_i = ||g_i|| (symbolic)."""
        if self.h:
            return self.h
        g = self._compute_covariant_basis()
        h = [_my_simplify(sqrt(g[i].dot(g[i]))) for i in range(3)]
        self.h = h
        return h

    def _compute_orthonormal_basis(self):
        """Unit orthonormal basis ê_i = g_i / h_i, expressed in parent axes."""
        if self.e_units:
            return self.e_units
        g = self._compute_covariant_basis()
        h = self._compute_scale_factors()
        e = []
        for i in range(3):
            ei = _my_simplify(g[i] / h[i])
            e.append(ei)
        self.e_units = e
        return e

    def apply(self, components, basis="orthonormal"):
        """
        Convert a vector specified in the CHILD system (components, basis) into
        a 3x1 Matrix expressed in the PARENT's Cartesian axes.

        basis: "orthonormal", "contravariant", "covariant"
        """
        components = Matrix(components)

        if basis == "orthonormal":
            # components are v_r, v_theta, v_phi relative to unit basis ê_i (physical components)
            if self._is_curvilinear:
                # ê_i are already expressed in PARENT axes
                e = self._compute_orthonormal_basis()
                V = Matrix([0, 0, 0])
                for i in range(3):
                    V += components[i] * e[i]
                return _my_simplify(V)
            else:
                # Cartesian child: orthonormal components are child Cartesian components;
                # express them in parent by rotating child->parent
                R = self.parent.rotation_matrix(self.child)
                V_child = Matrix(components)
                V_parent = R * V_child
                return _my_simplify(V_parent)

        elif basis == "contravariant":
            # components are v^i, vector = v^i g_i (g_i are covariant basis)
            g = self._compute_covariant_basis()
            V = Matrix([0, 0, 0])
            for i in range(3):
                V += components[i] * g[i]
            return _my_simplify(V)

        elif basis == "covariant":
            # components are v_i, vector = v_i g^i (g^i = G^{ij} g_j)
            g = self._compute_covariant_basis()
            G = Matrix([[g[i].dot(g[j]) for j in range(3)] for i in range(3)])
            Ginv = _my_simplify(G.inv())
            # build contravariant basis g^i = G^{ij} g_j
            g_contra = []
            for i in range(3):
                vec = Matrix([0, 0, 0])
                for j in range(3):
                    vec += Ginv[i, j] * g[j]
                g_contra.append(_my_simplify(vec))
            V = Matrix([0, 0, 0])
            for i in range(3):
                V += components[i] * g_contra[i]
            return _my_simplify(V)

        else:
            raise ValueError("basis must be 'orthonormal','contravariant' or 'covariant'")

# -----------------------
# VectorTransformationChain
# -----------------------
class VectorTransformationChain:
    """
    Build the chain of LocalTransforms from a given CoordSys3D up to the root system,
    and provide a method to rewrite vectors (given in the original system) into the
    root Cartesian basis.
    """
    def __init__(self, system: CoordSys3D):
        if not isinstance(system, CoordSys3D):
            raise TypeError("`system` must be an instance of CoordSys3D.")

        self.system = system
        # list of all the connected systems from `system` to its root.
        self._connected_systems = [system]
        # list of LocalTransform objects in upward order:
        # from `system` to its root
        self.links = self._build_chain(system)

    def _build_chain(self, system):
        """
        Walk from 'system' up to root, creating LocalTransform(child, parent) objects.
        The chain is ordered: [ (system->parent), (parent->grandparent), ... ]
        """
        chain = []
        current = system
        parent = get_parent(current)
        while parent is not None:
            self._connected_systems.append(parent)
            lt = LocalTransform(child=current, parent=parent)
            chain.append(lt)
            current = parent
            parent = get_parent(current)
        return chain

    def express(self, vector, system=None, basis="orthonormal"):
        """
        Convert `components` given in `self.system` (with basis specified) into
        a 3x1 Matrix expressed in the ROOT system's Cartesian axes.

        Algorithm:
          v := components (in child basis)
          for each LocalTransform LT in chain:
              v_parent := LT.apply(v, basis)   # Note: after first apply, the vector is in parent-cartesian
              # For subsequent links, the vector is already in the child-cartesian of the next link,
              # so we must update the representation: for further iteration, we always treat the
              # incoming v as a vector expressed in the current link's CHILD system (cartesian axes).
              # Therefore, set basis='cartesian' for subsequent transforms (handled below).
          return v (now in root-cartesian)
        
        Parameters
        ----------
        vector : Vector or list of Expr
            A Vector from the sympy.vector module or a list of 3 symbolic
            expressions, representing its components.
        system : CoordSys3D or None
            The system where `vector` should be expressed. If None, the vector
            will be expressed in the root system.
        """
        if system is None:
            system = self.system._root
        else:
            if not isinstance(system, CoordSys3D):
                raise TypeError(
                    "`system` must be an instance of CoordSys3D."
                )
            if not system in self._connected_systems:
                raise ValueError(
                    f"System `{system}` is not part of the path"
                    f" {" -> ".join(str(t) for t in self._connected_systems)}."
                )

        if isinstance(vector, VectorZero):
            return vector
        elif isinstance(vector, (VectorAdd, VectorMul, BaseVector)):
            sep = vector.separate()
            if len(sep) > 1:
                used_systems = ", ".join(str(t) for t in sep.keys())
                raise ValueError(
                    "`vector` must be defined in one coordinate system,"
                    f" {self.system}. Instead, `vector` was defined using"
                    f" multiple systems: {used_systems}"
                )

            curr_sys = list(sep.keys())[0]
            if curr_sys != self.system:
                raise ValueError(
                    f"`vector` must be defined in {self.system}. Instead,"
                    f" it is currently defined in {curr_sys}."
                )
            components = vector.to_matrix(self.system)
        elif not hasattr(vector, "__iter__"):
            raise ValueError("`vector` must be an iterable with 3 components.")
        elif len(vector) != 3:
            raise ValueError("`vector` must be an iterable with 3 components.")
        else:
            components = vector


        # Step 0: empty chain => system is root; then components must already
        # be Cartesian in root axes
        if not self.links:
            return Matrix(components)

        # Step 1: apply first link using provided basis
        v = self.links[0].apply(components, basis=basis)
        # After this call, v is in parent (of original system) Cartesian axes.

        # Step 2: for the remaining links, we have a vector expressed in the CURRENT system's Cartesian axes.
        # We must pass it through remaining LocalTransforms, BUT those expect "components relative to CHILD basis".
        # For Cartesian child links, the "components" for LT.apply should be child-cartesian components.
        # Because v is already in the 'current' system's Cartesian axes (which equals the child for the next link),
        # we can just feed the numeric components directly to the next link, but as a 3x1 column.
        for lt in self.links[1:]:
            if lt.child == system:
                break
            # v is currently expressed in lt.child Cartesian axes (by construction of chain)
            # So call lt.apply with basis='orthonormal' and components = v (3x1)
            # But lt.apply expects components iterable; ensure it's a 3-vector
            v = lt.apply(v, basis="orthonormal")
        return matrix_to_vector(_my_simplify(v), system)