from sympy import lambdify, Tuple, sympify, Expr
from sympy.matrices.dense import DenseMatrix
from sympy.utilities.iterables import ordered
import numpy as np

def get_lambda(expr, modules="numpy", **kwargs):
    """ Create a lambda function to numerically evaluate expr by sorting 
    alphabetically the function arguments.
    Parameters
    ----------
        expr : Expr
            The Sympy expression to convert.
        modules : str
            The numerical module to use for evaluation. Default to "numpy".
            See help(lambdify) for other choices.
        **kwargs
            Other keyword arguments to the function lambdify.
    Returns
    -------
        s : list
            The function signature: a list of the ordered function arguments.
        f : lambda
            The generated lambda function.ò 
    """
    signature = list(ordered(expr.free_symbols))
    return signature, lambdify(signature, expr, modules=modules, **kwargs)


def _plot_sympify(args):
    """ By allowing the users to set custom labels to the expressions being
    plotted, a critical issue is raised: whenever a special character like $,
    {, }, ... is used in the label (type string), sympify will raise an error.
    This function recursively loop over the arguments passed to the plot
    functions: the sympify function will be applied to all arguments except
    those of type string.
    """
    if isinstance(args, Expr):
        return args

    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, (list, tuple)):
            args[i] = Tuple(*_plot_sympify(a), sympify=False)
        elif not isinstance(a, str):
            args[i] = sympify(a)
    if isinstance(args, tuple):
        return Tuple(*args, sympify=False)
    return args


def _is_range(r):
    """ A range is defined as (symbol, start, end). start and end should
    be numbers.
    """
    return (isinstance(r, Tuple) and (len(r) == 3) and
                r.args[1].is_number and r.args[2].is_number)

def _unpack_args(*args, matrices=False):
    """ Given a list/tuple of arguments previously processed by _plot_sympify(),
    separates and returns its components: expressions, ranges, label.

    Parameters
    ==========
        matrices : boolean
            Default to False. If True, when a single DenseMatrix is given as
            the expression, it will be converted to a list. This is useful in
            order to deal with vectors (written in form of matrices) for
            iplot.

    Examples
    ========

    >>> from sympy import cos, sin, symbols
    >>> x, y = symbols('x, y')
    >>> args = (sin(x), (x, -10, 10), "f1")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
        ([sin(x)], [(x, -2, 2)], 'f1')

    >>> args = (sin(x**2 + y**2), (x, -2, 2), (y, -3, 3), "f2")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
        ([sin(x**2 + y**2)], [(x, -2, 2), (y, -3, 3)], 'f2')
    
    >>> args = (sin(x + y), cos(x - y), x + y, (x, -2, 2), (y, -3, 3), "f3")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
        ([sin(x + y), cos(x - y), x + y], [(x, -2, 2), (y, -3, 3)], 'f3')
    """
    ranges = [t for t in args if _is_range(t)]
    labels = [t for t in args if isinstance(t, str)]
    label = "" if not labels else labels[0]
    results = [not (_is_range(a) or isinstance(a, str)) for a in args]
    exprs = [a for a, b in zip(args, results) if b]

    if label == "":
        if len(exprs) == 1:
            label = str(exprs[0])
        else:
            label = str(tuple(exprs))
    
    if matrices:
        # TODO: need to deals with instances of Vector
        if (len(exprs) == 1) and (isinstance(exprs[0], (list, tuple, Tuple, DenseMatrix))):
            exprs = list(exprs[0])
    return exprs, ranges, label


def ij2k(cols, i, j):
    """ Create the connectivity for the mesh.
    https://github.com/K3D-tools/K3D-jupyter/issues/273
    """
    return  cols * i + j

def get_vertices_indices(x, y, z):
    """ Compute the vertices matrix (Nx3) and the connectivity list for
    triangular faces.

    Parameters
    ==========
        x, y, z : np.array
            2D arrays
    """
    rows, cols  = x.shape
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    vertices = np.vstack([x, y, z]).T
    indices = []
    for i in range(1,rows):
        for j in range(1,cols):
            indices.append( [ij2k(cols, i, j), ij2k(cols, i - 1, j), ij2k(cols, i, j- 1 )] )
            indices.append( [ij2k(cols, i - 1, j - 1), ij2k(cols, i , j - 1), ij2k(cols, i - 1, j)] )
    return vertices, indices

def get_seeds_points(xx, yy, zz, uu, vv, ww):
    """ Returns an optimal list of seeds points to be used to generate 3D
    streamlines.

    Parameters
    ==========
        xx, yy, zz: np.ndarray
            3D discretization of the space from meshgrid.

        uu, vv, ww: np.ndarray
            Vector components calculated at the discretized points in space.
    
    Returns
    =======
        points : np.ndarray
            [n x 3] matrix of seed-points coordinates.
    """

    # TODO: OPTIMIZE THIS CODE PLEASE!!!!!!!!!

    coords = np.stack([xx, yy, zz])
    # vector field
    vf = np.stack([uu, vv, ww])
    
    # extract the coordinate of the points at planes:
    # x_min, x_max, y_min, y_max, z_min, z_max
    c_xmin = coords[:, :, 0, :]
    c_xmax = coords[:, :, -1, :]
    c_ymin = coords[:, 0, :, :]
    c_ymax = coords[:, -1, :, :]
    c_zmin = coords[:, :, :, 0]
    c_zmax = coords[:, :, :, -1]

    # extract the vector field points at planes:
    # x_min, x_max, y_min, y_max, z_min, z_max
    vf_xmin = vf[:, :, 0, :]
    vf_xmax = vf[:, :, -1, :]
    vf_ymin = vf[:, 0, :, :]
    vf_ymax = vf[:, -1, :, :]
    vf_zmin = vf[:, :, :, 0]
    vf_zmax = vf[:, :, :, -1]
    
    def find_points_at_input_vectors(vf_plane, coords, i, sign="g"):
        check = {
            "g": lambda x: x > 0,
            "l": lambda x: x < 0,
        }
        # extract coordinates where the vectors are entering the plane
        tmp = np.where(check[sign](vf_plane[i, :, :]), coords, np.nan * np.zeros_like(coords))
        # reshape the matrix to obtain an [n x 3] array of coordinates
        tmp = np.array([tmp[0, :, :].flatten(), tmp[1, :, :].flatten(), tmp[2, :, :].flatten()]).T
        # remove NaN entries
        tmp = [a for a in tmp if not np.all([np.isnan(t) for t in a])]
        return tmp
    
    p_xmin = find_points_at_input_vectors(vf_xmin, c_xmin, 0, "g")
    p_xmax = find_points_at_input_vectors(vf_xmax, c_xmax, 0, "l")
    p_ymin = find_points_at_input_vectors(vf_ymin, c_ymin, 1, "g")
    p_ymax = find_points_at_input_vectors(vf_ymax, c_ymax, 1, "l")
    p_zmin = find_points_at_input_vectors(vf_zmin, c_zmin, 2, "g")
    p_zmax = find_points_at_input_vectors(vf_zmax, c_zmax, 2, "l")
    # TODO: there could be duplicates
    points = np.array(p_xmin + p_xmax + p_ymin + p_ymax + p_zmin + p_zmax)
    return points
