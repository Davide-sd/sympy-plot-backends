from sympy import lambdify, Tuple, sympify, Expr
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

def _unpack_args(*args):
    """ Given a list/tuple of arguments previously processed by _plot_sympify(),
    separates and returns its components: expressions, ranges, label.

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