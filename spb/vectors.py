from spb.backends.base_backend import Plot
from spb.defaults import TWO_D_B, THREE_D_B
from spb.series import (
    Vector2DSeries, Vector3DSeries, ContourSeries, _set_discretization_points
)
from spb.utils import _plot_sympify, _unpack_args, _split_vector, _is_range
from sympy import S, sqrt, Expr, Tuple, Dummy, Symbol
from sympy.vector import Vector, BaseScalar
from sympy.matrices.dense import DenseMatrix
from sympy.core.compatibility import is_sequence


"""
TODO:
*   check length of ranges and if the scalar free symbols are compatible with
    the ones provided in the vector.
*   slice planes for 3D vector fields
"""

def _build_series(expr, *ranges, label="", show=True, **kwargs):
    
    # convert expr to a list of 3 elements
    split_expr, ranges = _split_vector(expr, ranges)

    # free symbols contained in the provided vector
    fs = set().union(*[e.free_symbols for e in split_expr])
    
    if split_expr[2] is S.Zero: # 2D case
        kwargs = _set_discretization_points(kwargs.copy(), Vector2DSeries)
        if len(fs) > 2:
            raise ValueError("Too many free symbols. 2D vector plots require " +
            "at most 2 free symbols. Received {}".format(fs))

        # check validity of ranges
        fs_ranges = set().union([r[0] for r in ranges])

        if len(fs_ranges) < 2:
            missing = fs.difference(fs_ranges)
            if not missing:
                raise ValueError(
                    "In a 2D vector field, 2 unique ranges are expected. " +
                    "Unfortunately, it is not possible to deduce them from " +
                    "the provided vector.\n" +
                    "Vector: {}, Free symbols: {}\n".format(expr, fs) +
                    "Provided ranges: {}".format(ranges)
                )
            ranges = list(ranges)
            for m in missing:
                ranges.append(Tuple(m, -10, 10))

        if len(ranges) > 2:
            raise ValueError("Too many ranges for 2D vector plot.")
        return split_expr, ranges, Vector2DSeries(*split_expr[:2], *ranges, 
                label, **kwargs)
    else: # 3D case
        kwargs = _set_discretization_points(kwargs.copy(), Vector3DSeries)
        if len(fs) > 3:
            raise ValueError("Too many free symbols. 3D vector plots require " +
            "at most 3 free symbols. Received {}".format(fs))

        # check validity of ranges
        fs_ranges = set().union([r[0] for r in ranges])

        if len(fs_ranges) < 3:
            missing = fs.difference(fs_ranges)
            if not missing:
                raise ValueError(
                    "In a 3D vector field, 3 unique ranges are expected. " +
                    "Unfortunately, it is not possible to deduce them from " +
                    "the provided vector.\n" +
                    "Vector: {}, Free symbols: {}\n".format(expr, fs) +
                    "Provided ranges: {}".format(ranges)
                )
            ranges = list(ranges)
            for m in missing:
                ranges.append(Tuple(m, -10, 10))

        if len(ranges) > 3:
            raise ValueError("Too many ranges for 3D vector plot.")

        return split_expr, ranges, Vector3DSeries(*split_expr, *ranges, label, 
                **kwargs)

def _preprocess(*args):
    """ Loops over the arguments and build a list of arguments having the
    following form: [expr, *ranges, label].
    `expr` can be a vector, a matrix or a list/tuple/Tuple. 
    """
    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        # In this case we received arguments in one of the following forms.
        # Here we wrapped them into a list, so that they can be further
        # processed:
        #   v               -> [v]
        #   v, range        -> [v, range]
        #   v1, v2, ..., range   -> [v1, v2, range]
        args = [args]
    if any([_is_range(a) for a in args]):
        args = [args]

    new_args = []
    for a in args:
        exprs, ranges, label = _unpack_args(*a)
        if len(exprs) == 1:
            new_args.append([*exprs, *ranges, label])
        else:
            # this is the case where the user provided: v1, v2, ..., range
            # we use the same ranges for each expression
            for e in exprs:
                new_args.append([e, *ranges, str(e)])
    
    return new_args

def vector_plot(*args, show=True, **kwargs):
    """ Plot a 2D or 3D vector field. By default, the aspect ratio of the plot
    is set to ``aspect="equal"``.

    Arguments
    =========
        expr : Vector, or Matrix with 2 or 3 elements, or list/tuple  with 2 or
                3 elements)
            Represent the vector to be plotted.
            Note: if a 3D vector is given with a list/tuple, it might happens
            that the internal algorithm could think of it as a range. Therefore,
            3D vectors should be given as a Matrix or as a Vector: this reduces
            ambiguities.
        
        ranges : 3-element tuples
            Denotes the range of the variables. For example (x, -5, 5). For 2D
            vector plots, 2 ranges should be provided. For 3D vector plots, 3
            ranges are needed.

        label : str
            The name of the vector field to be eventually shown on the legend.
            If none is provided, the string representation of the vector will
            be used.
        
        To specify multiple vector fields, wrap them into a tuple. Refer to the
        examples to learn more.

    Keyword Arguments
    =================

        contours_kw : dict
            A dictionary of keywords/values which is passed to the plotting
            library contour function to customize the appearance. Refer to the
            plotting library (backend) manual for more informations.

        n1, n2, n3 : int
            Number of discretization points in the x/y/z-direction respectively
            for the quivers or streamlines. Default to 25.
        
        n : int
            Set the same number of discretization points in all directions for
            the quivers or streamlines. It overrides n1, n2, n3.  Default to 25.

        nc : int
            Number of discretization points for the scalar contour plot.
            Default to 100.

        quiver_kw : dict
            A dictionary of keywords/values which is passed to the plotting
            library quivers function to customize the appearance. Refer to the
            plotting library (backend) manual for more informations.

        scalar : boolean, Expr, None or list/tuple of 2 elements
            Represents the scalar field to be plotted in the background. Can be:
                True: plot the magnitude of the vector field.
                False/None: do not plot any scalar field.
                Expr: a symbolic expression representing the scalar field.
                List/Tuple: [scalar_expr, label], where the label will be shown
                    on the colorbar.
            Default to True.
        
        show : boolean
            Default to True, in which case the plot will be shown on the screen. 

        streamlines : boolean
            Whether to plot the vector field using streamlines (True) or quivers
            (False). Default to False.
        
        stream_kw : dict
            A dictionary of keywords/values which is passed to the backend
            streamlines-plotting function to customize the appearance. Refer to
            the backend's manual for more informations.
    
    Examples
    ========

    Quivers plot of a 2D vector field with a contour plot in background
    representing the vector's magnitude (a scalar field):

    .. code-block:: python
        x, y = symbols("x, y")
        vector_plot([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3))
    
    
    Streamlines plot of a 2D vector field with no background scalar field:
    
    .. code-block:: python
        x, y = symbols("x, y")
        vector_plot([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
                streamlines=True, scalar=None)
    

    Plot of two 2D vectors with background the contour plot of magnitude of the
    first vector:

    .. code-block:: python
        x, y = symbols("x, y")
        vector_plot([-sin(y), cos(x)], [y, x], n=20,
            scalar=sqrt((-sin(y))**2 + cos(x)**2), legend=True)
    

    3D vectpr plot:

    .. code-block:: python
        x, y, z = symbols("x, y, z")
        vector_plot([x, y, z], (x, -10, 10), (y, -10, 10), (z, -10, 10),
                n=8)
    """
    args = _plot_sympify(args)
    args = _preprocess(*args)

    kwargs = _set_discretization_points(kwargs, Vector3DSeries)
    if not "n1" in kwargs: kwargs["n1"] = 25
    if not "n2" in kwargs: kwargs["n2"] = 25
    if not "aspect" in kwargs.keys():
        kwargs["aspect"] = "equal"

    series = []
    all_ranges = []
    for a in args:
        split_expr, ranges, s = _build_series(a[0], *a[1:-1], label=a[-1], **kwargs)
        series.append(s)
        all_ranges.append(ranges)
    
    # add a scalar series only on 2D plots
    if all([isinstance(s, Vector2DSeries) for s in series]):
        backend = kwargs.pop("backend", TWO_D_B)

        # don't pop this keyword: some backend needs it to decide the color
        # for quivers (solid color if a scalar field is present, gradient color
        # otherwise)
        scalar = kwargs.get("scalar", True)
        if (len(series) == 1) and (scalar == True):
            scalar_field = sqrt(split_expr[0]**2 + split_expr[1]**2)
            scalar_label = "Magnitude"
        elif (scalar == True):
            scalar_field = None # do nothing when
        elif isinstance(scalar, Expr):
            scalar_field = scalar
            scalar_label = str(scalar)
        elif isinstance(scalar, (list, tuple)):
            scalar_field = scalar[0]
            scalar_label = scalar[1]
        elif not scalar:
            scalar_field = None
        else:
            raise ValueError("``scalar`` must be either:\n" +
                "1. True, in which case the magnitude of the vector field " +
                "will be plotted.\n" +
                "2. a symbolic expression representing a scalar field.\n" +
                "3. None/False: do not plot any scalar field.\n" +
                "4. list/tuple of two elements, [scalar_expr, label]."
            )
        
        if scalar_field:
            # TODO: does it makes sense to cross-check the free symbols of the
            # scalar field with those of the vectors?

            # if not fs.issuperset(scalar_field.free_symbols):
            #     raise ValueError("The free symbols of the scalar field must be " +
            #         "a subset of the free symbols in the vector. Received:\n"
            #         "Vector free symbols: {}\n".format(fs) + 
            #         "Scalar field free symbols: {}".format(scalar_field.free_symbols) 
            #     )

            # plot the scalar field over the entire region covered by all
            # vector fields
            _minx, _maxx = float("inf"), -float("inf")
            _miny, _maxy = float("inf"), -float("inf")
            for r in all_ranges:
                _xr, _yr = r
                if _xr[1] < _minx:  _minx = _xr[1]
                if _xr[2] > _maxx:  _maxx = _xr[2]
                if _yr[1] < _miny:  _miny = _yr[1]
                if _yr[2] > _maxy:  _maxy = _yr[2]
            cranges = [
                Tuple(all_ranges[-1][0][0], _minx, _maxx),
                Tuple(all_ranges[-1][1][0], _miny, _maxy)
            ]
            nc = kwargs.pop("nc", 100)
            cs_kwargs = kwargs.copy()
            cs_kwargs["n1"] = nc
            cs_kwargs["n2"] = nc
            cs = ContourSeries(scalar_field, *cranges, scalar_label, **cs_kwargs)
            series = [cs] + series
    elif all([isinstance(s, Vector3DSeries)  for s in series]):
        backend = kwargs.pop("backend", THREE_D_B)
    else:
        raise ValueError("Mixing 2D vectors with 3D vectors is not allowed.")
    
    p = Plot(*series, backend=backend, **kwargs)
    if show:
        p.show()
    return p
