from spb.defaults import TWO_D_B, THREE_D_B, cfg
from spb.functions import _set_labels
from spb.series import (
    BaseSeries, Vector2DSeries, Vector3DSeries, ContourSeries,
    SliceVector3DSeries, InteractiveSeries, _set_discretization_points
)
from spb.utils import (
    _plot_sympify, _unpack_args_extended, _split_vector, _is_range,
    _instantiate_backend
)
from sympy import (
    Tuple, sqrt, Expr, S, Plane
)
from sympy.external import import_module


def _build_series(*args, interactive=False, **kwargs):
    """Loop over args and create all the necessary series to display a vector
    plot.
    """
    np = import_module('numpy')
    series = []
    all_ranges = []
    is_vec_lambda_function = False
    for a in args:
        split_expr, ranges, s = _series(
            a[0], *a[1:-1], label=a[-1], interactive=interactive, **kwargs
        )
        all_ranges.append(ranges)
        if isinstance(s, (list, tuple)):
            series += s
        else:
            series.append(s)
        if any(callable(e) for e in split_expr):
            is_vec_lambda_function = True

    # add a scalar series only on 2D plots
    if all([s.is_2Dvector for s in series]):
        # NOTE: don't pop this keyword: some backend needs it to decide the
        # color for quivers (solid color if a scalar field is present, gradient
        # color otherwise)
        scalar = kwargs.get("scalar", True)
        if (len(series) == 1) and (scalar is True):
            if not is_vec_lambda_function:
                scalar_field = sqrt(split_expr[0] ** 2 + split_expr[1] ** 2)
            else:
                scalar_field = lambda x, y: (np.sqrt(
                    split_expr[0](x, y) ** 2 + split_expr[1](x, y) ** 2))
            scalar_label = "Magnitude"
        elif scalar is True:
            scalar_field = None  # do nothing when
        elif isinstance(scalar, Expr):
            scalar_field = scalar
            scalar_label = str(scalar)
        elif isinstance(scalar, (list, tuple)):
            scalar_field = scalar[0]
            scalar_label = scalar[1]
        elif callable(scalar):
            scalar_field = scalar
            scalar_label = "Magnitude"
        elif not scalar:
            scalar_field = None
        else:
            raise ValueError(
                "`scalar` must be either:\n"
                + "1. True, in which case the magnitude of the vector field "
                + "will be plotted.\n"
                + "2. a symbolic expression representing a scalar field.\n"
                + "3. None/False: do not plot any scalar field.\n"
                + "4. list/tuple of two elements, [scalar_expr, label].\n"
                + "5. a numerical function of 2 variables supporting "
                + "vectorization."
            )

        if scalar_field:
            # plot the scalar field over the entire region covered by all
            # vector fields
            _minx, _maxx = float("inf"), -float("inf")
            _miny, _maxy = float("inf"), -float("inf")
            for r in all_ranges:
                _xr, _yr = r
                if _xr[1] < _minx:
                    _minx = _xr[1]
                if _xr[2] > _maxx:
                    _maxx = _xr[2]
                if _yr[1] < _miny:
                    _miny = _yr[1]
                if _yr[2] > _maxy:
                    _maxy = _yr[2]
            cranges = [
                Tuple(all_ranges[-1][0][0], _minx, _maxx),
                Tuple(all_ranges[-1][1][0], _miny, _maxy),
            ]
            nc = kwargs.pop("nc", 100)
            cs_kwargs = kwargs.copy()
            cs_kwargs["n1"] = nc
            cs_kwargs["n2"] = nc
            if not interactive:
                cs = ContourSeries(scalar_field, *cranges, scalar_label, **cs_kwargs)
            else:
                cs = InteractiveSeries(
                    [scalar_field], cranges, scalar_label, **cs_kwargs
                )
            series = [cs] + series

    return series


def _series(expr, *ranges, label="", interactive=False, **kwargs):
    """Create a vector series from the provided arguments."""
    params = kwargs.get("params", dict())
    fill_ranges = True if params == dict() else False
    # convert expr to a list of 3 elements
    split_expr, ranges = _split_vector(expr, ranges, fill_ranges)

    # free symbols contained in the provided vector
    fs = set()
    if not any(callable(e) for e in split_expr):
        fs = fs.union(*[e.free_symbols for e in split_expr])
    # if we are building a parametric-interactive series, remove the
    # parameters
    fs = fs.difference(params.keys())

    if split_expr[2] is S.Zero:  # 2D case
        kwargs = _set_discretization_points(kwargs.copy(), Vector2DSeries)
        if len(fs) > 2:
            raise ValueError(
                "Too many free symbols. 2D vector plots require "
                + "at most 2 free symbols. Received {}".format(fs)
            )

        # check validity of ranges
        fs_ranges = set().union([r[0] for r in ranges])

        if len(fs_ranges) < 2:
            missing = fs.difference(fs_ranges)
            if not missing:
                raise ValueError(
                    "In a 2D vector field, 2 unique ranges are expected. "
                    + "Unfortunately, it is not possible to deduce them from "
                    + "the provided vector.\n"
                    + "Vector: {}, Free symbols: {}\n".format(expr, fs)
                    + "Provided ranges: {}".format(ranges)
                )
            ranges = list(ranges)
            for m in missing:
                ranges.append(Tuple(m, cfg["plot_range"]["min"], cfg["plot_range"]["max"]))

        if len(ranges) > 2:
            raise ValueError("Too many ranges for 2D vector plot.")
        if not interactive:
            return (
                split_expr,
                ranges,
                Vector2DSeries(*split_expr[:2], *ranges, label, **kwargs),
            )
        return (
            split_expr,
            ranges,
            InteractiveSeries(split_expr[:2], ranges, label, **kwargs),
        )
    else:  # 3D case
        kwargs = _set_discretization_points(kwargs.copy(), Vector3DSeries)
        if len(fs) > 3:
            raise ValueError(
                "Too many free symbols. 3D vector plots require "
                + "at most 3 free symbols. Received {}".format(fs)
            )

        # check validity of ranges
        fs_ranges = set().union([r[0] for r in ranges])

        if len(fs_ranges) < 3:
            missing = fs.difference(fs_ranges)
            if not missing:
                raise ValueError(
                    "In a 3D vector field, 3 unique ranges are expected. "
                    + "Unfortunately, it is not possible to deduce them from "
                    + "the provided vector.\n"
                    + "Vector: {}, Free symbols: {}\n".format(expr, fs)
                    + "Provided ranges: {}".format(ranges)
                )
            ranges = list(ranges)
            for m in missing:
                ranges.append(Tuple(m, cfg["plot_range"]["min"], cfg["plot_range"]["max"]))

        if len(ranges) > 3:
            raise ValueError("Too many ranges for 3D vector plot.")

        _slice = kwargs.pop("slice", None)
        if _slice is None:
            if not interactive:
                return (
                    split_expr,
                    ranges,
                    Vector3DSeries(*split_expr, *ranges, label, **kwargs),
                )
            return (
                split_expr,
                ranges,
                InteractiveSeries(split_expr, ranges, label, **kwargs),
            )

        # verify that the slices are of the correct type
        # NOTE: currently, the slice cannot be a lambda function. To understand
        # the reason, look at series.py -> _build_slice_series: we use
        # symbolic manipulation!
        def _check_slice(s):
            if not isinstance(s, (Expr, Plane, BaseSeries)):
                raise ValueError(
                    "A slice must be of type Plane or Expr or BaseSeries.\n"
                    + "Received: {}, {}".format(type(s), s)
                )

        if isinstance(_slice, (list, tuple, Tuple)):
            for s in _slice:
                _check_slice(s)
        else:
            _check_slice(_slice)
            _slice = [_slice]

        series = []
        for s in _slice:
            if not interactive:
                series.append(
                    SliceVector3DSeries(s, *split_expr, *ranges, label, **kwargs)
                )
            else:
                # TODO: this needs to be redone
                series.append(
                    InteractiveSeries(split_expr, ranges, label, slice=s, **kwargs)
                )
        return split_expr, ranges, series


def _preprocess(*args, matrices=False, fill_ranges=True):
    """Loops over the arguments and build a list of arguments having the
    following form: [expr, *ranges, label].
    `expr` can be a vector, a matrix or a list/tuple/Tuple.

    `matrices` and `fill_ranges` are going to be passed to
    `_unpack_args_extended`.
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
        exprs, ranges, label, rendering_kw = _unpack_args_extended(
            *a, matrices=matrices, fill_ranges=fill_ranges
        )
        if len(exprs) == 1:
            new_args.append([*exprs, *ranges, label])

        else:
            # this is the case where the user provided: v1, v2, ..., range
            # we use the same ranges for each expression
            for e in exprs:
                new_args.append([e, *ranges, None])
    return new_args


def plot_vector(*args, **kwargs):
    """
    Plot a 2D or 3D vector field. By default, the aspect ratio of the plot
    is set to `aspect="equal"`.

    Typical usage examples are in the followings:

    - Plotting a vector field with a single range.
        `plot(expr, range1, range2, range3 [optional], **kwargs)`

    - Plotting multiple vector fields with different ranges and custom labels.
        `plot((expr1, range1, range2, range3 [optional], label1 [optional]), (expr2, range4, range5, range6 [optional], label2 [optional]), **kwargs)`

    Parameters
    ==========

    args :
        expr : Vector, or Matrix/list/tuple with 2 or 3 elements
            Represents the vector to be plotted. It can be a:

            * Vector from the `sympy.vector` module or from the
              `sympy.physics.mechanics` module.
            * Matrix/list/tuple with 2 (or 3) symbolic elements.
            * list/tuple with 2 (or 3) numerical functions of 2 (or 3)
              variables.

            Note: if a 3D symbolic vector is given with a list/tuple, it might
            happens that the internal algorithm thinks of it as a range.
            Therefore, 3D vectors should be given as a Matrix or as a Vector:
            this reduces ambiguities.

        ranges : 3-element tuples
            Denotes the range of the variables. For example (x, -5, 5). For 2D
            vector plots, 2 ranges should be provided. For 3D vector plots, 3
            ranges are needed.

        label : str, optional
            The name of the vector field to be eventually shown on the legend
            or colorbar. If none is provided, the string representation of
            the vector will be used.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    contour_kw : dict
        A dictionary of keywords/values which is passed to the backend
        contour function to customize the appearance. Refer to the plotting
        library (backend) manual for more informations.

    label : list/tuple, optional
        The label to be shown in the colorbar if ``scalar=None``.
        If not provided, the string representation of `expr` will be used.
        The number of labels must be equal to the number of series generated
        by the plotting function.

    n1, n2, n3 : int
        Number of discretization points for the quivers or streamlines in the
        x/y/z-direction, respectively. Default to 25.

    n : int or three-elements tuple (n1, n2, n3), optional
        If an integer is provided, the ranges are sampled uniformly
        at `n` number of points. If a tuple is provided, it overrides
        `n1`, `n2` and `n3`. Default to 25.

    nc : int
        Number of discretization points for the scalar contour plot.
        Default to 100.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of ``iplot``.

    quiver_kw : dict
        A dictionary of keywords/values which is passed to the backend quivers-
        plotting function to customize the appearance. Refer to the plotting
        library (backend) manual for more informations.

    scalar : boolean, Expr, None or list/tuple of 2 elements
        Represents the scalar field to be plotted in the background of a 2D
        vector field plot. Can be:

        - `True`: plot the magnitude of the vector field. Only works when a
          single vector field is plotted.
        - `False`/`None`: do not plot any scalar field.
        - `Expr`: a symbolic expression representing the scalar field.
        - `list`/`tuple`: [scalar_expr, label], where the label will be
          shown on the colorbar.
        - a numerical function of 2 variables supporting vectorization.

        Default to True.

    show : boolean
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    slice : Plane, list, Expr
        Plot the 3D vector field over the provided slice. It can be:

        - a Plane object from sympy.geometry module.
        - a list of planes.
        - an instance of ``SurfaceOver2DRangeSeries`` or
          ``ParametricSurfaceSeries``.
        - a symbolic expression representing a surface of two variables.

        The number of discretization points will be `n1`, `n2`, `n3`.
        Note that:

        - only quivers plots are supported with slices. Streamlines plots
          are unaffected.
        - `n3` will only be used with planes parallel to xz or yz.
        - `n1`, `n2`, `n3` doesn't affect the slice if it is an instance of
          ``SurfaceOver2DRangeSeries`` or ``ParametricSurfaceSeries``.

    streamlines : boolean
        Whether to plot the vector field using streamlines (True) or quivers
        (False). Default to False.

    stream_kw : dict
        A dictionary of keywords/values which is passed to the backend
        streamlines-plotting function to customize the appearance. Refer to
        the Notes section to learn more.

        For 3D vector fields, by default the streamlines will start at the
        boundaries of the domain where the vectors are pointed inward.
        Depending on the vector field, this may results in too tight
        streamlines. Use the `starts` keyword argument to control the
        generation of streamlines:

        - `starts=None`: the default aforementioned behaviour.
        - `starts=dict(x=x_list, y=y_list, z=z_list)`: specify the starting
          points of the streamlines.
        - `starts=True`: randomly create starting points inside the domain.
          In this setup we can set the number of starting point with `npoints`
          (default value to 200).

        If 3D streamlines appears to be cut short inside the specified domain,
        try to increase `max_prop` (default value to 5000).

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    zlabel : str, optional
        Label for the z-axis. Only available for 3D plots.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    zlim : (float, float), optional
        Denotes the z-axis limits, `(min, max)`. Only available for 3D plots.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, cos, Plane, Matrix, sqrt
       >>> from spb.vectors import plot_vector
       >>> x, y, z = symbols('x, y, z')

    Quivers plot of a 2D vector field with a contour plot in background
    representing the vector's magnitude (a scalar field).

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
       ...     quiver_kw=dict(color="black"),
       ...     contour_kw={"cmap": "Blues_r", "levels": 20},
       ...     grid=False, xlabel="x", ylabel="y")
       Plot object containing:
       [0]: contour: sqrt(sin(y)**2 + cos(x)**2) for x over (-3.0, 3.0) and y over (-3.0, 3.0)
       [1]: 2D vector series: [-sin(y), cos(x)] over (x, -3.0, 3.0), (y, -3.0, 3.0)

    Streamlines plot of a 2D vector field with no background scalar field and
    custom label:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = [-sin(y), cos(x)]
       >>> plot_vector(expr, (x, -3, 3), (y, -3, 3),
       ...     streamlines=True, scalar=None,
       ...     label="Magnitude of %s" % str(expr), xlabel="x", ylabel="y")
       Plot object containing:
       [0]: 2D vector series: [-sin(y), cos(x)] over (x, -3.0, 3.0), (y, -3.0, 3.0)


    Plot multiple 2D vectors fields, setting a background scalar field to be
    the magnitude of the first vector.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector([-sin(y), cos(x)], [y, x], (x, -5, 5), (y, -3, 3), n=20,
       ...     scalar=sqrt((-sin(y))**2 + cos(x)**2), legend=True, grid=False,
       ...     xlabel="x", ylabel="y")
       Plot object containing:
       [0]: contour: sqrt(sin(y)**2 + cos(x)**2) for x over (-5.0, 5.0) and y over (-3.0, 3.0)
       [1]: 2D vector series: [-sin(y), cos(x)] over (x, -5.0, 5.0), (y, -3.0, 3.0)
       [2]: 2D vector series: [y, x] over (x, -5.0, 5.0), (y, -3.0, 3.0)

    Plotting a the streamlines of a 2D vector field defined with numerical
    functions instead of symbolic expressions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> f = lambda x, y: np.sin(2 * x + 2 * y)
       >>> fx = lambda x, y: np.cos(f(x, y))
       >>> fy = lambda x, y: np.sin(f(x, y))
       >>> plot_vector([fx, fy], ("x", -1, 1), ("y", -1, 1),
       ...     streamlines=True, scalar=False)

    Interactive-widget 2D vector plot. Refer to ``iplot`` documentation to
    learn more about the ``params`` dictionary.

    .. code-block:: python

       from sympy import *
       from spb import *
       x, y, u = symbols("x y u")
       plot_vector(
           [-sin(u * y), cos(x)], (x, -3, 3), (y, -3, 3),
           params={u: (1, 0, 2)},
           n=20,
           quiver_kw=dict(color="black"),
           contour_kw={"cmap": "Blues_r", "levels": 20},
           grid=False, xlabel="x", ylabel="y")

    3D vector field.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
       ...      label="Magnitude", n=8, quiver_kw={"length": 0.1},
       ...      xlabel="x", ylabel="y", zlabel="z")
       Plot object containing:
       [0]: 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0)

    3D vector field with 3 orthogonal slice planes.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
       ...      n=8, quiver_kw={"length": 0.1},
       ...      slice=[
       ...          Plane((-10, 0, 0), (1, 0, 0)),
       ...          Plane((0, 10, 0), (0, 2, 0)),
       ...          Plane((0, 0, -10), (0, 0, 1))],
       ...      label=["Magnitude"] * 3, xlabel="x", ylabel="y", zlabel="z")
       Plot object containing:
       [0]: sliced 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0) at Plane(Point3D(-10, 0, 0), (1, 0, 0))
       [1]: sliced 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0) at Plane(Point3D(0, 10, 0), (0, 2, 0))
       [2]: sliced 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0) at Plane(Point3D(0, 0, -10), (0, 0, 1))

    3D vector streamlines starting at a 1000 random points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector(
       ...     Matrix([z, y, x]), (x, -10, 10), (y, -10, 10), (z, -10, 10),
       ...     streamlines=True,
       ...     stream_kw=dict(
       ...         starts=True,
       ...         npoints=1000
       ...     ),
       ...     label="Magnitude", xlabel="x", ylabel="y", zlabel="z")
       Plot object containing:
       [0]: 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0)

    See Also
    ========

    iplot

    """
    args = _plot_sympify(args)
    args = _preprocess(*args)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    kwargs = _set_discretization_points(kwargs, Vector3DSeries)
    kwargs.setdefault("aspect", "equal")
    kwargs.setdefault("legend", True)

    params = kwargs.get("params", None)
    is_interactive = False if params is None else True
    kwargs["is_interactive"] = is_interactive
    if is_interactive:
        from spb.interactive import iplot
        kwargs["is_vector"] = True
        return iplot(*args, **kwargs)

    series = _build_series(*args, **kwargs)
    if all([isinstance(s, (Vector2DSeries, ContourSeries)) for s in series]):
        Backend = kwargs.pop("backend", TWO_D_B)
    elif all([isinstance(s, Vector3DSeries) for s in series]):
        Backend = kwargs.pop("backend", THREE_D_B)
    else:
        raise ValueError("Mixing 2D vectors with 3D vectors is not allowed.")

    _set_labels(series, labels, rendering_kw)
    return _instantiate_backend(Backend, *series, **kwargs)
