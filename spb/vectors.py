from spb.backends.base_backend import Plot
from spb.series import (
    Vector2DSeries,
    Vector3DSeries,
    ContourSeries,
    SliceVector3DSeries,
    _set_discretization_points,
    InteractiveSeries
)
from spb.utils import _plot_sympify, _unpack_args, _split_vector, _is_range
from sympy import S, sqrt, Expr, Tuple
from sympy.geometry import Plane


def _build_series(*args, interactive=False, **kwargs):
    """Loop over args and create all the necessary series to display a vector
    plot.
    """
    series = []
    all_ranges = []
    for a in args:
        split_expr, ranges, s = _series(
            a[0], *a[1:-1], label=a[-1], interactive=interactive, **kwargs
        )
        all_ranges.append(ranges)
        if isinstance(s, (list, tuple)):
            series += s
        else:
            series.append(s)

    # add a scalar series only on 2D plots
    if all([s.is_2Dvector for s in series]):
        # don't pop this keyword: some backend needs it to decide the color
        # for quivers (solid color if a scalar field is present, gradient
        # color otherwise)
        scalar = kwargs.get("scalar", True)
        if (len(series) == 1) and (scalar is True):
            scalar_field = sqrt(split_expr[0] ** 2 + split_expr[1] ** 2)
            scalar_label = "Magnitude"
        elif scalar is True:
            scalar_field = None  # do nothing when
        elif isinstance(scalar, Expr):
            scalar_field = scalar
            scalar_label = str(scalar)
        elif isinstance(scalar, (list, tuple)):
            scalar_field = scalar[0]
            scalar_label = scalar[1]
        elif not scalar:
            scalar_field = None
        else:
            raise ValueError(
                "`scalar` must be either:\n"
                + "1. True, in which case the magnitude of the vector field "
                + "will be plotted.\n"
                + "2. a symbolic expression representing a scalar field.\n"
                + "3. None/False: do not plot any scalar field.\n"
                + "4. list/tuple of two elements, [scalar_expr, label]."
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
    fs = set().union(*[e.free_symbols for e in split_expr])
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
                ranges.append(Tuple(m, -10, 10))

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
                ranges.append(Tuple(m, -10, 10))

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
        def _check_slice(s):
            if not isinstance(s, (Expr, Plane)):
                raise ValueError(
                    "A slice must be of type Plane or Expr.\n"
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

    `matrices` and `fill_ranges` are going to be passed to `_unpack_args`.
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
        exprs, ranges, label = _unpack_args(
            *a, matrices=matrices, fill_ranges=fill_ranges
        )
        if len(exprs) == 1:
            new_args.append([*exprs, *ranges, label])
        else:
            # this is the case where the user provided: v1, v2, ..., range
            # we use the same ranges for each expression
            for e in exprs:
                new_args.append([e, *ranges, str(e)])

    return new_args


def plot_vector(*args, show=True, **kwargs):
    """
    Plot a 2D or 3D vector field. By default, the aspect ratio of the plot
    is set to `aspect="equal"`.

    Typical usage examples are in the followings:

    - Plotting a vector field with a single range.

      .. code-block::

         plot(expr, range1, range2, range3 [optional], **kwargs)

    - Plotting multiple vector fields with different ranges and custom labels.

      .. code-block::

         plot((expr1, range1, range2, range3 [optional], label1),
            (expr2, range4, range5, range6 [optional], label2), **kwargs)

    Parameters
    ==========

    args :
        expr : Vector, or Matrix/list/tuple  with 2 or 3 elements
            Represent the vector to be plotted.
            Note: if a 3D vector is given with a list/tuple, it might happens
            that the internal algorithm could think of it as a range. Therefore,
            3D vectors should be given as a Matrix or as a Vector: this reduces
            ambiguities.

        ranges : 3-element tuples
            Denotes the range of the variables. For example (x, -5, 5). For 2D
            vector plots, 2 ranges should be provided. For 3D vector plots, 3
            ranges are needed.

        label : str, optional
            The name of the vector field to be eventually shown on the legend.
            If none is provided, the string representation of the vector will
            be used.

    contours_kw : dict
        A dictionary of keywords/values which is passed to the backend
        contour function to customize the appearance. Refer to the plotting
        library (backend) manual for more informations.

    n1, n2, n3 : int
        Number of discretization points for the quivers or streamlines in the
        x/y/z-direction, respectively. Default to 25.

    n : int
        Set the same number of discretization points in all directions for
        the quivers or streamlines. It overrides `n1`, `n2`, `n3`.
        Default to 25.

    nc : int
        Number of discretization points for the scalar contour plot.
        Default to 100.

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

        Default to True.

    show : boolean
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    slice : Plane, list, Expr
        Plot the 3D vector field over the provided slice. It can be:

        - a Plane object from sympy.geometry module.
        - a list of planes.
        - a symbolic expression representing a surface of two variables.

        The number of discretization points will be `n1`, `n2`, `n3`.
        Note that:

        - only quivers plots are supported with slices. Streamlines plots
          are unaffected.
        - `n3` will only be used with planes parallel to xz or yz.

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

        >>> plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3))
        Plot object containing:
        [0]: contour: sqrt(sin(y)**2 + cos(x)**2) for x over (-3.0, 3.0) and y over (-3.0, 3.0)
        [1]: 2D vector series: [-sin(y), cos(x)] over (x, -3.0, 3.0), (y, -3.0, 3.0)

    Streamlines plot of a 2D vector field with no background scalar field.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        ...     streamlines=True, scalar=None)
        Plot object containing:
        [0]: 2D vector series: [-sin(y), cos(x)] over (x, -3.0, 3.0), (y, -3.0, 3.0)


    Plot multiple 2D vectors fields, setting a background scalar field to be
    the magnitude of the first vector.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_vector([-sin(y), cos(x)], [y, x], (x, -5, 5), (y, -3, 3), n=20,
        ...     scalar=sqrt((-sin(y))**2 + cos(x)**2), legend=True)
        Plot object containing:
        [0]: contour: sqrt(sin(y)**2 + cos(x)**2) for x over (-5.0, 5.0) and y over (-3.0, 3.0)
        [1]: 2D vector series: [-sin(y), cos(x)] over (x, -5.0, 5.0), (y, -3.0, 3.0)
        [2]: 2D vector series: [y, x] over (x, -5.0, 5.0), (y, -3.0, 3.0)

    3D vector field.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
       ...      n=8, quiver_kw={"length": 0.1},
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
       ...      xlabel="x", ylabel="y", zlabel="z")
       Plot object containing:
       [0]: sliced 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0) at Plane(Point3D(-10, 0, 0), (1, 0, 0))
       [1]: sliced 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0) at Plane(Point3D(0, 10, 0), (0, 2, 0))
       [2]: sliced 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0) at Plane(Point3D(0, 0, -10), (0, 0, 1))

    3D vector streamlines starting at a 1000 random points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector(Matrix([z, y, x]), (x, -10, 10), (y, -10, 10), (z, -10, 10),
       ...     streamlines=True,
       ...     stream_kw=dict(
       ...         starts=True,
       ...         npoints=1000
       ...     ),
       ...     xlabel="x", ylabel="y", zlabel="z")
       Plot object containing:
       [0]: 3D vector series: [z, y, x] over (x, -10.0, 10.0), (y, -10.0, 10.0), (z, -10.0, 10.0)
    """
    args = _plot_sympify(args)
    args = _preprocess(*args)

    kwargs = _set_discretization_points(kwargs, Vector3DSeries)
    kwargs.setdefault("aspect", "equal")
    kwargs.setdefault("legend", True)

    series = _build_series(*args, **kwargs)
    if all([isinstance(s, (Vector2DSeries, ContourSeries)) for s in series]):
        from spb.defaults import TWO_D_B

        backend = kwargs.pop("backend", TWO_D_B)
    elif all([isinstance(s, Vector3DSeries) for s in series]):
        from spb.defaults import THREE_D_B

        backend = kwargs.pop("backend", THREE_D_B)
    else:
        raise ValueError("Mixing 2D vectors with 3D vectors is not allowed.")

    p = Plot(*series, backend=backend, **kwargs)
    if show:
        p.show()
    return p
