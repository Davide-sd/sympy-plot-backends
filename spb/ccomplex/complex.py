from spb.defaults import cfg, TWO_D_B, THREE_D_B
from spb.functions import (
    _set_labels, _plot3d_wireframe_helper
)
from spb.series import (
    LineOver1DRangeSeries, ComplexSurfaceBaseSeries,
    ComplexPointSeries, SurfaceOver2DRangeSeries, _set_discretization_points,
    Parametric2DLineSeries, ComplexDomainColoringSeries,
    Parametric2DLineSeries, List2DSeries, GenericDataSeries,
    RiemannSphereSeries
)
from spb.interactive import create_interactive_plot
from spb.utils import (
    _unpack_args, _instantiate_backend, _plot_sympify, _check_arguments,
    _is_range, prange, _get_free_symbols
)
from spb.vectors import plot_vector
from spb.plotgrid import plotgrid
from sympy import (latex, Tuple, sqrt, re, im, arg, Expr, Dummy, symbols, I,
    sin, cos, pi)
import warnings


# NOTE:
# * `abs` refers to the absolute value;
# * `arg` refers to the argument;
# * `absarg` refers to the absolute value and argument, which will be used to
#   create "domain coloring" plots.

def _build_complex_point_series(*args, allow_lambda=False, pc=False, **kwargs):
    """The following types of arguments are supported by plot_complex_list:

    * plot_complex_list(n1, n2, ...) where `n-ith` is a complex number
    * plot_complex_list((n1, label1, rend_kw1), (n2, label2, rend_kw2), ...)
      where `n-ith` is a complex number
    * plot_complex_list(l1, l2, ...) where `l-ith` is a list of complex numbers
    * plot_complex_list((l1, label1, rend_kw1), (l2, label2, rend_kw2), ...)
      where `l-ith` is a list of complex numbers

    This function implements the above logic.

    NOTE: this logic needs to be separated from the logic behind plot_complex,
    plot_real_imag, otherwise there will be ambiguities.
    """
    series = []
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)

    if all([isinstance(a, Expr) for a in args]):
        # args is a list of complex numbers
        for a in args:
            series.append(ComplexPointSeries([a], "", **kwargs))
    elif (
        (len(args) > 0)
        and all([isinstance(a, (list, tuple, Tuple)) for a in args])
        and all([len(a) > 0 for a in args])
        and all([isinstance(a[0], (list, tuple, Tuple)) for a in args])
    ):
        # args is a list of tuples of the form (list, label, rendering_kw)
        # where list contains complex points
        for a in args:
            expr, ranges, label, rkw = _unpack_args(*a)
            # Complex points do not require ranges. However, if 3 complex
            # points are given inside a list, _unpack_args will see them as a
            # range.
            expr = expr or ranges
            kw = kwargs.copy()
            kw["rendering_kw"] = rkw
            series.append(ComplexPointSeries(expr[0], label, **kw))
    elif (
        (len(args) > 0)
        and all([isinstance(a, (list, tuple, Tuple)) for a in args])
        and all([all([isinstance(t, Expr) and t.is_complex for t in a]) for a in args])
    ):
        # args is a list of lists
        for a in args:
            series.append(ComplexPointSeries(a, "", **kwargs))
    elif (
        (len(args) > 0)
        and all([isinstance(a, (list, tuple, Tuple)) for a in args])
        and all([len(a) > 0 for a in args])
    ):
        # args is a list of tuples of the form (number, label, rendering_kw)
        # where list contains complex points
        for a in args:
            expr, ranges, label, rkw = _unpack_args(*a)
            # Complex points do not require ranges. However, if 3 complex
            # points are given inside a list, _unpack_args will see them as a
            # range.
            expr = expr or ranges
            kw = kwargs.copy()
            kw["rendering_kw"] = rkw
            series.append(ComplexPointSeries([expr[0]], label, **kw))

    else:
        expr, ranges, label, rkw = _unpack_args(*args)
        if isinstance(expr, (list, tuple, Tuple)):
            expr = expr[0]
        kw = kwargs.copy()
        kw["rendering_kw"] = rkw
        series.append(ComplexPointSeries(expr, label, **kw))

    _set_labels(series, global_labels, global_rendering_kw)
    return series


def _build_series(*args, interactive=False, allow_lambda=False, **kwargs):
    series = []
    new_args = []
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)

    # apply the proper label.
    # NOTE: the label is going to wrap the string representation of the
    # expression. This design choice precludes the ability of setting latex
    # labels, but this is not a problem as the user has the ability to set
    # a custom alias for the function to be plotted.
    mapping = {
        "real": "Re(%s)",
        "imag": "Im(%s)",
        "abs": "Abs(%s)",
        # NOTE: absarg is used to plot the absolute value colored by the
        # argument. The colorbar indicates the argument, hence the following
        # label is "Arg"
        "absarg": "Arg(%s)",
        "arg": "Arg(%s)",
    }
    # option to be used with lambdify with complex functions
    kwargs.setdefault("modules", cfg["complex"]["modules"])

    def add_series(argument):
        nexpr, npar = 1, 1
        if len([b for b in argument if _is_range(b)]) > 1:
            # function of two variables
            npar = 2
        new_args.append(_check_arguments([argument], nexpr, npar, **kwargs)[0])

    if all(isinstance(a, (list, tuple, Tuple)) for a in args):
        # deals with the case:
        # plot_complex((expr1, "name1"), (expr2, "name2"), range)
        # Modify the tuples (expr, "name") to (expr, range, "name")
        npar = len([b for b in args if _is_range(b)])
        tmp = []
        for i in range(len(args) - npar):
            a = args[i]
            tmp.append(a)
            if len(a) == 2 and isinstance(a[-1], str):
                tmp[i] = (a[0], *args[len(args) - npar:], a[-1])

        # plotting multiple expressions
        for a in tmp:
            add_series(a)
    else:
        exprs, r, label, rkw = _unpack_args(*args)
        for e in exprs:
            add_series([e, *r, label, rkw])

    params = kwargs.get("params", dict())
    for a in new_args:
        expr, ranges, label, rend_kw = a[0], a[1:-2], a[-2], a[-1]
        if label is None:
            label = str(expr)

        kw = kwargs.copy()
        kw["rendering_kw"] = rend_kw
        if (not allow_lambda) and callable(expr):
            raise TypeError("expr must be a symbolic expression.")

        # NOTE:
        # 1. as a design choice, a complex function will create one
        #    or more data series, depending on the keyword arguments
        #    (one for the real part, one for the imaginary part, etc.).
        #    This is undoubtely inefficient as we must evaluate the same
        #    expression multiple times. On the other hand, it allows to
        #    maintain a one-to-one correspondance between Plot.series
        #    and backend.data, making it easier to work with interactive
        #    widgets plot.
        # 2. The expression used on each data series is the same one
        #    provided by the user. Each data series will receive the `return`
        #    keyword argument, which specify what data must be returned.
        #    So, if return="real", the series will return the real part
        #    of the function, and so on.
        #    Why not applying SymPy's re(), im(), arg(), ..., to the original
        #    expression and get rid of `return`? Because `re()` and `im()`
        #    evaluate the expression, usually creating new expressions
        #    containing many more terms, hence much slower evaluation. Instead,
        #    the series are going to evaluate the complex function and then
        #    extract the required data.

        absarg = kw.pop("absarg", True)
        real = kw.pop("real", False)
        imag = kw.pop("imag", False)
        _abs = kw.pop("abs", False)
        _arg = kw.pop("arg", False)

        # if ranges[0][1].imag == ranges[0][2].imag:
        if im(ranges[0][1]) == im(ranges[0][2]):
            # dealing with lines
            def add_series(flag, key):
                if flag:
                    kw2 = kw.copy()
                    kw2[key] = True
                    kw2["return"] = key
                    lbl_wrapper = mapping[key]
                    series.append(LineOver1DRangeSeries(expr, *ranges,
                        lbl_wrapper % label, **kw2))

        else:
            # 2D domain coloring or 3D plots
            kw.setdefault("coloring", cfg["complex"]["coloring"])
            def add_series(flag, key):
                if flag:
                    kw2 = kw.copy()
                    kw2[key] = True
                    kw2["return"] = key
                    lbl_wrapper = mapping[key]
                    if key == "absarg":
                        lbl_wrapper = "%s"
                    series.append(ComplexSurfaceBaseSeries(expr, *ranges,
                        lbl_wrapper % label, **kw2))

        add_series(absarg, "absarg")
        add_series(real, "real")
        add_series(imag, "imag")
        add_series(_abs, "abs")
        add_series(_arg, "arg")

    _set_labels(series, global_labels, global_rendering_kw)
    series += _plot3d_wireframe_helper(series, **kwargs)
    return series


def _plot_complex(*args, allow_lambda=False, pcl=False, **kwargs):
    """Create the series and setup the backend."""
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, ComplexSurfaceBaseSeries)

    if not pcl:
        series = _build_series(*args, allow_lambda=allow_lambda, **kwargs)
    else:
        series = _build_complex_point_series(*args, allow_lambda=allow_lambda, pcl=True, **kwargs)

    if len(series) == 0:
        warnings.warn("No series found. Check your keyword arguments.")

    _set_axis_labels(series, kwargs)

    if any(s.is_3Dsurface for s in series):
        Backend = kwargs.get("backend", THREE_D_B)
    else:
        Backend = kwargs.get("backend", TWO_D_B)

    if any(s.is_domain_coloring for s in series):
        kwargs.setdefault("legend", True)

        dc_2d_series = [s for s in series if s.is_domain_coloring and not s.is_3D]
        if ((len(dc_2d_series) > 0) and kwargs.get("riemann_mask", False)):
            # ask the backend to add annotations on unit circle in the complex
            # plane. We can't do it here because each backend requires
            # different data format.
            kwargs.setdefault("aouc", any(s.annotate for s in dc_2d_series))
            # add unit circle: hide it from legend and requests its color
            # to be black
            t = symbols("t")
            series.append(
                Parametric2DLineSeries(cos(t), sin(t), (t, 0, 2*pi), "__k__",
                    adaptive=False, n=1000, use_cm=False,
                    show_in_legend=False))


    if kwargs.get("params", None):
        return create_interactive_plot(*series, **kwargs)

    return _instantiate_backend(Backend, *series, **kwargs)


def _set_axis_labels(series, kwargs):
    """Set the axis labels for the plot, depending on the series being
    visualized.
    """
    if all(s.is_parametric for s in series):
        if kwargs.get("xlabel", None) is None:
            kwargs["xlabel"] = "Real"
        if kwargs.get("ylabel", None) is None:
            kwargs["ylabel"] = "Abs"
    elif all(s.is_domain_coloring or s.is_3Dsurface or
        isinstance(s, ComplexPointSeries) or
        s.is_parametric for s in series):
        # when plotting real/imaginary or domain coloring/3D plots, the
        # horizontal axis is the real, the vertical axis is the imaginary
        if kwargs.get("xlabel", None) is None:
            kwargs["xlabel"] = "Re"
        if kwargs.get("ylabel", None) is None:
            kwargs["ylabel"] = "Im"
        if kwargs.get("zlabel", None) is None and any(s.is_domain_coloring for s in series):
            kwargs["zlabel"] = "Abs"
    else:
        var = series[0].var

        if kwargs.get("xlabel", None) is None:
            fx = lambda use_latex: var.name if not use_latex else latex(var)
            kwargs.setdefault("xlabel", fx)
        if kwargs.get("ylabel", None) is None:
            wrap = lambda use_latex: "f(%s)" if not use_latex else r"f\left(%s\right)"
            x = kwargs["xlabel"] if callable(kwargs["xlabel"]) else lambda use_latex: kwargs["xlabel"]
            fy = lambda use_latex: wrap(use_latex) % x(use_latex)
            kwargs.setdefault("ylabel", fy)

    if (kwargs.get("aspect", None) is None) and any(
        (s.is_complex and s.is_domain_coloring and (not s.is_3D)) or s.is_point
        for s in series):
        # set aspect equal for 2D domain coloring or complex points
        kwargs.setdefault("aspect", "equal")


def plot_real_imag(*args, **kwargs):
    """Plot the real part, the imaginary parts, the absolute value and the
    argument of a complex function. By default, only the real and imaginary
    parts will be plotted. Use keyword argument to be more specific.
    By default, the aspect ratio of 2D plots is set to ``aspect="equal"``.

    Depending on the provided expression, this function will produce different
    types of plots:

    1. line plot over the reals.
    2. surface plot over the complex plane if `threed=True`.
    3. contour plot over the complex plane if `threed=False`.

    Typical usage examples are in the followings:

    - Plotting a single expression with a single range.
        `plot_real_imag(expr, range, **kwargs)`
    - Plotting a single expression with the default range (-10, 10).
        `plot_real_imag(expr, **kwargs)`
    - Plotting multiple expressions with a single range.
        `plot_real_imag(expr1, expr2, ..., range, **kwargs)`
    - Plotting multiple expressions with multiple ranges.
        `plot_real_imag((expr1, range1), (expr2, range2), ..., **kwargs)`
    - Plotting multiple expressions with custom labels and rendering options.
        `plot_real_imag((expr1, range1, label1, rendering_kw1), (expr2, range2, label2, rendering_kw2), ..., **kwargs)`

    Parameters
    ==========
    args :
        expr : Expr
            Represent the complex function to be plotted.

        range : 3-element tuple
            Denotes the range of the variables. For example:

            * ``(z, -5, 5)``: plot a line over the reals from point `-5` to
              `5`
            * ``(z, -5 + 2*I, 5 + 2*I)``: plot a line from complex point
              ``(-5 + 2*I)`` to ``(5 + 2 * I)``. Note the same imaginary part
              for the start/end point. Also note that we can specify the
              ranges by using standard Python complex numbers, for example
              ``(z, -5+2j, 5+2j)``.
            * ``(z, -5 - 3*I, 5 + 3*I)``: surface or contour plot of the
              complex function over the specified domain using a rectangular
              discretization.

        label : str, optional
            The name of the complex function to be eventually shown on the
            legend. If none is provided, the string representation of the
            function will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations. Note that
            the same options will be applied to all series generated for the
            specified expression.

    abs : boolean, optional
        If True, plot the modulus of the complex function. Default to False.

    adaptive : bool, optional
        If ``True``, creates line plots by using an adaptive algorithm.
        Use ``adaptive_goal`` and ``loss_fn`` to further customize the output.
        Image and surface plots do not use an adaptive algorithm.

        Default to ``False``, which uses a uniform sampling strategy.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * ``None`` (default):  it will use the following goal:
          ``lambda l: l.loss() < 0.01``
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          ``lambda l: l.loss() < number``
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn0]_ for more information.

    arg : boolean, optional
        If True, plot the argument of the complex function. Default to False.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    detect_poles : boolean, optional
        Chose whether to detect and correctly plot poles. Defaulto to False.
        It only works with line plots. To improve detection, increase the
        number of discretization points if ``adaptive=False`` and/or change
        the value of ``eps``.

    eps : float, optional
        An arbitrary small value used by the ``detect_poles`` algorithm.
        Default value to 0.1. Before changing this value, it is better to
        increase the number of discretization points.

    imag : boolean, optional
        If True, plot the imaginary part of the complex function.
        Default to True.

    label : list/tuple, optional
        The labels to be shown in the legend. If not provided, the string
        representation of ``expr`` will be used. The number of labels must be
        equal to the number of series generated by the plotting function.

    loss_fn : callable or None
        The loss function to be used by the adaptive learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          ``adaptive`` module.
        * callable : Refer to [#fn0]_ for more information. Specifically,
          look at ``adaptive.learner.learner1D`` to find more loss functions.

    modules : str, optional
        Specify the modules to be used for the numerical evaluation. Refer to
        ``lambdify`` to visualize the available options. Default to None,
        meaning Numpy/Scipy will be used. Note that other modules might
        produce different results, based on the way they deal with branch
        cuts.

    n1, n2 : int, optional
        Number of discretization points in the real/imaginary-directions,
        respectively, when `adaptive=False`. For line plots, default to 1000.
        For surface/contour plots (2D and 3D), default to 300.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, set the same number of discretization
        points in all directions. If a tuple is provided, it overrides
        ``n1`` and ``n2``. It only works when ``adaptive=False``.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of the interactive sub-module.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of series generated by the plotting function.

    real : boolean, optional
        If True, plot the real part of the complex function. Default to True.

    show : boolean, optional
        Default to True, in which case the plot will be shown on the screen.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    surface_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of surfaces. Refer to the
        plotting library (backend) manual for more informations.

    threed : boolean, optional
        It only applies to a complex function over a complex range. If False,
        contour plots will be shown. If True, 3D surfaces will be shown.
        Default to False.

    use_cm : boolean, optional
        If False, surfaces will be rendered with a solid color.
        If True, a color map highlighting the elevation will be used.
        Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    wireframe : boolean, optional
        Enable or disable a wireframe over the 3D surface. Depending on the
        number of wireframe lines (see ``wf_n1`` and ``wf_n2``), activating
        thisoption might add a considerable overhead during the plot's
        creation. Default to False (disabled).

    wf_n1, wf_n2 : int, optional
        Number of wireframe lines along the x and y ranges, respectively.
        Default to 10. Note that increasing this number might considerably
        slow down the plot's creation.

    wf_npoint : int or None, optional
        Number of discretization points for the wireframe lines. Default to
        None, meaning that each wireframe line will have ``n1`` or ``n2``
        number of points, depending on the line direction.

    wf_rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of wireframe lines.

    xlabel, ylabel, zlabel : str, optional
        Labels for the x-axis, y-axis or z-axis, respectively.
        ``zlabel`` is only available for 3D plots.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim, ylim, zlim : (float, float), optional
        Denotes the x-axis limits, y-axis limits or z-axis limits,
        respectively, ``(min, max)``. ``zlim`` is only available for 3D plots.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import I, symbols, exp, sqrt, cos, sin, pi, gamma
       >>> from spb import plot_real_imag
       >>> x, y, z = symbols('x, y, z')


    Plot the real and imaginary parts of a function over reals:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_real_imag(sqrt(x), (x, -3, 3))
       Plot object containing:
       [0]: cartesian line: re(sqrt(x)) for x over (-3.0, 3.0)
       [1]: cartesian line: im(sqrt(x)) for x over (-3.0, 3.0)

    Plot only the real part:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_real_imag(sqrt(x), (x, -3, 3), imag=False)
       Plot object containing:
       [0]: cartesian line: re(sqrt(x)) for x over (-3.0, 3.0)

    Plot only the imaginary part:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_real_imag(sqrt(x), (x, -3, 3), real=False)
       Plot object containing:
       [0]: cartesian line: im(sqrt(x)) for x over (-3.0, 3.0)

    Plot only the absolute value and argument:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_real_imag(sqrt(x), (x, -3, 3), real=False, imag=False, abs=True, arg=True)
       Plot object containing:
       [0]: cartesian line: abs(sqrt(x)) for x over (-3.0, 3.0)
       [1]: cartesian line: arg(sqrt(x)) for x over (-3.0, 3.0)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * for 1D ``plot_real_imag``, symbols going into ``prange`` must be real.
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       x, u = symbols("x, u")
       a = symbols("a", real=True)
       plot_real_imag(sqrt(x) * exp(-u * x**2), prange(x, -3*a, 3*a),
           params={u: (1, 0, 2), a: (1, 0, 2)},
           ylim=(-0.25, 2), use_latex=False)

    3D plot of the real and imaginary part of the principal branch of a
    function over a complex range. Note the jump in the imaginary part: that's
    a branch cut. The rectangular discretization is unable to properly capture
    it, hence the near vertical wall. Refer to ``plot3d_parametric_surface``
    for an example about plotting Riemann surfaces and properly capture
    the branch cuts.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_real_imag(sqrt(x), (x, -3-3j, 3+3j), n=100, threed=True,
       ...      use_cm=True)
       Plot object containing:
       [0]: complex cartesian surface: re(sqrt(x)) for re(x) over (-3.0, 3.0) and im(x) over (-3.0, 3.0)
       [1]: complex cartesian surface: im(sqrt(x)) for re(x) over (-3.0, 3.0) and im(x) over (-3.0, 3.0)

    3D plot of the absolute value of a function over a complex range:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_real_imag(sqrt(x), (x, -3-3j, 3+3j),
       ...     n=100, real=False, imag=False, abs=True, threed=True)
       Plot object containing:
       [0]: complex cartesian surface: abs(sqrt(x)) for re(x) over (-3.0, 3.0) and im(x) over (-3.0, 3.0)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       x, u, a, b = symbols("x, u, a, b")
       plot_real_imag(
           sqrt(x) * exp(u * x), prange(x, -3*a-b*3j, 3*a+b*3j),
           backend=PB, aspect="cube",
           wireframe=True, wf_rendering_kw={"line_width": 1},
           params={
               u: (0.25, 0, 1),
               a: (1, 0, 2),
               b: (1, 0, 2)
           }, n=25, threed=True, use_latex=False, use_cm=True)

    References
    ==========

    .. [#fn0] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot_complex, plot_complex_list, plot_complex_vector

    """
    kwargs["absarg"] = False
    kwargs.setdefault("abs", False)
    kwargs.setdefault("arg", False)
    kwargs.setdefault("real", True)
    kwargs.setdefault("imag", True)
    return _plot_complex(*args, **kwargs)


def plot_complex(*args, **kwargs):
    """Plot the absolute value of a complex function colored by its argument.
    By default, the aspect ratio of 2D plots is set to ``aspect="equal"``.

    Depending on the provided range, this function will produce different
    types of plots:

    1. Line plot over the reals.
    2. Image plot over the complex plane if ``threed=False``. This is also
       known as Domain Coloring. Use the ``coloring`` keyword argument to
       select a different coloring strategy and ``cmap`` to set a custom
       color map (default to HSV).
    3. If ``threed=True``, plot a 3D surface of the absolute value over the
       complex plane, colored by its argument. Use the ``coloring`` keyword
       argument to select a different coloring strategy and ``cmap`` to set
       a custom color map (default to HSV).

    Read the `Notes` section to learn more about colormaps.

    Typical usage examples are in the followings:

    - Plotting a single expression with a single range.
        `plot_complex(expr, range, **kwargs)`
    - Plotting a single expression with the default range (-10, 10).
        `plot_complex(expr, **kwargs)`
    - Plotting multiple expressions with a single range.
        `plot_complex(expr1, expr2, ..., range, **kwargs)`
    - Plotting multiple expressions with multiple ranges.
        `plot_complex((expr1, range1), (expr2, range2), ..., **kwargs)`
    - Plotting multiple expressions with custom labels and rendering options.
        `plot_complex((expr1, range1, label1, rendering_kw1), (expr2, range2, label2, rendering_kw2), ..., **kwargs)`

    Parameters
    ==========
    args :
        expr : Expr or callable
            Represent the complex function to be plotted. It can be a:

            * Symbolic expression.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        range : 3-element tuple
            Denotes the range of the variables. For example:

            * ``(z, -5, 5)``: plot a line over the reals from point `-5` to
              `5`
            * ``(z, -5 + 2*I, 5 + 2*I)``: plot a line from complex point
              ``(-5 + 2*I)`` to ``(5 + 2 * I)``. Note the same imaginary part
              for the start/end point. Also note that we can specify the
              ranges by using standard Python complex numbers, for example
              ``(z, -5+2j, 5+2j)``.
            * ``(z, -5 - 3*I, 5 + 3*I)``: surface or contour plot of the
              complex function over the specified domain.

        label : str, optional
            The name of the complex function to be eventually shown on the
            legend. If none is provided, the string representation of the
            function will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines, surfaces or images.
            Refer to the plotting library (backend) manual for more
            informations.

    adaptive : bool, optional
        If ``True``, creates line plots by using an adaptive algorithm.
        Use ``adaptive_goal`` and ``loss_fn`` to further customize the output.
        Image and surface plots do not use an adaptive algorithm.

        Default to ``False``, which uses a uniform sampling strategy.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * ``None`` (default):  it will use the following goal:
          ``lambda l: l.loss() < 0.01``
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          ``lambda l: l.loss() < number``
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn2]_ for more information.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    at_infinity : boolean, optional
        Apply the transformation $z \\rightarrow \\frac{1}{z}$ in order to
        study the behaviour of the function around the point at infinity.
        It is recommended to also set ``show_axis=False`` in order to avoid
        confusion.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    blevel : float, optional
        Controls the black level of ehanced domain coloring plots. It must be
        `0 (black) <= blevel <= 1 (white)`. Default to 0.75.

    cmap : str, iterable, optional
        Specify the colormap to be used on enhanced domain coloring plots
        (both images and 3d plots). Default to ``"hsv"``. Can be any colormap
        from matplotlib or colorcet.

    coloring : str or callable, optional
        Choose between different domain coloring options. Default to ``"a"``.
        Refer to [#fn1]_ for more information.

        - ``"a"``: standard domain coloring showing the argument of the
          complex function.
        - ``"b"``: enhanced domain coloring showing iso-modulus and iso-phase
          lines.
        - ``"c"``: enhanced domain coloring showing iso-modulus lines.
        - ``"d"``: enhanced domain coloring showing iso-phase lines.
        - ``"e"``: alternating black and white stripes corresponding to
          modulus.
        - ``"f"``: alternating black and white stripes corresponding to
          phase.
        - ``"g"``: alternating black and white stripes corresponding to
          real part.
        - ``"h"``: alternating black and white stripes corresponding to
          imaginary part.
        - ``"i"``: cartesian chessboard on the complex points space. The
          result will hide zeros.
        - ``"j"``: polar Chessboard on the complex points space. The result
          will show conformality.
        - ``"k"``: black and white magnitude of the complex function.
          Zeros are black, poles are white.
        - ``"l"``:enhanced domain coloring showing iso-modulus and iso-phase
          lines, blended with the magnitude: white regions indicates greater
          magnitudes. Can be used to distinguish poles from zeros.
        - ``"m"``: enhanced domain coloring showing iso-modulus lines, blended
          with the magnitude: white regions indicates greater magnitudes.
          Can be used to distinguish poles from zeros.
        - ``"n"``: enhanced domain coloring showing iso-phase lines, blended
          with the magnitude: white regions indicates greater magnitudes.
          Can be used to distinguish poles from zeros.
        - ``"o"``: enhanced domain coloring showing iso-phase lines, blended
          with the magnitude: white regions indicates greater magnitudes.
          Can be used to distinguish poles from zeros.

        The user can also provide a callable, ``f(w)``, where ``w`` is an
        [n x m] Numpy array (provided by the plotting module) containing
        the results (complex numbers) of the evaluation of the complex
        function. The callable should return:

        - img : ndarray [n x m x 3]
            An array of RGB colors (0 <= R,G,B <= 255)
        - colorscale : ndarray [N x 3] or None
            An array with N RGB colors, (0 <= R,G,B <= 255).
            If ``colorscale=None``, no color bar will be shown on the plot.

    label : str or list/tuple, optional
        The label to be shown in the legend or colorbar in case of a line plot.
        If not provided, the string representation of ``expr`` will be used.
        The number of labels must be  equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the adaptive learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          ``adaptive`` module.
        * callable : Refer to [#fn2]_ for more information. Specifically,
          look at `adaptive.learner.learner1D` to find more loss functions.

    modules : str, optional
        Specify the modules to be used for the numerical evaluation. Refer to
        ``lambdify`` to visualize the available options. Default to None,
        meaning Numpy/Scipy will be used. Note that other modules might
        produce different results, based on the way they deal with branch
        cuts.

    n1, n2 : int, optional
        Number of discretization points in the real/imaginary-directions,
        respectively, when ``adaptive=False``. For line plots, default to 1000.
        For surface/contour plots (2D and 3D), default to 300.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, set the same number of discretization
        points in all directions. If a tuple is provided, it overrides
        ``n1`` and ``n2``. It only works when ``adaptive=False``.

    params : dict, optional
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of the interactive sub-module.

    phaseres : int, optional
        Default value to 20. It controls the number of iso-phase and/or
        iso-modulus lines in domain coloring plots.

    phaseoffset : float, optional
        Controls the phase offset of the colormap in domain coloring plots.
        Default to 0.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines, surfaces or images.
        Refer to the plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of series generated by the plotting function.

    show : boolean, optional
        Default to True, in which case the plot will be shown on the screen.

    show_axis : boolean, optional
        Turn on/off the axis of the plot. Default to True (axis are visible).

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the backend.

    threed : boolean, optional
        It only applies to a complex function over a complex range. If False,
        a 2D image plot will be shown. If True, 3D surfaces will be shown.
        Default to False.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel, zlabel : str, optional
        Labels for the x-axis, y-axis or z-axis, respectively.
        ``zlabel`` is only available for 3D plots.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim, ylim, zlim : (float, float), optional
        Denotes the x-axis limits, y-axis limits or z-axis limits,
        respectively, ``(min, max)``. ``zlim`` is only available for 3D plots.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import I, symbols, exp, sqrt, cos, sin, pi, gamma
       >>> from spb import plot_complex
       >>> x, y, z = symbols('x, y, z')

    Plot the modulus of a complex function colored by its magnitude:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_complex(cos(x) + sin(I * x), "f", (x, -2, 2))
       Plot object containing:
       [0]: cartesian abs-arg line: cos(x) + I*sinh(x) for x over ((-2+0j), (2+0j))

    Interactive-widget plot of a Fourier Transform. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * for ``plot_complex``, symbols going into ``prange`` must be real.
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       x, k, a, b = symbols("x, k, a, b")
       c = symbols("c", real=True)
       f = exp(-x**2) * (Heaviside(x + a) - Heaviside(x - b))
       fs = fourier_transform(f, x, k)
       plot_complex(fs, prange(k, -c, c),
               params={a: (1, -2, 2), b: (-2, -2, 2), c: (4, 0.5, 4)},
               label="Arg(fs)", xlabel="k", yscale="log", ylim=(1e-03, 10),
               use_latex=False)

    Domain coloring plot. To improve the smoothness of the results, increase
    the number of discretization points and/or apply an interpolation (if the
    backend supports it):

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_complex(gamma(z), (z, -3-3j, 3+3j),
       ...     coloring="b", n=500, grid=False)
       Plot object containing:
       [0]: complex domain coloring: gamma(z) for re(z) over (-3.0, 3.0) and im(z) over (-3.0, 3.0)

    Domain coloring of the same function evaluated near the point
    $z=\\infty$:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_complex(gamma(z), (z, -1-1j, 1+1j), coloring="b", n=500,
       ...     grid=False, at_infinity=True, show_axis=False)
       Plot object containing:
       [0]: complex domain coloring: gamma(1/z) for re(z) over (-1.0, 1.0) and im(z) over (-1.0, 1.0)

    Interactive-widget domain coloring plot. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * setting a custom colormap and adjusting the black-level of the enhanced
      visualization.
    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       import colorcet
       z, u, a, b = symbols("z, u, a, b")
       plot_complex(
           sin(u * z), prange(z, -a - b*I, a + b*I),
           cmap=colorcet.colorwheel, blevel=0.85, use_latex=False,
           coloring="b", n=250, grid=False,
           params={
               u: (0.5, 0, 2),
               a: (pi, 0, 2*pi),
               b: (pi, 0, 2*pi),
           })

    The analytic landscape is 3D plot of the absolute value of a complex
    function colored by its argument:

    .. plotly::
       :context: reset

       from sympy import symbols, gamma, I
       from spb import plot_complex, PB
       z = symbols('z')
       plot_complex(gamma(z), (z, -3 - 3*I, 3 + 3*I), threed=True,
           backend=PB, zlim=(-1, 6), use_cm=True)

    Because the function goes to infinity at poles, sometimes it might be
    beneficial to visualize the logarithm of the absolute value in order to
    easily identify zeros:

    .. k3d-screenshot::
       :camera: -4.28, 6.55, 4.83, 0.13, -0.20, 1.9, 0.16, -0.24, 0.96

       from sympy import symbols, I
       from spb import plot_complex, KB
       z = symbols("z")
       expr = (z**3 - 5) / z
       plot_complex(expr, (z, -3-3j, 3+3j), coloring="b", threed=True,
           use_cm=True, grid=False, n=500, backend=KB, tz=np.log)


    Notes
    =====

    By default, a domain coloring plot will show the phase portrait: each point
    of the complex plane is color-coded according to its argument. The default
    colormap is HSV, which is characterized by 2 important problems:

    * It is not friendly to people affected by color deficiencies.
    * Because it isn't perceptually uniform, it might be misleading: features
      disappear at points of low perceptual contrast, or false features appear
      that are in the colormap but not in the data (refer to colorcet [#fn3]_
      for more information).

    Hence, it might be helpful to chose a perceptually uniform colormap.
    Domaing coloring plots are naturally suited to be represented by cyclic
    colormaps, but sequential colormaps can be used too. In the following
    example we illustrate the phase portrait of `f(z) = z` using different
    colormaps:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import symbols, pi
       import colorcet
       from spb import *

       z = symbols("z")
       cmaps = {
           "hsv": "hsv",
           "twilight": "twilight",
           "colorwheel": colorcet.colorwheel,
           "CET-C7": colorcet.CET_C7,
           "viridis": "viridis"
       }
       plots = []
       for k, v in cmaps.items():
           plots.append(
               plot_complex(z, (z, -2-2j, 2+2j), coloring="a",
                   grid=False, show=False, legend=True, cmap=v, title=k))

       fig = plotgrid(*plots, nc=2, show=False)
       fig.set_size_inches(6.5, 8)
       fig.tight_layout()
       fig

    In the above figure, when using the HSV colormap the eye is drawn to
    the yellow, cyan and magenta colors, where there is a lightness gradient:
    those are false features caused by the colormap. Indeed, there is nothing
    going on these regions when looking with a perceptually uniform colormap.

    Phase is computed with Numpy: it lies between [-pi, pi]. Then, phase is
    normalized between [0, 1] using `(arg / (2 * pi)) % 1`. The figure
    below shows the mapping between phase in radians and normalized phase.
    A phase of 0 radians corresponds to a normalized phase of 0, which gets
    mapped to the beginning of a colormap.

    .. plot:: ./modules/plot_complex_explanation.py
       :context: close-figs
       :include-source: False

    The zero radians phase is then located in the middle of the colorbar.
    Hence, the colorbar might feel "weird" if a sequential colormap is chosen,
    because there is a color-discontinuity in the middle of it, as can be seen
    in the previous example.
    The ``phaseoffset`` keyword argument allows to adjust the position of
    the colormap:

    .. plot::
       :context: close-figs
       :include-source: True

       p1 = plot_complex(
           z, (z, -2-2j, 2+2j), grid=False, show=False, legend=True,
           coloring="a", cmap="viridis", phaseoffset=0,
           title="phase offset = 0")
       p2 = plot_complex(
           z, (z, -2-2j, 2+2j), grid=False, show=False, legend=True,
           coloring="a", cmap="viridis", phaseoffset=pi,
           title=r"phase offset = $\pi$")
       fig = plotgrid(p1, p2, nc=2, show=False)
       fig.set_size_inches(6, 2)
       fig

    A pure phase portrait is rarely useful, as it conveys too little
    information. Let's now quickly visualize the different ``coloring``
    schemes. In the following, `arg` is the argument (phase), `mag` is the
    magnitude (absolute value) and `contour` is a line of constant value.
    Refer to [#fn1]_ for more information.

    .. plot::
       :context: close-figs
       :include-source: True

       from matplotlib import rcParams
       rcParams["font.size"] = 8
       colorings = "abcdlmnoefghijk"
       titles = [
           "phase portrait", "mag + arg contours", "mag contours", "arg contours",
           "'a' + poles", "'b' + poles", "'c' + poles", "'d' + poles",
           "mag stripes", "arg stripes", "real part stripes", "imag part stripes",
           "hide zeros", "conformality", "magnitude"]
       plots = []
       expr = (z - 1) / (z**2 + z + 1)
       for c, t in zip(colorings, titles):
           plots.append(
               plot_complex(expr, (z, -2-2j, 2+2j), coloring=c,
                   grid=False, show=False, legend=False, cmap=colorcet.CET_C7,
                   title=("'%s'" % c) + ": " + t, xlabel="", ylabel=""))

       fig = plotgrid(*plots, nc=4, show=False)
       fig.set_size_inches(8, 8.5)
       fig.tight_layout()
       fig

    From the above picture, we can see that:

    * Some enhancements decrese the lighness of the colors: depending on the
      colormap, it might be difficult to distinguish features in darker
      regions.
    * Other enhancements increases the lightness in proximity of poles. Hence,
      colormaps with very light colors might not convey enough information.

    The selection of a proper colormap is left to the user because not only
    it depends on the target audience of the visualization, but also on the
    function being visualized.

    References
    ==========

    .. [#fn1] Domain Coloring is based on Elias Wegert's book
       `"Visual Complex Functions" <https://www.springer.com/de/book/9783034801799>`_.
       The book provides the background to better understand the images.

    .. [#fn2] https://github.com/python-adaptive/adaptive

    .. [#fn3] https://colorcet.com/

    See Also
    ========

    plot_riemann_sphere, plot_real_imag, plot_complex_list, plot_complex_vector

    """
    kwargs["absarg"] = True
    kwargs["real"] = False
    kwargs["imag"] = False
    kwargs["abs"] = False
    kwargs["arg"] = False
    return _plot_complex(*args, allow_lambda=True, **kwargs)


def plot_complex_list(*args, **kwargs):
    """Plot lists of complex points. By default, the aspect ratio of the plot
    is set to ``aspect="equal"``.

    Typical usage examples are in the followings:

    - Plotting a single list of complex numbers.
        `plot_complex_list(l1, **kwargs)`
    - Plotting multiple lists of complex numbers.
        `plot_complex_list(l1, l2, **kwargs)`
    - Plotting multiple lists of complex numbers each one with a custom label.
        `plot_complex_list((l1, label1), (l2, label2), **kwargs)`


    Parameters
    ==========
    args :
        numbers : list, tuple
            A list of complex numbers.

        label : str
            The name associated to the list of the complex numbers to be
            eventually shown on the legend. Default to empty string.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations. Note that
            the same options will be applied to all series generated for the
            specified expression.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    is_point : boolean
        If True, a scatter plot will be produced. Otherwise a line plot will
        be created. Default to True.

    is_filled : boolean, optional
        Default to True, which will render empty circular markers. It only
        works if ``is_point=True``.
        If False, filled circular markers will be rendered.

    label : str or list/tuple, optional
        The name associated to the list of the complex numbers to be
        eventually shown on the legend. The number of labels must be equal to
        the number of series generated by the plotting function.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of the interactive sub-module.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of series generated by the plotting function.

    show : boolean
        Default to True, in which case the plot will be shown on the screen.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel : str, optional
        Labels for the x-axis or y-axis, respectively.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim, ylim : (float, float), optional
        Denotes the x-axis limits or y-axis limits, respectively,
        ``(min, max)``.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import I, symbols, exp, sqrt, cos, sin, pi, gamma
       >>> from spb import plot_complex_list
       >>> x, y, z = symbols('x, y, z')

    Plot individual complex points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_complex_list(3 + 2 * I, 4 * I, 2)
       Plot object containing:
       [0]: complex points: (3 + 2*I,)
       [1]: complex points: (4*I,)
       [2]: complex points: (2,)

    Plot two lists of complex points and assign to them custom labels:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr1 = z * exp(2 * pi * I * z)
       >>> expr2 = 2 * expr1
       >>> n = 15
       >>> l1 = [expr1.subs(z, t / n) for t in range(n)]
       >>> l2 = [expr2.subs(z, t / n) for t in range(n)]
       >>> plot_complex_list((l1, "f1"), (l2, "f2"))
       Plot object containing:
       [0]: complex points: (0.0, 0.0666666666666667*exp(0.133333333333333*I*pi), 0.133333333333333*exp(0.266666666666667*I*pi), 0.2*exp(0.4*I*pi), 0.266666666666667*exp(0.533333333333333*I*pi), 0.333333333333333*exp(0.666666666666667*I*pi), 0.4*exp(0.8*I*pi), 0.466666666666667*exp(0.933333333333333*I*pi), 0.533333333333333*exp(1.06666666666667*I*pi), 0.6*exp(1.2*I*pi), 0.666666666666667*exp(1.33333333333333*I*pi), 0.733333333333333*exp(1.46666666666667*I*pi), 0.8*exp(1.6*I*pi), 0.866666666666667*exp(1.73333333333333*I*pi), 0.933333333333333*exp(1.86666666666667*I*pi))
       [1]: complex points: (0, 0.133333333333333*exp(0.133333333333333*I*pi), 0.266666666666667*exp(0.266666666666667*I*pi), 0.4*exp(0.4*I*pi), 0.533333333333333*exp(0.533333333333333*I*pi), 0.666666666666667*exp(0.666666666666667*I*pi), 0.8*exp(0.8*I*pi), 0.933333333333333*exp(0.933333333333333*I*pi), 1.06666666666667*exp(1.06666666666667*I*pi), 1.2*exp(1.2*I*pi), 1.33333333333333*exp(1.33333333333333*I*pi), 1.46666666666667*exp(1.46666666666667*I*pi), 1.6*exp(1.6*I*pi), 1.73333333333333*exp(1.73333333333333*I*pi), 1.86666666666667*exp(1.86666666666667*I*pi))

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary.

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       z, u = symbols("z u")
       expr1 = z * exp(2 * pi * I * z)
       expr2 = u * expr1
       n = 15
       l1 = [expr1.subs(z, t / n) for t in range(n)]
       l2 = [expr2.subs(z, t / n) for t in range(n)]
       plot_complex_list(
           (l1, "f1"), (l2, "f2"),
           params={u: (0.5, 0, 2)}, use_latex=False,
           xlim=(-1.5, 2), ylim=(-2, 1))


    See Also
    ========

    plot_real_imag, plot_complex, plot_complex_vector

    """
    kwargs["absarg"] = False
    kwargs["abs"] = False
    kwargs["arg"] = False
    kwargs["real"] = False
    kwargs["imag"] = False
    kwargs["threed"] = False
    return _plot_complex(*args, allow_lambda=False, pcl=True, **kwargs)


def plot_complex_vector(*args, **kwargs):
    """Plot the vector field `[re(f), im(f)]` for a complex function `f`
    over the specified complex domain. By default, the aspect ratio of 2D
    plots is set to ``aspect="equal"``.

    Typical usage examples are in the followings:

    - Plotting a vector field of a complex function.
        `plot_complex_vector(expr, range, **kwargs)`

    - Plotting multiple vector fields with different ranges and custom labels.
        `plot_complex_vector((expr1, range1, label1 [optional]), (expr2, range2, label2 [optional]), **kwargs)`

    Parameters
    ==========

    args :
        expr : Expr
            Represent the complex function.

        range : 3-element tuples
            Denotes the range of the variables. For example
            ``(z, -5 - 3*I, 5 + 3*I)``. Note that we can specify the range
            by using standard Python complex numbers, for example
            ``(z, -5-3j, 5+3j)``.

        label : str, optional
            The name of the complex expression to be eventually shown on the
            legend. If none is provided, the string representation of the
            expression will be used.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    contours_kw : dict
        A dictionary of keywords/values which is passed to the backend
        contour function to customize the appearance. Refer to the plotting
        library (backend) manual for more informations.

    n1, n2 : int
        Number of discretization points for the quivers or streamlines in the
        x/y-direction, respectively. Default to 25.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, set the same number of discretization
        points in all directions for quivers or streamlines. If a tuple is
        provided, it overrides ``n1`` and ``n2``. It only works when
        ``adaptive=False``. Default to 25.

    nc : int
        Number of discretization points for the scalar contour plot.
        Default to 100.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of the interactive sub-module.

    quiver_kw : dict
        A dictionary of keywords/values which is passed to the backend
        quivers-plotting function to customize the appearance. Refer to the
        plotting library (backend) manual for more informations.

    scalar : boolean, Expr, None or list/tuple of 2 elements
        Represents the scalar field to be plotted in the background of a 2D
        vector field plot. Can be:

        - ``True``: plot the magnitude of the vector field. Only works when a
          single vector field is plotted.
        - ``False``/``None``: do not plot any scalar field.
        - ``Expr``: a symbolic expression representing the scalar field.
        - ``list``/``tuple``: [scalar_expr, label], where the label will be
          shown on the colorbar.

        Remember: the scalar function must return real data.

        Default to True.

    show : boolean
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the backend.

    streamlines : boolean
        Whether to plot the vector field using streamlines (True) or quivers
        (False). Default to False.

    stream_kw : dict
        A dictionary of keywords/values which is passed to the backend
        streamlines-plotting function to customize the appearance. Refer to
        the plotting library (backend) manual for more informations.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel : str, optional
        Labels for the x-axis or y-axis, respectively.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim, ylim, zlim : (float, float), optional
        Denotes the x-axis limits ory-axis limits, respectively,
        ``(min, max)``.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import I, symbols, gamma, latex, log
       >>> from spb import plot_complex_vector, plot_complex
       >>> z = symbols('z')

    Quivers plot with normalize lengths and a contour plot in background
    representing the vector's magnitude (a scalar field).

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = z**2 + 2
       >>> plot_complex_vector(expr, (z, -5 - 5j, 5 + 5j),
       ...     quiver_kw=dict(color="orange"), normalize=True, grid=False)
       Plot object containing:
       [0]: contour: sqrt(4*(re(_x) - im(_y))**2*(re(_y) + im(_x))**2 + ((re(_x) - im(_y))**2 - (re(_y) + im(_x))**2 + 2)**2) for _x over (-5.0, 5.0) and _y over (-5.0, 5.0)
       [1]: 2D vector series: [(re(_x) - im(_y))**2 - (re(_y) + im(_x))**2 + 2, 2*(re(_x) - im(_y))*(re(_y) + im(_x))] over (_x, -5.0, 5.0), (_y, -5.0, 5.0)

    Only quiver plot with normalized lengths and solid color.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_complex_vector(expr, (z, -5 - 5j, 5 + 5j),
       ...     scalar=False, use_cm=False, normalize=True)
       Plot object containing:
       [0]: 2D vector series: [(re(_x) - im(_y))**2 - (re(_y) + im(_x))**2 + 2, 2*(re(_x) - im(_y))*(re(_y) + im(_x))] over (_x, -5.0, 5.0), (_y, -5.0, 5.0)

    Only streamlines plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_complex_vector(expr, (z, -5 - 5j, 5 + 5j),
       ...     "Magnitude of $%s$" % latex(expr),
       ...     scalar=False, streamlines=True)
       Plot object containing:
       [0]: 2D vector series: [(re(_x) - im(_y))**2 - (re(_y) + im(_x))**2 + 2, 2*(re(_x) - im(_y))*(re(_y) + im(_x))] over (_x, -5.0, 5.0), (_y, -5.0, 5.0)

    Overlay the quiver plot to a domain coloring plot. By setting ``n=26``
    (even number) in the complex vector plot, the quivers won't to cross
    the branch cut.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = z * log(2 * z) + 3
       >>> p1 = plot_complex(expr, (z, -2-2j, 2+2j), grid=False, show=False)
       >>> p2 = plot_complex_vector(expr, (z, -2-2j, 2+2j),
       ...      n=26, grid=False, scalar=False, use_cm=False, normalize=True,
       ...      quiver_kw={"color": "k", "pivot": "tip"}, show=False)
       >>> (p1 + p2).show()
       >>> (p1 + p2)
       Plot object containing:
       [0]: complex domain coloring: z*log(2*z) + 3 for re(z) over (-2.0, 2.0) and im(z) over (-2.0, 2.0)
       [1]: 2D vector series: [(re(_x) - im(_y))*log(Abs(2*_x + 2*_y*I)) - (re(_y) + im(_x))*arg(_x + _y*I) + 3, (re(_x) - im(_y))*arg(_x + _y*I) + (re(_y) + im(_x))*log(Abs(2*_x + 2*_y*I))] over (_x, -2.0, 2.0), (_y, -2.0, 2.0)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       z, u, a, b = symbols("z u a b")
       plot_complex_vector(
           log(gamma(u * z)), prange(z, -5*a - b*5j, 5*a + b*5j),
           params={
               u: (1, 0, 2),
               a: (1, 0, 2),
               b: (1, 0, 2)
           }, n=20, grid=False, use_latex=False,
           quiver_kw=dict(color="orange", headwidth=4))

    See Also
    ========

    plot_real_imag, plot_complex, plot_complex_list, plot_vector

    """
    # for each argument, generate one series. Those series will be used to
    # generate the proper input arguments for plot_vector
    kwargs["absarg"] = False
    kwargs["abs"] = False
    kwargs["arg"] = False
    kwargs["real"] = True
    kwargs["imag"] = False
    kwargs["threed"] = False
    kwargs.setdefault("xlabel", "Re")
    kwargs.setdefault("ylabel", "Im")
    global_labels = kwargs.pop("label", [])

    args = _plot_sympify(args)
    params = kwargs.get("params", None)
    series = _build_series(*args, allow_lambda=False, **kwargs)
    multiple_expr = len(series) > 1

    def get_label(i):
        _iterable = args[i] if multiple_expr else args
        for t in _iterable:
            if isinstance(t, str):
                return t
        return str(args[i][0] if multiple_expr else args[0])

    # create new arguments to be used by plot_vector
    new_args = []
    x, y = symbols("x, y", cls=Dummy)
    for i, s in enumerate(series):
        expr1 = re(s.expr)
        expr2 = im(s.expr)
        free_symbols = s.expr.free_symbols
        if params is not None:
            free_symbols = free_symbols.difference(params.keys())
        free_symbols = list(free_symbols)
        if len(free_symbols) > 0:
            fs = free_symbols[0]
            expr1 = expr1.subs({fs: x + I * y})
            expr2 = expr2.subs({fs: x + I * y})
        r1 = prange(x, re(s.start), re(s.end))
        r2 = prange(y, im(s.start), im(s.end))
        label = get_label(i)
        new_args.append(((expr1, expr2), r1, r2, label))

    # substitute the complex variable in the scalar field
    scalar = kwargs.get("scalar", None)
    if scalar is not None:
        if isinstance(scalar, Expr):
            scalar = scalar.subs({fs: x + I * y})
        elif isinstance(scalar, (list, tuple)):
            scalar = list(scalar)
            scalar[0] = scalar[0].subs({fs: x + I * y})
        kwargs["scalar"] = scalar

    kwargs["label"] = global_labels
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    return plot_vector(*new_args, **kwargs)


def plot_riemann_sphere(*args, **kwargs):
    """Visualize stereographic projections of the Riemann sphere.

    Note:

    1. Differently from other plot functions that return instances of
       ``BaseBackend``, this function returns a Matplotlib figure.
    2. This function calls ``plot_complex``: refer to its documentation for
       the full list of keyword arguments.

    Parameters
    ==========

    args :
        expr : Expr
            Represent the complex function to be plotted.

        range : 3-element tuple, optional
            Denotes the range of the variables. Only works for 2D plots.
            Default to ``(z, -1.25 - 1.25*I, 1.25 + 1.25*I)``.

    annotate : boolean, optional
        Turn on/off the annotations on the 2D projections of the Riemann
        sphere. Default to True (annotations are visible). They can only
        be visible when ``riemann_mask=True``.

    riemann_mask : boolean, optional
        Turn on/off the unit disk mask representing the Riemann sphere on the
        2D projections. Default to True (mask is active).

    show_axis : boolean, optional
        Turn on/off the axis of the 2D subplots. Default to False (axis not
        visible).

    size : (width, height)
        Specify the size of the resulting figure.

    title : str, list, optional
        A list of two strings representing the titles for the two plots.


    Notes
    =====

    The Riemann sphere is a model of the extented complex plane, comprised of
    the complex plane plus a point at infinity. Let's consider a 3D space with
    a sphere with radius 1 centered at the origin. The xy plane, representing
    the complex plane, cut the sphere in half at the equator.
    The stereographic projection of any point in the complex plane on the
    sphere is given by the intersection point between a line connecting the
    complex point with the north pole of the sphere.
    Let's consider the magnitude of a complex point:

    * if its lower than one (points inside the unit disk), then the point is
      mapped to the Southern Hemisphere (the line connecting the complex point
      to the north pole intersects the sphere in the Southern Hemisphere).
      The origin of the complex plane is mapped to the south pole.
    * if its equal to one (points in the unit circle), then the point is
      already on the sphere, specifically in its equator.
    * if its greater than one (points outside the unit disk), then the point
      is mapped to the Northen Hemisphere. The north pole represents the point
      at infinity.

    Visualizing a 3D sphere is difficult (refer to Wegert [#fn4]_ for more
    information): the most obvious problem is that only a part can be seen
    from any location. A better way to fully visualize the sphere is with
    two 2D charts depicting the sphere from the inside:

    1. a stereographic projection of the sphere from the north pole, which
       depict the Southern Hemisphere. It corresponds to an ordinary
       (enhanced) domain coloring plot around the complex point $z=0$.
    2. a stereographic projection of the sphere from the south pole, which
       depict the Northen Hemisphere. It corresponds to an ordinary
       (enhanced) domain coloring plot around the complex point $z=\\infty$
       (infinity). Practically, it depicts the transformation
       $z \\rightarrow \\frac{1}{z}$.

    Let's look at an example:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import symbols, pi
       from spb import *
       z = symbols("z")
       expr = (z - 1) / (z**2 + z + 2)
       plot_riemann_sphere(expr, coloring="b", n=800)

    The saturated disks represents the hemispheres. The black circle is the
    equator. Also, a few important points are displayed to make the plot
    easier to understand.

    Note the orientation of the Northen Hemisphere: it has been rotated
    around the point at infinity by an angle `pi` and flipped about the real
    axis. This is convenient because:

    1. we can now imagine to fold the two charts so that the points 1, i, -i
       are overlayed, glue the equator and blow it up to obtain a sphere.
    2. imagine bringing the two discs closer so that they touch at the point 1.
       Now, roll the two discs together: assuming there are no branch cuts,
       there is continuity of argument and absolute value across the equator:
       what is outside of the disc in the left plot, is inside of the disk in
       the second plot, and vice-versa.

    From the above plots, the zero located at $z=1$ is clearly visible, as
    well as the two poles located at
    $z = -\\frac{1}{2} - i \\frac{\\sqrt{7}}{2}$ and
    $z = -\\frac{1}{2} + i \\frac{\\sqrt{7}}{2}$.
    Not obvious at first, there is a zero located at $z=\\infty$. We can tell
    its a zero by looking at ordering of colors around it in comparison to
    the poles. Alternatively, we can use some enhanced color scheme, for
    example one which brings poles to white:

    .. plot::
       :context: close-figs
       :include-source: True

       plot_riemann_sphere(expr, coloring="m", n=800)


    Examples
    ========

    Standard output:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import symbols, Rational, I
       from spb import *
       z = symbols("z")
       expr = 1 / (2 * z**2) + z
       plot_riemann_sphere(expr, coloring="b", n=800)

    Hide annotations:

    .. plot::
       :context: close-figs
       :include-source: True

       plot_riemann_sphere(expr, coloring="b", n=800, annotate=False)


    Hiding Riemann disk mask and annotations, set a custom domain, show axis
    (note that the right-most plot might be misleading because the center
    represents infinity), custom colormap, set the black level of contours,
    set titles.

    .. plot::
       :context: close-figs
       :include-source: True

       import colorcet
       expr = z**5 + Rational(1, 10)
       l = 2
       plot_riemann_sphere(
           expr, (z, -l-l*I, l+l*I), coloring="b", n=800,
           riemann_mask=False, show_axis=True, grid=False,
           cmap=colorcet.CET_C2, blevel=0.85,
           title=["Around zero", "Around infinity"])

    3D plot of a complex function on the Riemann sphere. Note, the higher the
    number of discretization points, the better the final results, but the
    higher memory consumption:

    .. k3d-screenshot::
       :camera: 1.87, 1.40, 1.96, 0, 0, 0, -0.45, -0.4, 0.8

       from sympy import *
       from spb import *
       z = symbols("z")
       expr = (z - 1) / (z**2 + z + 1)
       plot_riemann_sphere(expr, threed=True, n=150,
           coloring="b", backend=KB, legend=False, grid=False)


    See Also
    ========

    plot_complex


    References
    ==========

    .. [#fn4] Domain Coloring is based on Elias Wegert's book
       `"Visual Complex Functions" <https://www.springer.com/de/book/9783034801799>`_.
       The book provides the background to better understand the images.

    """
    if kwargs.get("params", dict()):
        raise NotImplementedError("Interactive widgets plots over the "
            "Riemann sphere is not implemented.")
    args = _plot_sympify(args)

    if kwargs.get("threed", False):
        kwargs = _set_discretization_points(kwargs, ComplexSurfaceBaseSeries)
        kwargs.setdefault("xlabel", "Re")
        kwargs.setdefault("ylabel", "Im")
        Backend = kwargs.get("backend", THREE_D_B)
        t, p = symbols("theta phi")
        # Northen and Southern hemispheres
        s1 = RiemannSphereSeries(args[0], (t, 0, pi/2), (p, 0, 2*pi),
            legend=False, **kwargs)
        s2 = RiemannSphereSeries(args[0], (t, pi/2, pi), (p, 0, 2*pi),
            legend=True, **kwargs)
        return _instantiate_backend(Backend, s1, s2, **kwargs)

    # look for the range: if not given, set it to an appropriate domain
    r, found_r, fs = None, False, set()
    for a in args:
        if _is_range(a):
            r = a
            found_r = True
        elif isinstance(a, Expr):
            fs = _get_free_symbols([a])
    if not r:
        s = fs.pop() if len(fs) > 0 else symbols("z")
        args.append(Tuple(s, -1.25 - 1.25 * I, 1.25 + 1.25 * I))

    kwargs = _set_discretization_points(kwargs, ComplexSurfaceBaseSeries)
    # don't show the individual plots
    show = kwargs.get("show", False)
    kwargs["show"] = False
    # set default options for Riemann sphere plots
    kwargs.setdefault("show_axis", False)
    kwargs.setdefault("riemann_mask", True)
    kwargs.setdefault("annotate", True)
    # size is applied to the final figure, not individual plots
    size = kwargs.pop("size", None)
    title = kwargs.pop("title", None)
    get_title = lambda i: title[i] if isinstance(title, (tuple, list)) else title

    # hide colorbar on first plot
    legend = kwargs.get("legend", None)
    kwargs["legend"] = False
    kwargs["title"] = get_title(0) if title is not None else "Southern Hemisphere"
    kwargs["at_infinity"] = False
    p1 = plot_complex(*args, **kwargs)
    test = (ComplexDomainColoringSeries, Parametric2DLineSeries,
        List2DSeries, GenericDataSeries)
    series = [s for s in p1.series if not isinstance(s, test)]
    if len(series) > 1:
        msg = "\n".join(str(s) for s in p1.series)
        raise ValueError("Only one symbolic expression can be plotted. "
            "Instead, the following have been received:\n" + msg)

    kwargs["title"] = get_title(1) if title is not None else "Northen Hemisphere"
    kwargs["at_infinity"] = True
    kwargs["legend"] = True if legend or (legend is None) else False
    p2 = plot_complex(*args, **kwargs)

    if legend or (legend is None):
        return plotgrid(p1, p2, nc=2, show=show, imagegrid=True, size=size)
    return plotgrid(p1, p2, nc=2, show=show, size=size)
