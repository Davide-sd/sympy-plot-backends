from sympy import Tuple, re, im, sqrt, arg
from spb.defaults import cfg
from spb.series import (
    LineOver1DRangeSeries,
    ComplexSeries,
    ComplexInteractiveSeries,
    ComplexPointSeries,
    ComplexPointInteractiveSeries,
    _set_discretization_points,
    SurfaceOver2DRangeSeries,
    InteractiveSeries,
)
from spb.utils import _plot_sympify, _check_arguments, _is_range
from spb.backends.base_backend import Plot
from spb.defaults import TWO_D_B, THREE_D_B


def _build_series(*args, interactive=False, **kwargs):
    series = []
    # apply the user-specified function to the expression
    #   keys: the user specified keyword arguments
    #   values: [function, label]
    mapping = {
        "real": [lambda t: re(t), "Re(%s)"],
        "imag": [lambda t: im(t), "Im(%s)"],
        "abs": [lambda t: sqrt(re(t)**2 + im(t)**2), "Abs(%s)"],
        "absarg": [lambda t: sqrt(re(t)**2 + im(t)**2), "Abs(%s)"],
        "arg": [lambda t: arg(t), "Arg(%s)"],
    }
    # option to be used with lambdify with complex functions
    kwargs.setdefault("modules", cfg["complex"]["modules"])

    if all([a.is_complex for a in args]):
        # args is a list of complex numbers
        cls = ComplexPointSeries if not interactive else ComplexPointInteractiveSeries
        for a in args:
            # series.append(cls([a], None, str(a), **kwargs))
            series.append(cls([a], str(a), **kwargs))
    elif (
        (len(args) > 0)
        and all([isinstance(a, (list, tuple, Tuple)) for a in args])
        and all([len(a) > 0 for a in args])
        and all([isinstance(a[0], (list, tuple, Tuple)) for a in args])
    ):
        # args is a list of tuples of the form (list, label) where list
        # contains complex points
        cls = ComplexPointSeries if not interactive else ComplexPointInteractiveSeries
        for a in args:
            # series.append(cls(a[0], None, a[-1], **kwargs))
            series.append(cls(a[0], a[-1], **kwargs))
    else:
        new_args = []

        def add_series(argument):
            nexpr, npar = 1, 1
            if len([b for b in argument if _is_range(b)]) > 1:
                # function of two variables
                npar = 2
            new_args.append(_check_arguments([argument], nexpr, npar)[0])

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
            # plotting a single expression
            add_series(args)

        for a in new_args:
            expr, ranges, label = a[0], a[1:-1], a[-1]

            if len(ranges) == 2:
                # function of two variables
                kw = kwargs.copy()
                real = kw.pop("real", True)
                imag = kw.pop("imag", True)
                _abs = kw.pop("abs", False)
                _arg = kw.pop("arg", False)

                def add_surface_series(flag, key):
                    if flag:
                        kw2 = kw.copy()
                        kw2[key] = True
                        kw2.setdefault("is_complex", True)
                        f, lbl_wrapper = mapping[key]
                        if not interactive:
                            series.append(SurfaceOver2DRangeSeries(f(expr), *ranges, lbl_wrapper % label, **kw2))
                        else:
                            series.append(InteractiveSeries([f(expr)], ranges, lbl_wrapper % label, **kw2))

                add_surface_series(real, "real")
                add_surface_series(imag, "imag")
                add_surface_series(_abs, "abs")
                add_surface_series(_arg, "arg")
                continue

            # From now on we are dealing with a function of one variable.
            # ranges need to contain complex numbers
            ranges = list(ranges)
            for i, r in enumerate(ranges):
                ranges[i] = (r[0], complex(r[1]), complex(r[2]))

            if expr.is_complex:
                # complex number with its own label
                cls = ComplexPointSeries if not interactive else ComplexPointInteractiveSeries
                series.append(cls([expr], label, **kwargs))

            else:
                if ranges[0][1].imag == ranges[0][2].imag:
                    # NOTE: as a design choice, a complex function plotted over
                    # a line will create one or more data series, depending on
                    # the keyword arguments (one for the real part, one for the
                    # imaginary part, etc.). This is undoubtely inefficient as
                    # we must evaluate the same expression multiple times.
                    # On the other hand, it allows to maintain a  one-to-one
                    # correspondance between Plot.series and backend.data, which
                    # doesn't require a redesign of the backend in order to work
                    # with iplot (backend._update_interactive).

                    kw = kwargs.copy()
                    absarg = kw.pop("absarg", False)
                    if absarg:
                        real = kw.pop("real", False)
                        imag = kw.pop("imag", False)
                        _abs = kw.pop("abs", False)
                        _arg = kw.pop("arg", False)
                    else:
                        real = kw.pop("real", True)
                        imag = kw.pop("imag", True)
                        _abs = kw.pop("abs", False)
                        _arg = kw.pop("arg", False)

                    def add_line_series(flag, key):
                        if flag:
                            kw2 = kw.copy()
                            # NOTE: in case of absarg=True, set absarg=expr so
                            # that the series knows the original expression from
                            # which to compute the argument
                            kw2[key] = True if key != "absarg" else expr
                            kw2["is_complex"] = True
                            f, lbl_wrapper = mapping[key]
                            if not interactive:
                                series.append(LineOver1DRangeSeries(f(expr), *ranges, lbl_wrapper % label, **kw2))
                            else:
                                series.append(InteractiveSeries([f(expr)], ranges, lbl_wrapper % label, **kw2))

                    add_line_series(real, "real")
                    add_line_series(imag, "imag")
                    add_line_series(_abs, "abs")
                    add_line_series(_arg, "arg")
                    add_line_series(absarg, "absarg")

                else:
                    # 2D domain coloring or 3D plots
                    cls = ComplexSeries if not interactive else ComplexInteractiveSeries

                    if not kwargs.get("threed", False):
                        mkw = kwargs.copy()
                        mkw.setdefault("coloring", cfg["complex"]["coloring"])
                        series.append(cls(expr, *ranges, label, domain_coloring=True, **mkw))

                    else:
                        # 3D plots of complex functions over a complex range

                        # NOTE: need this kw copy in case the user is plotting
                        # multiple expressions
                        kw = kwargs.copy()
                        real = kw.pop("real", False)
                        imag = kw.pop("imag", False)
                        _abs = kw.pop("abs", False)
                        _arg = kw.pop("arg", False)

                        if all(not t for t in [real, imag, _abs, _arg]):
                            # add abs plot colored by the argument
                            mkw = kwargs.copy()
                            mkw.setdefault("coloring", cfg["complex"]["coloring"])
                            series.append(cls(expr, *ranges, label, domain_coloring=True, **mkw))

                        def add_complex_series(flag, key):
                            if flag:
                                kw2 = kw.copy()
                                kw2["domain_coloring"] = False
                                f, lbl_wrapper = mapping[key]
                                series.append(cls(f(expr), *ranges, lbl_wrapper % label, **kw2))

                        add_complex_series(real, "real")
                        add_complex_series(imag, "imag")
                        add_complex_series(_abs, "abs")
                        add_complex_series(_arg, "arg")

    return series


def plot_complex(*args, show=True, **kwargs):
    """Plot complex numbers or complex functions. By default, the aspect ratio
    of the plot is set to ``aspect="equal"``.

    Depending on the provided expression, this function will produce different
    types of plots:

    * list of complex numbers: creates a scatter plot.
    * function of 1 variable over a real range:
        1. line plot separating the real and imaginary parts.
        2. line plot of the modulus of the complex function colored by its
           argument, if `absarg=True`.
        3. line plot of the modulus and the argument, if `abs=True, arg=True`.
    * function of 2 variables over 2 real ranges:
        1. By default, a surface plot of the real and imaginary part is created.
        2. By toggling `real=True, imag=True, abs=True, arg=True` we can create
           surface plots of the real, imaginary part or the absolute value or
           the argument.
    * complex function over a complex range:
        1. domain coloring plot.
        2. 3D plot of the modulus colored by the argument, if `threed=True`.
        3. 3D plot of the real and imaginary part by toggling `real=True`,
           `imag=True`.

    Parameters
    ==========
    args :
        expr : Expr
            Represent the complex number or complex function to be plotted.

        range : 3-element tuple
            Denotes the range of the variables. For example:

            * ``(z, -5, 5)``: plot a line from complex point ``(-5 + 0*I)`` to
              ``(5 + 0*I)``
            * ``(z, -5 + 2*I, 5 + 2*I)``: plot a line from complex point
              ``(-5 + 2*I)`` to ``(5 + 2 * I)``. Note the same imaginary part
              for the start/end point. Also note that we can specify the ranges
              by using standard Python complex numbers, for example
              ``(z, -5+2j, 5+2j)``.
            * ``(z, -5 - 3*I, 5 + 3*I)``: domain coloring plot of the complex
              function over the specified domain.

        label : str
            The name of the complex function to be eventually shown on the
            legend. If none is provided, the string representation of the
            function will be used.

    absarg : boolean
        If True, plot the modulus of the complex function colored by its
        argument. If False, separately plot the real and imaginary parts.
        Default to False. This is only available for line plots.

    abs : boolean
        If True, and if the provided range is a real segment, plot the
        modulus of the complex function. Default to False.

    adaptive : boolean
        Attempt to create line plots by using an adaptive algorithm.
        Default to True.

    arg : boolean
        If True, and if the provided range is a real segment, plot the
        argument of the complex function. Default to False.

    depth : int
        Controls the smootheness of the overall evaluation. The higher
        the number, the smoother the function, the more memory will be
        used by the recursive procedure. Default value is 9.

    detect_poles : boolean
        Chose whether to detect and correctly plot poles. Defaulto to False.
        This improve detection, increase the number of discretization points
        and/or change the value of `eps`.

    eps : float
        An arbitrary small value used by the `detect_poles` algorithm.
        Default value to 0.1. Before changing this value, it is better to
        increase the number of discretization points.

    n1, n2 : int
        Number of discretization points in the real/imaginary-directions,
        respectively. For domain coloring plots (2D and 3D), default to 300.
        For line plots default to 1000.

    n : int
        Set the same number of discretization points in all directions.
        For domain coloring plots (2D and 3D), default to 300. For line
        plots default to 1000.

    real : boolean
        If True, and if the provided range is a real segment, plot the
        real part of the complex function.
        If a complex range is given and ``threed=True``, plot a 3D
        representation of the real part. Default to False.

    imag : boolean
        If True, and if the provided range is a real segment, plot the
        imaginary part of the complex function.
        If a complex range is given and ``threed=True``, plot a 3D
        representation of the imaginary part. Default to False.

    show : boolean
        Default to True, in which case the plot will be shown on the screen.

    threed : boolean
        Default to False. When True, it will plot a 3D representation of the
        absolute value of the complex function colored by its argument.

    use_cm : boolean
        If ``absarg=True`` and ``use_cm=True`` then plot the modulus of the
        complex function colored by its argument. If ``use_cm=False``, plot
        the modulus of the complex function with a solid color.
        Default to True.

    coloring : str or callable
        Choose between different domain coloring options. Default to "a".

        - ``"a"``: standard domain coloring using HSV.
        - ``"b"``: enhanced domain coloring using HSV, showing iso-modulus
            and is-phase lines.
        - ``"c"``: enhanced domain coloring using HSV, showing iso-modulus
            lines.
        - ``"d"``: enhanced domain coloring using HSV, showing iso-phase
            lines.
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

        The user can also provide a callable, `f(w)`, where `w` is an
        [n x m] Numpy array (provided by the plotting module) containing
        the results (complex numbers) of the evaluation of the complex
        function. The callable should return:

        - img : ndarray [n x m x 3]
            An array of RGB colors (0 <= R,G,B <= 255)
        - colorscale : ndarray [N x 3] or None
            An array with N RGB colors, (0 <= R,G,B <= 255).
            If `colorscale=None`, no color bar will be shown on the plot.

    phaseres : int
        Default value to 20. It controls the number of iso-phase and/or
        iso-modulus lines in domain coloring plots.


    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import I, symbols, exp, sqrt, cos, sin, pi, gamma
        >>> from spb import plot_complex
        >>> x, y, z = symbols('x, y, z')

    Plot individual complex points:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex(3 + 2 * I, 4 * I, 2)
        Plot object containing:
        [0]: complex point (3 + 2*I,)
        [1]: complex point (4*I,)
        [2]: complex point (2,)

    Plot two lists of complex points:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> expr1 = z * exp(2 * pi * I * z)
        >>> expr2 = 2 * expr1
        >>> n = 15
        >>> l1 = [expr1.subs(z, t / n) for t in range(n)]
        >>> l2 = [expr2.subs(z, t / n) for t in range(n)]
        >>> plot_complex((l1, "f1"), (l2, "f2"))
        Plot object containing:
        [0]: complex points (0.0, 0.0666666666666667*exp(0.133333333333333*I*pi), 0.133333333333333*exp(0.266666666666667*I*pi), 0.2*exp(0.4*I*pi), 0.266666666666667*exp(0.533333333333333*I*pi), 0.333333333333333*exp(0.666666666666667*I*pi), 0.4*exp(0.8*I*pi), 0.466666666666667*exp(0.933333333333333*I*pi), 0.533333333333333*exp(1.06666666666667*I*pi), 0.6*exp(1.2*I*pi), 0.666666666666667*exp(1.33333333333333*I*pi), 0.733333333333333*exp(1.46666666666667*I*pi), 0.8*exp(1.6*I*pi), 0.866666666666667*exp(1.73333333333333*I*pi), 0.933333333333333*exp(1.86666666666667*I*pi))
        [1]: complex points (0, 0.133333333333333*exp(0.133333333333333*I*pi), 0.266666666666667*exp(0.266666666666667*I*pi), 0.4*exp(0.4*I*pi), 0.533333333333333*exp(0.533333333333333*I*pi), 0.666666666666667*exp(0.666666666666667*I*pi), 0.8*exp(0.8*I*pi), 0.933333333333333*exp(0.933333333333333*I*pi), 1.06666666666667*exp(1.06666666666667*I*pi), 1.2*exp(1.2*I*pi), 1.33333333333333*exp(1.33333333333333*I*pi), 1.46666666666667*exp(1.46666666666667*I*pi), 1.6*exp(1.6*I*pi), 1.73333333333333*exp(1.73333333333333*I*pi), 1.86666666666667*exp(1.86666666666667*I*pi))

    Plot the real and imaginary part of a function:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex(sqrt(x), (x, -3, 3))
        Plot object containing:
        [0]: cartesian line: (re(x)**2 + im(x)**2)**(1/4)*cos(atan2(im(x), re(x))/2) for x over ((-3+0j), (3+0j))
        [1]: cartesian line: (re(x)**2 + im(x)**2)**(1/4)*sin(atan2(im(x), re(x))/2) for x over ((-3+0j), (3+0j))

    Plot the modulus of a complex function colored by its magnitude:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex((cos(x) + sin(I * x), "f"), (x, -2, 2), absarg=True)
        Plot object containing:
        [0]: cartesian line: sqrt((-sin(re(x))*sinh(im(x)) + cos(im(x))*sinh(re(x)))**2 + (-sin(im(x))*cosh(re(x)) + cos(re(x))*cosh(im(x)))**2) for x over ((-2+0j), (2+0j))

    Plot the modulus and the argument of a complex function:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex((cos(x) + sin(I * x), "f"), (x, -2, 2),
        ...     abs=True, arg=True, real=False, imag=False)
        Plot object containing:
        [0]: cartesian line: sqrt((-sin(re(x))*sinh(im(x)) + cos(im(x))*sinh(re(x)))**2 + (-sin(im(x))*cosh(re(x)) + cos(re(x))*cosh(im(x)))**2) for x over ((-2+0j), (2+0j))
        [1]: cartesian line: arg(cos(x) + I*sinh(x)) for x over ((-2+0j), (2+0j))

    Plot the real and imaginary part of a function of two variables over two
    real ranges:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex(sqrt(x*y), (x, -5, 5), (y, -5, 5),
        ...     real=True, imag=True)
        Plot object containing:
        [0]: cartesian surface: (re(x*y)**2 + im(x*y)**2)**(1/4)*cos(atan2(im(x*y), re(x*y))/2) for x over ((-5+0j), (5+0j)) and y over ((-5+0j), (5+0j))
        [1]: cartesian surface: (re(x*y)**2 + im(x*y)**2)**(1/4)*sin(atan2(im(x*y), re(x*y))/2) for x over ((-5+0j), (5+0j)) and y over ((-5+0j), (5+0j))

    Domain coloring plot. Note that it might be necessary to increase the number
    of discretization points in order to get a smoother plot:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex(gamma(z), (z, -3 - 3*I, 3 + 3*I), coloring="b", n=500)
        Plot object containing:
        [0]: domain coloring: gamma(z) for re(z) over (-3.0, 3.0) and im(z) over (-3.0, 3.0)

    3D plot of the absolute value of a complex function colored by its argument:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex(gamma(z), (z, -3 - 3*I, 3 + 3*I), threed=True,
        ...     zlim=(-1, 6))
        Plot object containing:
        [0]: cartesian surface: gamma(z) for re(z) over (-3.0, 3.0) and im(z) over (-3.0, 3.0)

    3D plot of the real part a complex function:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_complex(gamma(z), (z, -3 - 3*I, 3 + 3*I), threed=True,
        ...     real=True, imag=False)
        Plot object containing:
        [0]: cartesian surface: re(gamma(z)) for re(z) over (-3.0, 3.0) and im(z) over (-3.0, 3.0)

    References
    ==========

    Domain Coloring colorschemes are based on Elias Wegert's book
    `"Visual Complex Functions" <https://www.springer.com/de/book/9783034801799>`_.
    The book provides the background to better understand the images.

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, ComplexSeries)

    series = _build_series(*args, **kwargs)

    if "backend" not in kwargs:
        kwargs["backend"] = TWO_D_B
        if any(s.is_3Dsurface for s in series):
            kwargs["backend"] = THREE_D_B

    if all(
        isinstance(s, (SurfaceOver2DRangeSeries, InteractiveSeries)) for s in series
    ):
        # function of 2 variables
        if kwargs.get("xlabel", None) is None:
            kwargs["xlabel"] = str(series[0].var_x)
        if kwargs.get("ylabel", None) is None:
            kwargs["ylabel"] = str(series[0].var_y)
        # do not set anything for zlabel since it could be f(x,y) or
        # abs(f(x, y)) or something else
    elif all(not s.is_parametric for s in series):
        # when plotting real/imaginary or domain coloring/3D plots, the
        # horizontal axis is the real, the vertical axis is the imaginary
        if kwargs.get("xlabel", None) is None:
            kwargs["xlabel"] = "Re"
        if kwargs.get("ylabel", None) is None:
            kwargs["ylabel"] = "Im"
        if kwargs.get("zlabel", None) is None:
            kwargs["zlabel"] = "Abs"
    else:
        if kwargs.get("xlabel", None) is None:
            kwargs["xlabel"] = "Real"
        if kwargs.get("ylabel", None) is None:
            kwargs["ylabel"] = "Abs"

    if (kwargs.get("aspect", None) is None) and any(
        (s.is_complex and s.is_domain_coloring) or s.is_point for s in series
    ):
        # set aspect equal for 2D domain coloring or complex points
        kwargs.setdefault("aspect", "equal")

    p = Plot(*series, **kwargs)
    if show:
        p.show()
    return p

def plot3d_complex(*args, **kwargs):
    """ Wrapper function of ``plot_complex``, which sets ``threed=True``.
    As such, it is not guaranteed that the output plot is 3D: it dependes on
    the user provided arguments.

    Read ``plot_complex`` documentation to learn about its usage.

    See Also
    ========

    plot_complex

    """
    kwargs["threed"] = True
    return plot_complex(*args, **kwargs)
