from spb.defaults import cfg, TWO_D_B, THREE_D_B
from spb.functions import (
    _set_labels
)
from spb.series import ComplexPointSeries
from spb.graphics import (
    complex_points, line_abs_arg, line_abs_arg_colored, line_real_imag,
    surface_abs_arg, surface_real_imag, domain_coloring, analytic_landscape,
    riemann_sphere_2d, riemann_sphere_3d, complex_vector_field, graphics,
    contour_abs_arg, contour_real_imag
)
from spb.utils import (
    _unpack_args, _plot_sympify, _check_arguments,
    _is_range, _get_free_symbols
)
from spb.plotgrid import plotgrid
from sympy import latex, Tuple, im, Expr, symbols, I
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
            series.extend(complex_points(a, **kwargs))
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
            series.extend(complex_points(
                expr[0], label=label, rendering_kw=rkw, **kwargs))
    elif (
        (len(args) > 0)
        and all([isinstance(a, (list, tuple, Tuple)) for a in args])
        and all([all([isinstance(t, Expr) and t.is_complex for t in a]) for a in args])
    ):
        # args is a list of lists
        for a in args:
            series.extend(complex_points(a, **kwargs))
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
            series.extend(complex_points(expr[0], label=label,
                rendering_kw=rkw, **kwargs))

    else:
        expr, ranges, label, rkw = _unpack_args(*args)
        if isinstance(expr, (list, tuple, Tuple)):
            expr = expr[0]
        series.append(complex_points(expr, label=label,
            rendering_kw=rkw, **kwargs))

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

        if im(ranges[0][1]) == im(ranges[0][2]):
            # dealing with lines
            if absarg:
                series.extend(
                    line_abs_arg_colored(expr, ranges[0], label, **kw))
            if real or imag:
                series.extend(
                    line_real_imag(expr, ranges[0], label,
                    real=real, imag=imag, **kw))
            if _abs or _arg:
                series.extend(
                    line_abs_arg(expr, ranges[0], label,
                    abs=_abs, arg=_arg, **kw))
        else:
            threed = kw.pop("threed", False)
            if absarg:
                func = analytic_landscape if threed else domain_coloring
                asd = func(expr, ranges[0], label, **kw)
                series.extend(asd)
            if real or imag:
                func = surface_real_imag if threed else contour_real_imag
                series.extend(
                    func(expr, ranges[0], label, real=real, imag=imag, **kw))
            if _abs or _arg:
                func = surface_abs_arg if threed else contour_abs_arg
                series.extend(
                    func(expr, ranges[0], label, abs=_abs, arg=_arg, **kw))

    _set_labels(series, global_labels, global_rendering_kw)
    return series


def _plot_complex(*args, allow_lambda=False, pcl=False, **kwargs):
    """Create the series and setup the backend."""
    args = _plot_sympify(args)

    if not pcl:
        series = _build_series(*args, allow_lambda=allow_lambda, **kwargs)
    else:
        series = _build_complex_point_series(*args, allow_lambda=allow_lambda, pcl=True, **kwargs)

    if len(series) == 0:
        warnings.warn("No series found. Check your keyword arguments.")

    _set_axis_labels(series, kwargs)
    return graphics(*series, **kwargs)


def _set_axis_labels(series, kwargs):
    """Set the axis labels for the plot, depending on the series being
    visualized.
    """
    if all(s.is_parametric for s in series):
        if kwargs.get("xlabel", None) is None:
            kwargs["xlabel"] = "Real"
        if kwargs.get("ylabel", None) is None:
            kwargs["ylabel"] = "Abs"
    elif all(s.is_domain_coloring or s.is_3Dsurface or s.is_contour or
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
    """Plot the real and imaginary parts, the absolute value and the
    argument of a complex function. By default, only the real and imaginary
    parts will be plotted. Use keyword argument to be more specific.
    By default, the aspect ratio of 2D plots is set to ``aspect="equal"``.

    Depending on the provided expression, this function will produce different
    types of plots:

    1. line plot over the reals.
    2. surface plot over the complex plane if `threed=True`.
    3. contour plot over the complex plane if `threed=False`.

    Typical usage examples are in the followings:

    - Plotting a single expression with the default range (-10, 10):

      .. code-block::

         plot_real_imag(expr, **kwargs)

    - Plotting multiple expressions with a single range:

      .. code-block::

         plot_real_imag(expr1, expr2, ..., range, **kwargs)

    - Plotting multiple expressions with multiple ranges, custom labels and
      rendering options:

      .. code-block::

         plot_real_imag(
            (expr1, range1, label1 [opt], rendering_kw1 [opt]),
            (expr2, range2, label2 [opt], rendering_kw2 [opt]), ..., **kwargs)

    Refer to :func:`~spb.graphics.complex_analysis.line_real_imag` or
    :func:`~spb.graphics.complex_analysis.surface_real_imag` for a full
    list of keyword arguments to customize the appearances of lines and
    surfaces.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========
    real : boolean, optional
        Show/hide the real part. Default to True (visible).
    imag : boolean, optional
        Show/hide the imaginary part. Default to True (visible).
    abs : boolean, optional
        Show/hide the absolute value. Default to False (hidden).
    arg : boolean, optional
        Show/hide the argument. Default to False (hidden).

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

    Typical usage examples are in the followings:

    - Plotting a single expression with a single range:

      .. code-block::

         plot_complex(expr, range, **kwargs)

    - Plotting multiple expressions with different ranges, custom labels and
      rendering options:

      .. code-block::

         plot_complex(
            (expr1, range1, label1 [opt], rendering_kw1 [opt]),
            (expr2, range2, label2 [opt], rendering_kw2 [opt]),
            ..., **kwargs)

    Refer to :func:`~spb.graphics.complex_analysis.line_abs_arg_colored` or
    :func:`~spb.graphics.complex_analysis.domain_coloring` or
    :func:`~spb.graphics.complex_analysis.analytic_landscape` for a full
    list of keyword arguments to customize the appearances of lines and
    surfaces.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

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
       ...     grid=False, at_infinity=True, axis=False)
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

    - Plotting a single list of complex numbers:

      .. code-block::

         plot_complex_list(l1, **kwargs)

    - Plotting multiple lists of complex numbers:

      .. code-block::

         plot_complex_list(l1, l2, **kwargs)

    - Plotting multiple lists of complex numbers each one with a custom label
      and rendering options:

      .. code-block::

         plot_complex_list(
            (l1, label1, rendering_kw1),
            (l2, label2, rendering_kw2), **kwargs)`

    Refer to :func:`~spb.graphics.complex_analysis.complex_points` for a full
    list of keyword arguments to customize the appearances of lines.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

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

    - Plotting a vector field of a complex function:

      .. code-block::

         plot_complex_vector(expr, range, **kwargs)

    - Plotting multiple vector fields with different ranges and custom labels:

      .. code-block::

         plot_complex_vector(
            (expr1, range1, label1 [optional]),
            (expr2, range2, label2 [optional]), **kwargs)

    Refer to :func:`~spb.graphics.vectors.vector_field_2d` for a full
    list of keyword arguments to customize the appearances of quivers,
    streamlines and contour.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

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

    new_series = []
    for i, s in enumerate(series):
        new_series.extend(
            complex_vector_field(s.expr, s.ranges[0], label=get_label(i), **kwargs)
        )
    for s, lbl in zip(new_series, global_labels):
        s.label = lbl
    return graphics(*new_series, **kwargs)


def plot_riemann_sphere(expr, range=None, annotate=True, riemann_mask=True,
    **kwargs):
    """Visualize stereographic projections of the Riemann sphere.

    Note:

    1. Differently from other plot functions that return instances of
       ``BaseBackend``, this function returns a Matplotlib figure.
    2. This function calls :func:`~plot_complex`: refer to its documentation
       for the full list of keyword arguments.

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

    axis : boolean, optional
        Turn on/off the axis of the 2D subplots. Default to False (axis not
        visible).

    size : (width, height)
        Specify the size of the resulting figure.

    title : str, list, optional
        A list of two strings representing the titles for the two plots.


    Notes
    =====

    The [Riemann-sphere]_ is a model of the extented complex plane,
    comprised of the complex plane plus a point at infinity. Let's consider
    a 3D space with a sphere of radius 1 centered at the origin. The xy plane,
    representing the complex plane, cut the sphere in half at the equator.
    The [Stereographic]_ projection of any point in the complex plane on the
    sphere is given by the intersection point between a line connecting the
    complex point with the north pole of the sphere.
    Let's consider the magnitude of a complex point:

    * if its lower than one (points inside the unit disk), then the point is
      mapped to the Southern Hemisphere (the line connecting the complex point
      to the north pole intersects the sphere in the Southern Hemisphere).
      The origin of the complex plane is mapped to the south pole.
    * if its equal to one (points in the unit circle), then the point is
      already on the sphere, specifically in its equator.
    * if its greater than one (point outside the unit disk), then the point
      is mapped to the Northen Hemisphere. The north pole represents the point
      at infinity.

    Visualizing a 3D sphere is difficult (refer to [Wegert]_ for more
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
           riemann_mask=False, axis=True, grid=False,
           cmap=colorcet.CET_C2, blevel=0.85,
           title=["Around zero", "Around infinity"])

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates
    the use of the ``params`` dictionary to specify sliders in their basic
    form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy import *
       from sympy.abc import a, b, c
       from spb import *
       z = symbols("z")
       expr = (z - 1) / (a * z**2 + b * z + c)
       plot_riemann_sphere(
           expr, coloring="b", n=300,
           params={
               a: (1, -2, 2),
               b: (1, -2, 2),
               c: (2, -10, 10),
           },
           use_latex=False
       )

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

    """
    expr = _plot_sympify(expr)
    params = kwargs.get("params", {})

    if kwargs.get("threed", False):
        if kwargs.get("params", dict()):
            raise NotImplementedError("Interactive widgets plots over the "
                "Riemann sphere is not implemented.")
        series = riemann_sphere_3d(expr, **kwargs)
        kwargs.setdefault("xlabel", "Re")
        kwargs.setdefault("ylabel", "Im")
        kwargs.setdefault("zlabel", "")
        return graphics(*series, **kwargs)

    if not range:
        fs = _get_free_symbols(expr).difference(params.keys())
        s = fs.pop() if len(fs) > 0 else symbols("z")
        range = Tuple(s, -1.25 - 1.25 * I, 1.25 + 1.25 * I)

    # don't show the individual plots
    show = kwargs.get("show", True)
    kwargs["show"] = False
    # set default options for Riemann sphere plots
    kwargs.setdefault("axis", False)
    kwargs["riemann_mask"] = riemann_mask
    kwargs["annotate"] = annotate
    # size is applied to the final figure, not individual plots
    size = kwargs.pop("size", None)
    title = kwargs.pop("title", None)
    get_title = lambda i: title[i] if isinstance(title, (tuple, list)) else title

    # hide colorbar on first plot
    kwargs["colorbar"] = False
    kwargs["title"] = get_title(0) if title is not None else "Southern Hemisphere"
    kwargs["at_infinity"] = False
    p1 = graphics(riemann_sphere_2d(expr, range, **kwargs), **kwargs)

    kwargs["title"] = get_title(1) if title is not None else "Northen Hemisphere"
    kwargs["at_infinity"] = True
    kwargs["colorbar"] = True
    p2 = graphics(riemann_sphere_2d(expr, range, **kwargs), **kwargs)

    pg_interactive_kwargs = dict(
        imodule=kwargs.get("imodule", None),
        layout=kwargs.get("layout", "tb"),
        template=kwargs.get("template", None),
        ncols=kwargs.get("ncols", 2),
    )
    pg = plotgrid(p1, p2, nc=2, imagegrid=True, size=size, show=False,
        **pg_interactive_kwargs)

    if len(params) == 0:
        if pg.is_matplotlib_fig:
            if show:
                pg.show()
                return pg
            return pg
        if show:
            return pg.show()
        return pg
    if show:
        return pg.show()
    return pg
