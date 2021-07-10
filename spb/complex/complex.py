from sympy import Tuple, Expr
from spb.series import (
    ComplexSeries, ComplexInteractiveSeries, _set_discretization_points
)
from spb.vectors import _preprocess
from spb.utils import _plot_sympify
from spb.utils import _check_arguments
from spb.backends.base_backend import Plot
from spb.defaults import TWO_D_B, THREE_D_B

# def _build_series(expr, ranges, label, kwargs):
def _build_series(*args, interactive=False, **kwargs):
    series = []
    cls = ComplexSeries if not interactive else ComplexInteractiveSeries
    
    if all([a.is_complex for a in args]):
        # args is a list of complex numbers
        for a in args:
            series.append(cls([a], None, str(a), **kwargs))
    elif ((len(args) > 0) and 
            all([isinstance(a, (list, tuple, Tuple)) for a in args]) and
            all([len(a) > 0 for a in args]) and
            all([isinstance(a[0], (list, tuple, Tuple)) for a in args])):
        # args is a list of tuples of the form (list, label) where list 
        # contains complex points
        for a in args:
            series.append(cls(a[0], None, a[-1], **kwargs))
    else:
        args = _check_arguments(args, 1, 1)
        
        for a in args:
            expr, ranges, label = a[0], a[1:-1], a[-1]
            # ranges need to contain complex numbers
            ranges = list(ranges)
            for i, r in enumerate(ranges):
                ranges[i] = (r[0], complex(r[1]), complex(r[2]))

            if expr.is_complex:
                # complex number
                series.append(cls([expr], None, label, **kwargs))
            else:
                if ((ranges[0][1].imag == ranges[0][2].imag) and
                        not kwargs.get('absarg', False)):
                    # complex expression evaluated over a line: need to add two
                    # series, one for the real and imaginary part, respectively

                    # NOTE: as a design choice, a complex function plotted over 
                    # a line will create two data series, one for the real part,
                    # the other for the imaginary part. This is undoubtely
                    # inefficient as we must evaluate the same expression two
                    # times. On the other hand, it allows to maintain a 
                    # one-to-one correspondance between Plot.series and 
                    # backend.data, which doesn't require a redesign of the
                    # backend in order to work with iplot
                    # (backend._update_interactive).

                    kw = kwargs.copy()
                    real = kw.pop("real", True)
                    imag = kw.pop("imag", True)
                    _abs = kw.pop("abs", False)
                    arg = kw.pop("arg", False)

                    def func(flag, key):
                        if flag:
                            kw2 = kw.copy()
                            kw2[key] = True
                            series.append(cls(expr, *ranges, label, **kw2))
                    
                    func(real, "real")
                    func(imag, "imag")
                    func(_abs, "abs")
                    func(arg, "arg")
                else:
                    real = kwargs.get("real", False)
                    imag = kwargs.get("imag", False)
                    if real and imag:
                        kw1, kw2 = kwargs.copy(), kwargs.copy()
                        kw1["real"], kw1["imag"] = True, False
                        kw2["real"], kw2["imag"] = False, True
                        series.append(cls(expr, *ranges, label, **kw1))
                        series.append(cls(expr, *ranges, label, **kw2))
                    else:
                        if ((kwargs.get("coloring", None) == "f") and 
                            (not kwargs.get("threed", False))):
                            # cplot shows contour lines for absolute value and 
                            # the argument of a complex function.
                            def pop_kw(k, *dicts):
                                for d in dicts:
                                    if k in d.keys():
                                        d.pop(k)

                            kw1 = kwargs.copy()
                            kw2 = kwargs.copy()
                            pop_kw("arg", kw1, kw2)
                            pop_kw("abs", kw1, kw2)
                            pop_kw("coloring", kw1)
                            if kwargs.get("abs", False):
                                series.append(cls(expr, *ranges, label, abs=True, **kw1))
                            if kwargs.get("arg", False):
                                series.append(cls(expr, *ranges, label, arg=True, levels1=True, **kw1))
                                series.append(cls(expr, *ranges, label, arg=True, levels1=False, **kw1))
                            series.append(cls(expr, *ranges, label, **kw2))
                        else:
                            series.append(cls(expr, *ranges, label, **kwargs))
    return series

def complex_plot(*args, show=True, **kwargs):
    """ Plot complex numbers or complex functions. By default, the aspect ratio 
    of the plot is set to ``aspect="equal"``.
    
    Depending on the provided expression, this function will produce different 
    types of plots:
    * list of complex numbers: creates a scatter plot.
    * complex function over a real range:
        1. line plot separating the real and imaginary parts.
        2. line plot of the modulus of the complex function colored by its
            argument, if `absarg=True`.
        3. line plot of the modulus and the argument.
    * complex function over a complex range:
        1. domain coloring plot.
        2. 3D plot of the modulus colored by the argument, if `threed=True`.
        3. 3D plot of the real and imaginary part.

    Arguments
    =========
        expr : Expr
            Represent the complex number or complex function to be plotted.
        
        range : 3-element tuple
            Denotes the range of the variables. For example:
            * (z, -5, 5): plot a line from complex point (-5 + 0*I) to (5 + 0*I)
            * (z, -5 + 2*I, 5 + 2*I): plot a line from complex point (-5 + 2*I)
                to (5 + 2 * I). Note the same imaginary part for the start/end
                point.
            * (z, -5 - 3*I, 5 + 3*I): domain coloring plot of the complex
                function over the specified domain.

        label : str
            The name of the complex function to be eventually shown on the
            legend. If none is provided, the string representation of the 
            function will be used.
        
        To specify multiple complex functions, wrap them into a tuple.
        Refer to the examples to learn more.

    Keyword Arguments
    =================

        absarg : boolean
            If True, plot the modulus of the complex function colored by its
            argument. If False, separately plot the real and imaginary parts.
            Default to False.
        
        abs : boolean
            If True, and if the provided range is a real segment, plot the 
            modulus of the complex function. Default to False.
        
        adaptive : boolean
            Attempt to create line plots by using an adaptive algorithm.
            Default to True. If `absarg=True`, the function will automatically
            switch to `adaptive=False`, using a uniformly-spaced grid.
        
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
            If a complex range is given and `threed=True`, plot a 3D 
            representation of the real part. Default to False.
        
        imag : boolean
            If True, and if the provided range is a real segment, plot the 
            imaginary part of the complex function.
            If a complex range is given and `threed=True`, plot a 3D 
            representation of the imaginary part. Default to False.
        
        show : boolean
            Default to True, in which case the plot will be shown on the screen.
        
        threed : boolean
            Default to False. When True, it will plot a 3D representation of the
            absolute value of the complex function colored by its argument.

        use_cm : boolean
            If `absarg=True` and `use_cm=True` then plot the modulus of the 
            complex function colored by its argument. If `use_cm=False`, plot 
            the modulus of the complex function with a solid color.
            Default to True.

    Domain Coloring Arguments
    =========================

        coloring : str
            Default to "a". Chose between different coloring options:
            "a": standard domain coloring using HSV.
            "b": enhanced domain coloring using HSV, showing iso-modulus and
                is-phase lines.
            "c": enhanced domain coloring using HSV, showing iso-modulus lines.
            "d": enhanced domain coloring using HSV, showing iso-phase lines.
            "e": HSV color grading. Read the following article to understand it:
                https://www.codeproject.com/Articles/80641/Visualizing-Complex-Functions
            "f": domain coloring implemented by cplot:
                https://github.com/nschloe/cplot
                Use the following keywords to further customize the appearance:
                    `abs_scaling`: str
                        Default to "h-1". It can be used to adjust the use of 
                        colors. h with a value less than 1.0 adds more color 
                        which can help isolating the roots and poles (which are 
                        still black and white, respectively). "h-0.0" ignores 
                        the magnitude of f(z) completely. "arctan" is another
                        possible scaling.
                    `colorspace` : str
                        Default to "cam16". Can be set to "hsl" to get the 
                        common fully saturated, vibrant colors.
                    `abs` and/or `args` : boolean
                        Set them to True to show contour lines for absolute 
                        value and argument.
                    `levels` : (n_abs, n_arg)
                        Number of contour levels for the absolute value and the
                        argument.
                WARNING: if `abs=True` and/or `arg=True`, only MatplotlibBackend
                will be able to render the plot! Moreover, `iplot` won't be
                able to update these contour lines.
            "g": alternating black and white stripes corresponding to modulus.
            "h": alternating black and white stripes corresponding to phase.
            "i": alternating black and white stripes corresponding to real part.
            "j": alternating black and white stripes corresponding to imaginary 
                part.
            "k": cartesian chessboard on the complex points space. The result 
                will hide zeros.
            "l": polar Chessboard on the complex points space. The result will 
                show conformality.

        alpha : float
            This parameter works when `coloring="f"`.
            Default to 1. Can be `0 <= alpha <= 1`. It adjust the use of colors.
            A value less than 1 adds more color which can help isolating the
            roots and poles (which are still black and white, respectively).
            alpha=0 ignores the magnitude of f(z) completely.
        
        colorspace : str
            This parameter works when `coloring="f"`.
            Default to `"cam16"`. Other options are `"cielab", "oklab", "hsl"`.
            It can be set to `"hsl"` to get the common fully saturated, vibrant
            colors. This is usually a bad idea since it creates artifacts which
            are not related with the underlying data.
        
        phaseres : int
            This parameter works when `coloring` is different from `"f"`.
            Default value to 20. It controls the number of iso-phase or 
            iso-modulus lines.

    Examples
    ========

    Plot individual complex points:

    .. code-block:: python
        complex_plot(3 + 2 * I, 4 * I, 2, aspect="equal", legend=True)

    Plot two lists of complex points:

    .. code-block:: python
        z = symbols("z")
        expr1 = z * exp(2 * pi * I * z)
        expr2 = 2 * expr1
        l1 = [expr1.subs(z, t / 20) for t in range(20)]
        l2 = [expr2.subs(z, t / 20) for t in range(20)]
        complex_plot((l1, "f1"), (l2, "f2"), aspect="equal", legend=True)

    Plot the real and imaginary part of a function:

    .. code-block:: python
        z = symbols("z")
        complex_plot(sqrt(z), (z, -3, 3), legend=True)
    
    .. code-block:: python
        z = symbols("z")
        complex_plot((cos(z) + sin(I * z), "f"), (z, -2, 2), legend=True)

    Plot the modulus of a complex function colored by its magnitude:

    .. code-block:: python
        z = symbols("z")
        complex_plot((cos(z) + sin(I * z), "f"), (z, -2, 2), legend=True,
            absarg=True)
    
    Plot the modulus and the argument of a complex function:

    .. code-block:: python
        z = symbols("z")
        complex_plot((cos(z) + sin(I * z), "f"), (z, -2, 2), legend=True,
            abs=True, arg=True, real=False, imag=False)
    
    Domain coloring plot. Note that it might be necessary to increase the number
    of discretization points in order to get a smoother plot:

    .. code-block:: python
        z = symbols("z")
        complex_plot(gamma(z), (z, -3 - 3*I, 3 + 3*I), coloring="b", n=500)
    
    3D plot of the absolute value of a complex function colored by its argument:

    .. code-block:: python
        z = symbols("z")
        complex_plot(gamma(z), (z, -3 - 3*I, 3 + 3*I), threed=True, 
            legend=True, zlim=(-1, 6))
    
    3D plot of the real part a complex function:

    .. code-block:: python
        z = symbols("z")
        complex_plot(gamma(z), (z, -3 - 3*I, 3 + 3*I), threed=True, 
            real=True)

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, ComplexSeries)
    
    series = _build_series(*args, **kwargs)
    
    if not "backend" in kwargs:
        kwargs["backend"] = TWO_D_B
        if any(s.is_3Dsurface for s in series):
            kwargs["backend"] = THREE_D_B
    
    if all(not s.is_parametric for s in series):
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

    if ((kwargs.get("aspect", None) is None) and 
            any(s.is_domain_coloring for s in series)):
        kwargs["aspect"] = "equal"
    
    p = Plot(*series, **kwargs)
    if show:
        p.show()
    return p
