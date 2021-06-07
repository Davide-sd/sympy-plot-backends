from spb.series import ComplexSeries, _set_discretization_points
from spb.vectors import _preprocess
from spb.utils import _plot_sympify
from spb.utils import _check_arguments
from spb.backends.base_backend import Plot
from spb.defaults import TWO_D_B, THREE_D_B

def _build_series(expr, ranges, label, kwargs):
    pass

def complex_plot(*args, show=True, **kwargs):
    """ Plot complex numbers or complex functions. Depending on the provided
    expression, this function will produce different types of plots:
    * list of complex numbers: creates a scatter plot.
    * complex function over a line range: line plot separating the real and
        imaginary parts.
    * complex function over a complex range: domain coloring plot.


    Domain Coloring Arguments
    =========================

        alpha : float
            Default to 1. Can be `0 <= alpha <= 1`. It adjust the use of colors.
            A value less than 1 adds more color which can help isolating the
            roots and poles (which are still black and white, respectively).
            alpha=0 ignores the magnitude of f(z) completely.
        
        colorspace : str
            Default to `"cam16"`. Other options are `"cielab", "oklab", "hsl"`.
            It can be set to `"hsl"` to get the common fully saturated, vibrant
            colors. This is usually a bad idea since it creates artifacts which
            are not related with the underlying data.

    [], (): one or more complex points
    expr: a complex function

    Parameters
    ==========

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, ComplexSeries)
    print("args", args)
    series = []
    if all([a.is_complex for a in args]):
        print("case 1")
        # list of complex numbers
        series.append(ComplexSeries(args, None, "", **kwargs))
    else:
        print("case 2")
        args = _check_arguments(args, 1, 1)
        # args = _preprocess(*args)
        # args = _preprocess(*args)
        for a in args:
            expr, ranges, label = a[0], a[1:-1], a[-1]

            # ranges need to contain complex numbers
            ranges = list(ranges)
            for i, r in enumerate(ranges):
                ranges[i] = (r[0], complex(r[1]), complex(r[2]))

            if expr.is_complex:
                # complex number
                series.append(ComplexSeries([expr], None, label, **kwargs))
            else:
                if ranges[0][1].imag == ranges[0][2].imag:
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

                    kw1, kw2 = kwargs.copy(), kwargs.copy()
                    kw1["real"], kw1["imag"] = True, False
                    kw2["real"], kw2["imag"] = False, True
                    series.append(ComplexSeries(expr, *ranges, label, **kw1))
                    series.append(ComplexSeries(expr, *ranges, label, **kw2))
                else:
                    series.append(ComplexSeries(expr, *ranges, label, **kwargs))
            print("expr, ranges, label", expr, ranges, label)
    
    if not "backend" in kwargs:
        kwargs["backend"] = TWO_D_B
        
    p = Plot(*series, **kwargs)
    if show:
        p.show()
    return p
    # # args = _preprocess(*args)
    # kwargs = _set_discretization_points(kwargs, ComplexSeries)
    # print("", [a for a in args], [a[0].is_complex for a in args])
    # is_number = all([a[0].is_complex for a in args])
    # print("is_number", is_number)
    # series = []
    # for a in args:
    #     print("arg", a)
    #     # split_expr, ranges, s = _build_series(a[0], *a[1:-1], label=a[-1], **kwargs)
    #     # series.append(s)