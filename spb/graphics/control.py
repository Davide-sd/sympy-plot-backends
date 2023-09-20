from spb.defaults import TWO_D_B
from spb.series import (
    List2DSeries, LineOver1DRangeSeries, HVLineSeries, NyquistLineSeries,
    NicholsLineSeries
)
from spb.utils import prange
import numpy as np
from sympy import (
    roots, exp, Poly, degree, re, im, apart, Dummy,
    I, log, Abs, arg, sympify, S, Min, Max, Piecewise, sqrt,
    floor, ceiling, frac, pi, fraction, Expr, Tuple
)
from sympy.physics.control.lti import (
    SISOLinearTimeInvariant, TransferFunctionMatrix, TransferFunction
)
from mergedeep import merge

# TODO: remove this and update setup.py
from packaging import version
import sympy
curr_sympy_ver = version.parse(sympy.__version__)
sympy_1_12 = version.parse("1.12.0")
if curr_sympy_ver >= sympy_1_12:
    from sympy.integrals.laplace import _fast_inverse_laplace
else:
    from sympy.integrals.transforms import _fast_inverse_laplace


__all__ = [
    'pole_zero',
    'step_response',
    'impulse_response',
    'ramp_response',
    'bode_magnitude',
    'bode_phase',
    'nyquist',
    'nichols'
]


def _preprocess_system(system, **kwargs):
    """Allow users to provide a transfer function with the following form:

    1. instance of ``TransferFunction``.
    2. symbolic expression.
    3. tuple of 2/3 elements ``(num, den, generator)``.

    Returns
    =======
    system : TransferFunction
    """
    if isinstance(system, (SISOLinearTimeInvariant, TransferFunctionMatrix)):
        return system

    if isinstance(system, (list, tuple)):
        if len(system) == 2:
            num, den = system
            fs = Tuple(num, den).free_symbols.pop()
        elif len(system) == 3:
            num, den, fs = system
        else:
            raise ValueError(
                "If a tuple/list is provided, it must have "
                "two or three elements: (num, den, free_symbol [opt]). "
                f"Received len(system) = {len(system)}"
            )
        return TransferFunction(num, den, fs)

    if isinstance(system, Expr):
        params = kwargs.get("params", dict())
        num, den = fraction(system)
        fs = system.free_symbols.difference(params.keys())
        if len(fs) > 1:
            raise ValueError(f"Too many free symbols: {fs}")
        elif len(fs) == 0:
            raise ValueError(
                "An expression with one free symbol is required.")
        return TransferFunction(num, den, fs.pop())

    raise TypeError(f"type(system) = {type(system)} not recognized.")


def _check_system(system):
    """Function to check whether the dynamical system passed for plots is
    compatible or not."""
    if not isinstance(system, SISOLinearTimeInvariant):
        raise NotImplementedError(
            "Only SISO LTI systems are currently supported.")
    sys = system.to_expr()
    if sys.has(exp):
        # Should test that exp is not part of a constant, in which case
        # no exception is required, compare exp(s) with s*exp(1)
        raise NotImplementedError("Time delay terms are not supported.")


def _get_zeros_poles_from_symbolic_tf(system):
    s = system.var
    num_poly = Poly(system.num, s)
    den_poly = Poly(system.den, s)

    zeros = roots(num_poly, s)
    if sum(zeros.values()) != degree(num_poly, s):
        raise ValueError(
            "Coult not compute all the roots of the numerator of the "
            "transfer function.")
    poles = roots(den_poly, s)
    if sum(poles.values()) != degree(den_poly, s):
        raise ValueError(
            "Coult not compute all the roots of the denominator of the "
            "transfer function.")
    zeros = list(zeros.keys())
    poles = list(poles.keys())
    return zeros, poles


def control_axis(hor=True, ver=True, rendering_kw=None, **kwargs):
    """Create two axis lines to be used with control-plotting.

    Parameters
    ==========
    hor, ver : bool, optional
        Wheter to add the horizontal and/or vertical axis.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

    Returns
    =======

    A list containing up to two instances of ``HVLineSeries``.

    """
    s = []
    if hor:
        s.append(
            HVLineSeries(
                0, True, show_in_legend=False, rendering_kw=rendering_kw))
    if ver:
        s.append(
            HVLineSeries(
                0, False, show_in_legend=False, rendering_kw=rendering_kw))
    return s


def _pole_zero_helper(
    system, label, multiple_systems, pole_markersize, zero_markersize,
    **kwargs
):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)
    system = system.doit()  # Get the equivalent TransferFunction object.

    if len(system.free_symbols) == 1:
        s = system.var
        num_poly = Poly(system.num, s)
        den_poly = Poly(system.den, s)
        num_poly = np.array(num_poly.all_coeffs(), dtype=np.complex128)
        den_poly = np.array(den_poly.all_coeffs(), dtype=np.complex128)
        zeros = np.roots(num_poly)
        poles = np.roots(den_poly)
        zeros_re, zeros_im = np.real(zeros), np.imag(zeros)
        poles_re, poles_im = np.real(poles), np.imag(poles)
    else:
        zeros, poles = _get_zeros_poles_from_symbolic_tf(system)
        zeros_re, zeros_im = [re(z) for z in zeros], [im(z) for z in zeros]
        poles_re, poles_im = [re(p) for p in poles], [im(p) for p in poles]

    params = kwargs.get("params", {})
    Backend = kwargs.get("backend", TWO_D_B)

    z_rendering_kw = kwargs.pop("z_rendering_kw", {})
    p_rendering_kw = kwargs.pop("p_rendering_kw", {})
    z_kw, p_kw = {}, {}
    if hasattr(Backend, "_library") and (Backend._library == "matplotlib"):
        z_kw = dict(marker="o", markersize=zero_markersize)
        p_kw = dict(marker="x", markersize=pole_markersize)
        zero_color = kwargs.pop("zero_color", None)
        if zero_color:
            z_kw["color"] = zero_color
        pole_color = kwargs.pop("pole_color", None)
        if pole_color:
            z_kw["color"] = pole_color
    z_rendering_kw = merge(z_kw, z_rendering_kw)
    p_rendering_kw = merge(p_kw, p_rendering_kw)

    get_label = lambda t: t if label is None else t + " of " + label
    z_series = List2DSeries(
        zeros_re, zeros_im, get_label("zeros"),
        scatter=True, is_filled=True, rendering_kw=z_rendering_kw,
        params=params
    )
    p_series = List2DSeries(
        poles_re, poles_im, get_label("poles"),
        scatter=True, is_filled=True, rendering_kw=p_rendering_kw,
        params=params
    )
    return [p_series, z_series]


def pole_zero(
    system, pole_markersize=10, zero_markersize=7, show_axes=False,
    label=None, **kwargs
):
    """
    Computes the [Pole-Zero]_ plot (also known as PZ Plot or PZ Map) of
    a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    pole_color : str, tuple, optional
        The color of the pole points on the plot.
    pole_markersize : Number, optional
        The size of the markers used to mark the poles in the plot.
        Default pole markersize is 10.
    zero_color : str, tuple, optional
        The color of the zero points on the plot.
    zero_markersize : Number, optional
        The size of the markers used to mark the zeros in the plot.
        Default zero markersize is 7.
    z_rendering_kw : dict
        A dictionary of keyword arguments to further customize the appearance
        of zeros.
    p_rendering_kw : dict
        A dictionary of keyword arguments to further customize the appearance
        of poles.
    label : str, optional
        The label to be shown on the legend.
    **kwargs :
        See ``plot`` for a list of keyword arguments to further customize
        the resulting figure.

    Returns
    =======

    A list containing two instances of ``List2DSeries``.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import *
        >>> tf1 = TransferFunction(
        ...     s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        >>> graphics(
        ...     pole_zero(tf1),
        ...     xlabel="Real", ylabel="Imaginary"
        ... )
        Plot object containing:
        [0]: 2D list plot
        [1]: 2D list plot

    Interactive-widgets plot of multiple systems, one of which is parametric:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, d, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *
       tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
       tf2 = TransferFunction(s**2 + b, s**4 + a*s**3 + b*s**2 + c*s + d, s)
       params = {
           a: (3, 0, 5),
           b: (5, 0, 10),
           c: (4, 0, 8),
           d: (3, 0, 5),
       }
       graphics(
           control_axis(),
           pole_zero(tf1, label="A"),
           pole_zero(tf2, label="B", params=params),
           grid=False, xlim=(-4, 1), ylim=(-4, 4), use_latex=False,
           xlabel="Real", ylabel="Imaginary")

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot
    """
    series = _pole_zero_helper(
        system, label, False,
        pole_markersize, zero_markersize, **kwargs.copy()
    )
    return series


def _step_response_helper(
    system, label, lower_limit, upper_limit, prec, **kwargs
):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    expr = system.to_expr() / system.var
    expr = apart(expr, system.var, full=True)

    _x = Dummy("x")
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    # if `params` is given, _y might contain RootSum, which is not implemented
    # in Numpy. `doit()` is going to expand it, so that Numpy can be used.
    _y = _y.doit()

    return LineOver1DRangeSeries(
        _y, prange(_x, lower_limit, upper_limit),
        label, **kwargs
    )


def step_response(
    system, lower_limit=0, upper_limit=10, prec=8,
    label=None, rendering_kw=None, **kwargs
):
    """
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one instance of ``LineOver1DRangeSeries``.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import *
        >>> tf1 = TransferFunction(
        ...     8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> graphics(
        ...     step_response(tf1),
        ...     xlabel="Time [s]", ylabel="Amplitude"
        ... )   # doctest: +SKIP


    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit`` and ``upper_limit``.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy.abc import a, b, c, d, e, f, g, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *
       tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
       tf2 = TransferFunction(s**2 + a*s + b, s**3 + c*s**2 + d*s + e, s)
       params = {
           a: (3.7, 0, 5),
           b: (10, 0, 20),
           c: (7, 0, 8),
           d: (6, 0, 25),
           e: (16, 0, 25),
           # NOTE: remove `None` if using ipywidgets
           f: (0, 0, 10, 50, None, "lower limit"),
           g: (10, 0, 25, 50, None, "upper limit"),
       }
       graphics(
           step_response(
               tf1, label="A", lower_limit=f, upper_limit=g, params=params),
           step_response(
               tf2, label="B", lower_limit=f, upper_limit=g, params=params),
           use_latex=False, xlabel="Time [s]", ylabel="Amplitude"
       )

    See Also
    ========

    impulse_response, ramp_response

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError(
            "Lower limit of time must be greater than or equal to zero."
        )

    series = [
        _step_response_helper(
            system, label, lower_limit, upper_limit, prec,
            rendering_kw=rendering_kw, **kwargs
        )
    ]
    return series


def _impulse_response_helper(
    system, label, lower_limit, upper_limit, prec, **kwargs
):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    _x = Dummy("x")
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    _y = _y.doit()

    return LineOver1DRangeSeries(
        _y, prange(_x, lower_limit, upper_limit),
        label, **kwargs
    )


def impulse_response(
    system, prec=8, lower_limit=0, upper_limit=10,
    label=None, rendering_kw=None, **kwargs
):
    """
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one instance of ``LineOver1DRangeSeries``.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import *
        >>> tf1 = TransferFunction(
        ...     8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> graphics(
        ...     impulse_response(tf1),
        ...     xlabel="Time [s]", ylabel="Amplitude"
        ... )   # doctest: +SKIP

    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit`` and ``upper_limit``.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy.abc import a, b, c, d, e, f, g, h, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *
       tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
       tf2 = TransferFunction(a*s**2 + b*s + c, s**3 + d*s**2 + e*s + f, s)
       params = {
           a: (4, 0, 10),
           b: (24, 0, 40),
           c: (50, 0, 50),
           d: (3, 0, 25),
           e: (12.5, 0, 25),
           f: (17.5, 0, 50),
           # NOTE: remove `None` if using ipywidgets
           g: (0, 0, 10, 50, None, "lower limit"),
           h: (8, 0, 25, 50, None, "upper limit"),
       }
       graphics(
           impulse_response(
               tf1, label="A", lower_limit=g, upper_limit=h, params=params),
           impulse_response(
               tf2, label="B", lower_limit=g, upper_limit=h, params=params),
           use_latex=False, xlabel="Time [s]", ylabel="Amplitude"
       )


    See Also
    ========

    step_response, ramp_response

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.impulse.html

    """
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError(
            "Lower limit of time must be greater than or equal to zero."
        )

    return [
        _impulse_response_helper(
            system, label, lower_limit, upper_limit, prec,
            rendering_kw=rendering_kw, **kwargs
        )
    ]


def _ramp_response_helper(
    system, label, lower_limit, upper_limit, prec, slope, **kwargs
):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    _x = Dummy("x")
    expr = (slope*system.to_expr()) / ((system.var)**2)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    _y = _y.doit()

    return LineOver1DRangeSeries(
        _y, prange(_x, lower_limit, upper_limit),
        label, **kwargs
    )


def ramp_response(
    system, prec=8, slope=1, lower_limit=0, upper_limit=10,
    label=None, rendering_kw=None, **kwargs
):
    """
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line passing through origin
    ($f(x) = mx$). The slope of the ramp function can be varied by the
    user and the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one instance of ``LineOver1DRangeSeries``.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import *
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> graphics(
        ...     ramp_response(tf1, upper_limit=2),
        ...     xlabel="Time [s]", ylabel="Amplitude"
        ... )    # doctest: +SKIP

    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit``, ``upper_limit`` and ``slope``.

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy.abc import a, b, c, d, e, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *
       tf1 = TransferFunction(s, (s+4)*(s+8), s)
       tf2 = TransferFunction(s, (s+a)*(s+b), s)
       params = {
           a: (6, 0, 10),
           b: (7, 0, 10),
           # NOTE: remove `None` if using ipywidgets
           c: (1, 0, 10, 50, None, "slope"),
           d: (0, 0, 5, 50, None, "lower limit"),
           e: (5, 2, 10, 50, None, "upper limit"),
       }
       graphics(
           ramp_response(
               tf1, label="A", slope=c, lower_limit=d, upper_limit=e,
               params=params),
           ramp_response(
               tf2, label="B", slope=c, lower_limit=d, upper_limit=e,
               params=params),
           xlabel="Time [s]", ylabel="Amplitude", use_latex=False)

    See Also
    ========

    plot_step_response, plot_impulse_response

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError(
            "Lower limit of time must be greater than or equal to zero."
        )

    return [
        _ramp_response_helper(
            system, label, lower_limit, upper_limit, prec, slope,
            rendering_kw=rendering_kw, **kwargs
        )
    ]


def _bode_magnitude_helper(
    system, label, initial_exp, final_exp, freq_unit, **kwargs
):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    expr = system.to_expr()
    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    mag = 20*log(Abs(w_expr), 10)
    return LineOver1DRangeSeries(
        mag, prange(_w, 10**initial_exp, 10**final_exp),
        label, xscale='log', **kwargs
    )


def bode_magnitude(
    system, initial_exp=-5, final_exp=5, freq_unit='rad/sec',
    phase_unit='rad', label=None, rendering_kw=None, **kwargs
):
    """
    Returns the Bode magnitude plot of a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'``
        (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree)
        as phase units.
    unwrap : bool, optional
        Depending on the transfer function, there could be discontinuities in
        the phase plot. Set ``unwrap=True`` to get a continuous phase.
        Default to False.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one instance of ``LineOver1DRangeSeries``.

    Notes
    =====

    :func:`~spb.plot_functions.control.plot_bode` returns a
    :func:`~spb.plotgrid.plotgrid` of two visualizations, one with
    the Bode magnitude, the other with the Bode phase.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import *
        >>> tf1 = TransferFunction(
        ...     1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        >>> graphics(
        ...     bode_magnitude(tf1, initial_exp=0.2, final_exp=0.7),
        ...     xscale="log", xlabel="Frequency [rad/s]",
        ...     ylabel="Magnitude [dB]"
        ... )   # doctest: +SKIP

    Interactive-widget plot:

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy.abc import a, b, c, d, e, f, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *
       tf1 = TransferFunction(a*s**2 + b*s + c, d*s**4 + e*s**3 + f*s**2, s)
       params = {
           a: (0.5, -10, 10),
           b: (0.1, -1, 1),
           c: (8, -10, 10),
           d: (10, -10, 10),
           e: (0.1, -1, 1),
           f: (1, -10, 10),
       }
       graphics(
           bode_magnitude(tf1, initial_exp=-2, final_exp=2, params=params),
           imodule="panel", ncols=3, use_latex=False,
           xscale="log", xlabel="Frequency [rad/s]", ylabel="Magnitude [dB]"
       )

    See Also
    ========

    bode_phase, nyquist, nichols, spb.plot_functions.control.plot_bode

    """
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError(
            'Only "rad/sec" and "Hz" are accepted frequency units.'
        )

    return [
        _bode_magnitude_helper(
            system, label, initial_exp, final_exp,
            freq_unit, rendering_kw=rendering_kw, **kwargs
        )
    ]


def _bode_phase_helper(
    system, label, initial_exp, final_exp, freq_unit, phase_unit, **kwargs
):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    expr = system.to_expr()
    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    if phase_unit == 'deg':
        phase = arg(w_expr)*180/pi
    else:
        phase = arg(w_expr)

    return LineOver1DRangeSeries(
        phase, prange(_w, 10**initial_exp, 10**final_exp),
        label, xscale='log', **kwargs
    )


def bode_phase(
    system, initial_exp=-5, final_exp=5, freq_unit='rad/sec',
    phase_unit='rad', label=None, rendering_kw=None, **kwargs
):
    """
    Returns the Bode phase plot of a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'``
        (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree)
        as phase units.
    unwrap : bool, optional
        Depending on the transfer function, there could be discontinuities in
        the phase plot. Set ``unwrap=True`` to get a continuous phase.
        Default to False.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one instance of ``LineOver1DRangeSeries``.

    Notes
    =====

    :func:`~spb.plot_functions.control.plot_bode` returns a
    :func:`~spb.plotgrid.plotgrid` of two visualizations, one with
    the Bode magnitude, the other with the Bode phase.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import *
        >>> tf1 = TransferFunction(
        ...     1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        >>> graphics(
        ...     bode_phase(tf1, initial_exp=0.2, final_exp=0.7),
        ...     xscale="log", xlabel="Frequency [rad/s]",
        ...     ylabel="Magnitude [dB]"
        ... )   # doctest: +SKIP

    Interactive-widget plot:

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy.abc import a, b, c, d, e, f, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *
       tf1 = TransferFunction(a*s**2 + b*s + c, d*s**4 + e*s**3 + f*s**2, s)
       params = {
           a: (0.5, -10, 10),
           b: (0.1, -1, 1),
           c: (8, -10, 10),
           d: (10, -10, 10),
           e: (0.1, -1, 1),
           f: (1, -10, 10),
       }
       graphics(
           bode_phase(tf1, initial_exp=-2, final_exp=2, params=params),
           imodule="panel", ncols=3, use_latex=False,
           xscale="log", xlabel="Frequency [rad/s]", ylabel="Magnitude [dB]"
       )

    See Also
    ========

    bode_magnitude, nyquist, nichols

    """
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError(
            'Only "rad/sec" and "Hz" are accepted frequency units.')

    return [
        _bode_phase_helper(
            system, label, initial_exp, final_exp,
            freq_unit, phase_unit, rendering_kw=rendering_kw, **kwargs
        )
    ]


def _compute_range_helper(system, **kwargs):
    omega_limits = kwargs.get("omega_limits", None)
    s = system.var

    if not omega_limits:
        # find a suitable omega range
        # NOTE: the following procedure follows what is implemented in:
        # https://github.com/python-control/python-control/blob/main/control/freqplot.py
        # specifically inside _default_frequency_range()
        # Here it is adapted for symbolic computation, which allows
        # interactive plots.
        zeros, poles = _get_zeros_poles_from_symbolic_tf(system)
        features = zeros + poles
        # Get rid of poles and zeros at the origin
        features = [f for f in features if f.evalf() is not S.Zero]

        # don't use Abs to compute magnitude, because errors can be raised when
        # substituting into the piecewise function
        magnitude = lambda t: sqrt(re(t)**2 + im(t)**2)
        features = [magnitude(f) for f in features]
        # Make sure there is at least one point in the range
        if len(features) == 0:
            features = [1]
        features = [log(f, 10) for f in features]
        rint = Piecewise((floor(s), (frac(s) < S.Half)), (ceiling(s), True))

        feature_periphery_decades = 2
        lsp_min = rint.subs(
            s, Min(*features, evaluate=False) - feature_periphery_decades)
        lsp_max = rint.subs(
            s, Max(*features, evaluate=False) + feature_periphery_decades)
        _range = prange(s, 10**lsp_min, 10**lsp_max)
    else:
        _range = prange(s, *omega_limits)

    return _range, omega_limits


def _nyquist_helper(system, label, **kwargs):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)
    _range, omega_limits = _compute_range_helper(system, **kwargs)
    kwargs.setdefault("xscale", "log")
    kwargs.setdefault("use_cm", False)
    kwargs.setdefault("omega_range_given", not (omega_limits is None))
    return NyquistLineSeries(system, _range, label, **kwargs)


def nyquist(system, label=None, **kwargs):
    """Nyquist plot for a system

    Plots a Nyquist plot for the system over a (optional) frequency range.
    The curve is computed by evaluating the Nyqist segment along the positive
    imaginary axis, with a mirror image generated to reflect the negative
    imaginary axis.  Poles on or near the imaginary axis are avoided using a
    small indentation.  The portion of the Nyquist contour at infinity is not
    explicitly computed (since it maps to a constant value for any system with
    a proper transfer function).

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    label : str, optional
        The label to be shown on the legend.
    arrows : int or 1D/2D array of floats, optional
        Specify the number of arrows to plot on the Nyquist curve.  If an
        integer is passed, that number of equally spaced arrows will be
        plotted on each of the primary segment and the mirror image.  If a 1D
        array is passed, it should consist of a sorted list of floats between
        0 and 1, indicating the location along the curve to plot an arrow.
    encirclement_threshold : float, optional
        Define the threshold for generating a warning if the number of net
        encirclements is a non-integer value.  Default value is 0.05.
    indent_direction : str, optional
        For poles on the imaginary axis, set the direction of indentation to
        be 'right' (default), 'left', or 'none'.
    indent_points : int, optional
        Number of points to insert in the Nyquist contour around poles that
        are at or near the imaginary axis.
    indent_radius : float, optional
        Amount to indent the Nyquist contour around poles on or near the
        imaginary axis. Portions of the Nyquist plot corresponding to indented
        portions of the contour are plotted using a different line style.
    max_curve_magnitude : float, optional
        Restrict the maximum magnitude of the Nyquist plot to this value.
        Portions of the Nyquist plot whose magnitude is restricted are
        plotted using a different line style.
    max_curve_offset : float, optional
        When plotting scaled portion of the Nyquist plot, increase/decrease
        the magnitude by this fraction of the max_curve_magnitude to allow
        any overlaps between the primary and mirror curves to be avoided.
    mirror_style : [str, str] or [dict, dict] or dict or False, optional
        Linestyles for mirror image of the Nyquist curve. If a list is given,
        the first element is used for unscaled portions of the Nyquist curve,
        the second element is used for portions that are scaled
        (using max_curve_magnitude). `dict` is a dictionary of keyword
        arguments to be passed to the plotting function, for example to
        `plt.plot`. If `False` then omit completely.
        Default linestyle is `['--', ':']`.
    m_circles : bool, optional
        Turn on/off [M-circles]_, which are circles of constant closed loop
        magnitude.
    primary_style : [str, str] or [dict, dict] or dict, optional
        Linestyles for primary image of the Nyquist curve. If a list is given,
        the first element is used for unscaled portions of the Nyquist curve,
        the second element is used for portions that are scaled
        (using max_curve_magnitude). `dict` is a dictionary of keyword
        arguments to be passed to the plotting function, for example to
        Matplotlib's `plt.plot`. Default linestyle is `['-', '-.']`.
    omega_limits : array_like of two values, optional
        Limits to the range of frequencies.
    start_marker : str or dict, optional
        Marker to use to mark the starting point of the Nyquist plot. If
        `dict` is provided, it must containts keyword arguments to be passed
        to the plot function, for example to Matplotlib's `plt.plot`.
    warn_encirclements : bool, optional
        If set to 'False', turn off warnings about number of encirclements not
        meeting the Nyquist criterion.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line_parametric_2d`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one instance of ``NyquistLineSeries``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hall_circles

    See Also
    ========

    bode, nichols

    Notes
    =====

    1. If a continuous-time system contains poles on or near the imaginary
       axis, a small indentation will be used to avoid the pole.  The radius
       of the indentation is given by `indent_radius` and it is taken to the
       right of stable poles and the left of unstable poles.  If a pole is
       exactly on the imaginary axis, the `indent_direction` parameter can be
       used to set the direction of indentation.  Setting `indent_direction`
       to `none` will turn off indentation.  If `return_contour` is True, the
       exact contour used for evaluation is returned.

    2. For those portions of the Nyquist plot in which the contour is
       indented to avoid poles, resuling in a scaling of the Nyquist plot,
       the line styles are according to the settings of the `primary_style`
       and `mirror_style` keywords.  By default the scaled portions of the
       primary curve use a dotted line style and the scaled portion of the
       mirror image use a dashdot line style.

    Examples
    ========

    Plotting a single transfer function:

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import Rational
       >>> from sympy.abc import s
       >>> from sympy.physics.control.lti import TransferFunction
       >>> from spb import *
       >>> tf1 = TransferFunction(
       ...     4 * s**2 + 5 * s + 1, 3 * s**2 + 2 * s + 5, s)
       >>> graphics(
       ...     nyquist(tf1),
       ...     xlabel="Real", ylabel="Imaginary"
       ... )                                # doctest: +SKIP

    Visualizing M-circles:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

    >>> graphics(
    ...     nyquist(tf1, m_circles=True),
    ...     xlabel="Real", ylabel="Imaginary"
    ... )

    Interactive-widgets plot of a systems:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, d, e, f, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *
       tf = TransferFunction(a * s**2 + b * s + c, d**2 * s**2 + e * s + f, s)
       params = {
           a: (2, 0, 10),
           b: (5, 0, 10),
           c: (1, 0, 10),
           d: (1, 0, 10),
           e: (2, 0, 10),
           f: (3, 0, 10),
       }
       graphics(
           nyquist(tf, params=params),
           xlabel="Real", ylabel="Imaginary", use_latex=False,
           xlim=(-1, 4), ylim=(-2.5, 2.5), aspect="equal"
       )

    """
    return [
        _nyquist_helper(system, label, **kwargs.copy())
    ]


def _nichols_helper(system, label, **kwargs):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)
    s = system.var
    omega = Dummy("omega")
    _range, omega_limits = _compute_range_helper(system, **kwargs)
    _range = prange(omega, *_range[1:])

    system_expr = system.to_expr()
    system_expr = system_expr.subs(s, I * omega)

    kwargs.setdefault("use_cm", False)
    kwargs.setdefault("xscale", "log")
    return NicholsLineSeries(
        arg(system_expr), Abs(system_expr), _range, label, **kwargs)


def nichols(system, label=None, rendering_kw=None, **kwargs):
    """Nichols plot for a system over a (optional) frequency range.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
    ngrid : bool, optional
        Turn on/off the [Nichols]_ grid lines.
    omega_limits : array_like of two values, optional
        Limits to the range of frequencies.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line_parametric_2d`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one instance of ``NicholsLineSeries``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hall_circles

    Examples
    ========

    Plotting a single transfer function:

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy.abc import s
       >>> from sympy.physics.control.lti import TransferFunction
       >>> from spb import *
       >>> tf = TransferFunction(50*s**2 - 20*s + 15, -10*s**2 + 40*s + 30, s)
       >>> graphics(
       ...     nichols(tf),
       ...     xlabel="Open-Loop Phase [deg]",
       ...     ylabel="Open-Loop Magnitude [dB]",
       ...     grid=False
       ... )    # doctest: +SKIP

    Turning off the Nichols grid lines:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     nichols(tf, ngrid=False),
       ...     xlabel="Open-Loop Phase [deg]",
       ...     ylabel="Open-Loop Magnitude [dB]",
       ...     grid=False
       ... )    # doctest: +SKIP

    Interactive-widgets plot of a systems. For these kind of plots, it is
    recommended to set both ``omega_limits`` and ``xlim``:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, s
       from spb import *
       from sympy.physics.control.lti import TransferFunction
       tf = TransferFunction(a*s**2 + b*s + c, s**3 + 10*s**2 + 5 * s + 1, s)
       params = {
           a: (-25, -100, 100),
           b: (60, -300, 300),
           c: (-100, -1000, 1000),
       }
       graphics(
           nichols(tf, omega_limits=[1e-03, 1e03], n=1e04, params=params),
           xlabel="Open-Loop Phase [deg]",
           ylabel="Open-Loop Magnitude [dB]",
           xlim=(-360, 360), grid=False, use_latex=False
       )

    See Also
    ========

    bode_magnitude, bode_phase, nyquist

    """
    return [
        _nichols_helper(
            system, label, rendering_kw=rendering_kw, **kwargs.copy()
        )
    ]
