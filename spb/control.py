from spb.defaults import TWO_D_B
from spb.interactive import create_interactive_plot
from spb.plotgrid import plotgrid
from spb.series import (
    List2DSeries, LineOver1DRangeSeries, HVLineSeries, NyquistLineSeries
)
from spb.utils import _instantiate_backend, prange
import numpy as np
from sympy import (roots, exp, Poly, degree, re, im, latex, apart, Dummy,
    I, log, Abs, arg, sympify, S, Min, Max, Piecewise, sqrt,
    floor, ceiling, frac
)
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from mergedeep import merge


__all__ = [
    'pole_zero_plot', 'plot_pole_zero',
    'step_response_plot', 'step_response_plot',
    'impulse_response_plot', 'plot_impulse_response',
    'ramp_response_plot', 'plot_ramp_response',
    'bode_magnitude_plot', 'plot_bode_magnitude',
    'bode_phase_plot', 'plot_bode_phase',
    'bode_plot', 'plot_bode'
]


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


def _create_axes_series():
    """Create two data series representing straight lines for the axes,
    crossing at (0, 0).
    """
    hor = HVLineSeries(0, True, show_in_legend=False)
    ver = HVLineSeries(0, False, show_in_legend=False)
    return [hor, ver]


def _create_plot_helper(series, show_axes, **kwargs):
    if show_axes:
        series = _create_axes_series() + series

    Backend = kwargs.pop("backend", TWO_D_B)
    if kwargs.get("params", None):
        return create_interactive_plot(*series, backend=Backend, **kwargs)

    return _instantiate_backend(Backend, *series, **kwargs)


def _unpack_systems(systems):
    """Unpack `systems` into `[(sys1, label1), (sys2, label2), ...]`.
    """
    if (len(systems) > 1) and isinstance(systems[0], dict):
        raise ValueError(
            "Received a list of systems in which the first item is a "
            "dictionary. This configuration is not supported.")
    if isinstance(systems[0], dict):
        systems = list(systems[0].items())
    if not isinstance(systems[0], (list, tuple)):
        labels = [f"System {i+1}" for i in range(len(systems))]
        systems = [(system, label) for system, label in zip(systems, labels)]
    return systems


def _create_title_helper(systems, base):
    """Create a suitable title depending on the number of systems being shown
    and wheter the backend supports latex or not.
    """
    def func(wrapper, use_latex):
        title = base
        if len(systems) == 1:
            label = systems[0][1]
            if label == "System 1":
                print_func = latex if use_latex else str
                label = wrapper % f"{print_func(systems[0][0])}"
            title = base + f" of {label}"
        return title
    return func


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


def _pole_zero_helper(system, label, multiple_systems,
    pole_markersize, zero_markersize, **kwargs):
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

    get_label = lambda t: t if not multiple_systems else t + " of " + label
    z_series = List2DSeries(zeros_re, zeros_im, get_label("zeros"),
        is_point=True, is_filled=True, rendering_kw=z_rendering_kw,
        params=params)
    p_series = List2DSeries(poles_re, poles_im, get_label("poles"),
        is_point=True, is_filled=True, rendering_kw=p_rendering_kw,
        params=params)
    return [p_series, z_series]


def plot_pole_zero(*systems, pole_markersize=10, zero_markersize=7, show_axes=False, **kwargs):
    """
    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a sequence of LTI SISO systems.
        * a sequence of 2-tuples ``(LTI SISO system, label)``.
        * a dict mapping LTI SISO systems to labels.
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
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    z_rendering_kw : dict
        A dictionary of keyword arguments to further customize the appearance
        of zeros.
    p_rendering_kw : dict
        A dictionary of keyword arguments to further customize the appearance
        of poles.
    **kwargs :
        See ``plot`` for a list of keyword arguments to further customize
        the resulting figure.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb.control import pole_zero_plot
        >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        >>> pole_zero_plot(tf1)   # doctest: +SKIP

    Interactive-widgets plot of multiple systems, one of which is parametric:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, d, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_pole_zero
       tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
       tf2 = TransferFunction(s**2 + b, s**4 + a*s**3 + b*s**2 + c*s + d, s)
       plot_pole_zero(
           (tf1, "A"), (tf2, "B"),
           params={
               a: (3, 0, 5),
               b: (5, 0, 10),
               c: (4, 0, 8),
               d: (3, 0, 5),
           },
           xlim=(-4, 1), ylim=(-4, 4),
           show_axes=True, use_latex=False)

    See Also
    ========

    pole_zero_numerical_data

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot
    """
    systems = _unpack_systems(systems)
    ms = len(systems) > 1
    series = []
    for system, label in systems:
        series.extend(_pole_zero_helper(system, label, ms,
            pole_markersize, zero_markersize, **kwargs.copy()))

    kwargs.setdefault("xlabel", "Real axis")
    kwargs.setdefault("ylabel", "Imaginary axis")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Poles and Zeros"))
    return _create_plot_helper(series, show_axes, **kwargs)


pole_zero_plot = plot_pole_zero


def _unit_response_helper(system, label, lower_limit, upper_limit,
    prec, **kwargs):
    _check_system(system)

    expr = system.to_expr() / system.var
    expr = apart(expr, system.var, full=True)

    _x = Dummy("x")
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    # if `params` is given, _y might contain RootSum, which is not implemented
    # in Numpy. `doit()` is going to expand it, so that Numpy can be used.
    _y = _y.doit()

    return LineOver1DRangeSeries(_y, prange(_x, lower_limit, upper_limit),
        label, **kwargs)


def plot_step_response(*systems, lower_limit=0, upper_limit=10,
    prec=8, show_axes=False, **kwargs):
    """
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Step Response is to be computed.
        It can be:

        * a single LTI SISO system.
        * a sequence of LTI SISO systems.
        * a sequence of 2-tuples ``(LTI SISO system, label)``.
        * a dict mapping LTI SISO systems to labels.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    **kwargs :
        See ``plot`` for a list of keyword arguments to further customize
        the resulting figure.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb.control import step_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> step_response_plot(tf1)   # doctest: +SKIP


    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit`` and ``upper_limit``.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy.abc import a, b, c, d, e, f, g, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_step_response
       tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
       tf2 = TransferFunction(s**2 + a*s + b, s**3 + c*s**2 + d*s + e, s)
       plot_step_response(
           (tf1, "A"), (tf2, "B"), lower_limit=f, upper_limit=g,
           params={
               a: (3.7, 0, 5),
               b: (10, 0, 20),
               c: (7, 0, 8),
               d: (6, 0, 25),
               e: (16, 0, 25),
               # NOTE: remove `None` if using ipywidgets
               f: (0, 0, 10, 50, None, "lower limit"),
               g: (10, 0, 25, 50, None, "upper limit"),
           })

    See Also
    ========

    impulse_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")

    systems = _unpack_systems(systems)
    series = [_unit_response_helper(s, l,
        lower_limit, upper_limit, prec, **kwargs) for s, l in systems]

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Unit Response"))
    return _create_plot_helper(series, show_axes, **kwargs)


step_response_plot = plot_step_response


def _impulse_response_helper(system, label, lower_limit, upper_limit,
    prec, **kwargs):
    _check_system(system)

    _x = Dummy("x")
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    _y = _y.doit()

    return LineOver1DRangeSeries(_y, prange(_x, lower_limit, upper_limit),
        label, **kwargs)


def plot_impulse_response(*systems, prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, **kwargs):
    """
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Impulse Response is to be computed.
        It can be:

        * a single LTI SISO system.
        * a sequence of LTI SISO systems.
        * a sequence of 2-tuples ``(LTI SISO system, label)``.
        * a dict mapping LTI SISO systems to labels.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    **kwargs :
        See ``plot`` for a list of keyword arguments to further customize
        the resulting figure.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb.control import impulse_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> impulse_response_plot(tf1)   # doctest: +SKIP

    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit`` and ``upper_limit``.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy.abc import a, b, c, d, e, f, g, h, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_impulse_response
       tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
       tf2 = TransferFunction(a*s**2 + b*s + c, s**3 + d*s**2 + e*s + f, s)
       plot_impulse_response(
           (tf1, "A"), (tf2, "B"), lower_limit=g, upper_limit=h,
           params={
               a: (4, 0, 10),
               b: (24, 0, 40),
               c: (50, 0, 50),
               d: (3, 0, 25),
               e: (12.5, 0, 25),
               f: (17.5, 0, 50),
               # NOTE: remove `None` if using ipywidgets
               g: (0, 0, 10, 50, None, "lower limit"),
               h: (8, 0, 25, 50, None, "upper limit"),
           })


    See Also
    ========

    step_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.impulse.html

    """
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")

    systems = _unpack_systems(systems)
    series = [_impulse_response_helper(s, l,
        lower_limit, upper_limit, prec, **kwargs) for s, l in systems]

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Impulse Response"))
    return _create_plot_helper(series, show_axes, **kwargs)


impulse_response_plot = plot_impulse_response


def _ramp_response_helper(system, label, lower_limit, upper_limit,
    prec, slope, **kwargs):
    _check_system(system)

    _x = Dummy("x")
    expr = (slope*system.to_expr()) / ((system.var)**2)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    _y = _y.doit()

    return LineOver1DRangeSeries(_y, prange(_x, lower_limit, upper_limit),
        label, **kwargs)


def plot_ramp_response(*systems, slope=1, prec=8,
    lower_limit=0, upper_limit=10, show_axes=False, **kwargs):
    """
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line
    passing through origin ($f(x) = mx$). The slope of
    the ramp function can be varied by the user and
    the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Ramp Response is to be computed.
        It can be:

        * a single LTI SISO system.
        * a sequence of LTI SISO systems.
        * a sequence of 2-tuples ``(LTI SISO system, label)``.
        * a dict mapping LTI SISO systems to labels.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    **kwargs :
        See ``plot`` for a list of keyword arguments to further customize
        the resulting figure.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb.control import ramp_response_plot
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> ramp_response_plot(tf1, upper_limit=2)   # doctest: +SKIP

    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit``, ``upper_limit`` and ``slope``.

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy.abc import a, b, c, d, e, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_ramp_response
       tf1 = TransferFunction(s, (s+4)*(s+8), s)
       tf2 = TransferFunction(s, (s+a)*(s+b), s)
       plot_ramp_response(
           (tf1, "A"), (tf2, "B"),
           slope=c, lower_limit=d, upper_limit=e,
           params={
               a: (6, 0, 10),
               b: (7, 0, 10),
               # NOTE: remove `None` if using ipywidgets
               c: (1, 0, 10, 50, None, "slope"),
               d: (0, 0, 5, 50, None, "lower limit"),
               e: (5, 2, 10, 50, None, "upper limit"),
           })

    See Also
    ========

    step_response_plot, impulse_response_plot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    # allows parametric slope
    slope = sympify(slope)
    if slope.is_Number and slope < 0:
        raise ValueError("Slope must be greater than or equal"
            " to zero.")
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")

    systems = _unpack_systems(systems)
    series = [_ramp_response_helper(s, l,
        lower_limit, upper_limit, prec, slope, **kwargs) for s, l in systems]

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Ramp Response"))
    return _create_plot_helper(series, show_axes, **kwargs)


ramp_response_plot = plot_ramp_response


def _bode_magnitude_helper(system, label, initial_exp, final_exp,
    freq_unit, **kwargs):
    _check_system(system)

    expr = system.to_expr()
    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    mag = 20*log(Abs(w_expr), 10)
    return LineOver1DRangeSeries(mag,
        prange(_w, 10**initial_exp, 10**final_exp),
        label, xscale='log', **kwargs)


def plot_bode_magnitude(*systems, initial_exp=-5, final_exp=5,
    freq_unit='rad/sec', show_axes=False, **kwargs):
    """
    Returns the Bode magnitude plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    systems = _unpack_systems(systems)
    series = [_bode_magnitude_helper(s, l, initial_exp, final_exp,
        freq_unit, **kwargs) for s, l in systems]

    kwargs.setdefault("xlabel", 'Frequency [%s]' % freq_unit)
    kwargs.setdefault("ylabel", 'Magnitude (dB)')
    kwargs.setdefault("title", _create_title_helper(
        systems, "Bode Plot (Magnitude)"))
    kwargs.setdefault("xscale", "log")
    return _create_plot_helper(series, show_axes, **kwargs)


bode_magnitude_plot = plot_bode_magnitude


def _bode_phase_helper(system, label, initial_exp, final_exp,
    freq_unit, phase_unit, **kwargs):
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

    return LineOver1DRangeSeries(phase,
        prange(_w, 10**initial_exp, 10**final_exp),
        label, xscale='log', **kwargs)


def plot_bode_phase(*systems, initial_exp=-5, final_exp=5,
    freq_unit='rad/sec', phase_unit='rad', show_axes=False, **kwargs):
    """
    Returns the Bode phase plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    freq_units = ('rad/sec', 'Hz')
    phase_units = ('rad', 'deg')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')
    if phase_unit not in phase_units:
        raise ValueError('Only "rad" and "deg" are accepted phase units.')

    systems = _unpack_systems(systems)
    series = [_bode_phase_helper(s, l, initial_exp, final_exp,
        freq_unit, phase_unit, **kwargs) for s, l in systems]

    kwargs.setdefault("xlabel", 'Frequency [%s]' % freq_unit)
    kwargs.setdefault("ylabel", 'Phase [%s]' % phase_unit)
    kwargs.setdefault("title", _create_title_helper(
        systems, "Bode Plot (Phase)"))
    kwargs.setdefault("xscale", "log")
    return _create_plot_helper(series, show_axes, **kwargs)


bode_phase_plot = plot_bode_phase


def plot_bode(*systems, initial_exp=-5, final_exp=5,
    freq_unit='rad/sec', phase_unit='rad', show_axes=False, **kwargs):
    """
    Returns the Bode phase and magnitude plots of a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Bode Plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a sequence of LTI SISO systems.
        * a sequence of 2-tuples ``(LTI SISO system, label)``.
        * a dict mapping LTI SISO systems to labels.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'``
        (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree)
        as phase units.
    **kwargs :
        See ``plot`` for a list of keyword arguments to further customize
        the resulting figure.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb.control import bode_plot
        >>> tf1 = TransferFunction(1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        >>> bode_plot(tf1, initial_exp=0.2, final_exp=0.7)   # doctest: +SKIP

    See Also
    ========

    bode_magnitude_plot, bode_phase_plot

    """
    if kwargs.get("params", None):
        raise NotImplementedError(
            "`plot_bode` internally uses `plotgrid`, which doesn't support "
            "interactive widgets plots.")

    show = kwargs.pop("show", True)
    kwargs["show"] = False
    p1 = plot_bode_magnitude(*systems, show_axes=show_axes,
        initial_exp=initial_exp, final_exp=final_exp,
        freq_unit=freq_unit, **kwargs)
    p2 = plot_bode_phase(*systems, show_axes=show_axes,
        initial_exp=initial_exp, final_exp=final_exp,
        freq_unit=freq_unit, phase_unit=phase_unit,
        title="", **kwargs)

    systems = _unpack_systems(systems)
    title = "Bode Plot"
    if len(systems) == 1:
        title = f'Bode Plot of ${latex(systems[0][0])}$'
    p1.title = title
    p = plotgrid(p1, p2, show=False)

    if show:
        p.show()
    return p


bode_plot = plot_bode


def _nyquist_helper(system, label, **kwargs):
    _check_system(system)

    omega_limits = kwargs.get("omega_limits", None)
    omega = Dummy("omega", real=True)
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
        # print("2", features)
        features = [log(f, 10) for f in features]
        rint = Piecewise((floor(s), (frac(s) < S.Half)), (ceiling(s), True))

        feature_periphery_decades = 2
        lsp_min = rint.subs(s, Min(*features, evaluate=False) - feature_periphery_decades)
        lsp_max = rint.subs(s, Max(*features, evaluate=False) + feature_periphery_decades)
        _range = prange(s, 10**lsp_min, 10**lsp_max)
    else:
        _range = prange(s, *omega_limits)

    kwargs.setdefault("xscale", "log")
    kwargs.setdefault("use_cm", False)
    kwargs.setdefault("omega_range_given", not (omega_limits is None))
    return NyquistLineSeries(system, _range, label, **kwargs)


def plot_nyquist(*systems, **kwargs):
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

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Bode Plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a sequence of LTI SISO systems.
        * a sequence of 2-tuples ``(LTI SISO system, label)``.
        * a dict mapping LTI SISO systems to labels.

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
        Turn on/off M-circles, which are circles of constant closed loop
        magnitude. Refer to [#fn1]_ for more information.

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
        See ``plot_parametric`` for a list of keyword arguments to further
        customize the resulting figure.

    References
    ==========

    .. [#fn1] https://en.wikipedia.org/wiki/Hall_circles

    https://en.wikipedia.org/wiki/Hall_circles

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
       >>> from spb.control import plot_nyquist
       >>> tf1 = TransferFunction(4 * s**2 + 5 * s + 1, 3 * s**2 + 2 * s + 5, s)
       >>> plot_nyquist(tf1)

    Visualizing M-circles:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_nyquist(tf1, m_circles=True)

    Plotting multiple transfer functions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> tf2 = TransferFunction(1, s + Rational(1, 3), s)
       >>> plot_nyquist(tf1, tf2)

    Interactive-widgets plot of a systems:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, d, e, f, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_nyquist
       tf = TransferFunction(a * s**2 + b * s + c, d**2 * s**2 + e * s + f, s)
       plot_nyquist(
           tf,
           params={
               a: (2, 0, 10),
               b: (5, 0, 10),
               c: (1, 0, 10),
               d: (1, 0, 10),
               e: (2, 0, 10),
               f: (3, 0, 10),
           },
           m_circles=False, use_latex=False,
           xlim=(-1, 4), ylim=(-2.5, 2.5), aspect="equal"
       )

    """
    systems = _unpack_systems(systems)
    series = [_nyquist_helper(s, l, **kwargs.copy()) for s, l in systems]
    kwargs.setdefault("xlabel", "Real axis")
    kwargs.setdefault("ylabel", "Imaginary axis")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Nyquist Plot"))
    kwargs.setdefault("grid", not kwargs.get("m_circles", False))
    return _create_plot_helper(series, False, **kwargs)


nyquist_plot = plot_nyquist


# from control import StateSpace, TransferFunction
# from control.ctrlutil import unwrap
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# def nyquist_plot2(
#         syslist, omega=None, plot=True, omega_limits=None, omega_num=None,
#         label_freq=0, color=None, return_contour=False,
#         warn_encirclements=True, warn_nyquist=True, **kwargs):
#     """Nyquist plot for a system

#     Plots a Nyquist plot for the system over a (optional) frequency range.
#     The curve is computed by evaluating the Nyqist segment along the positive
#     imaginary axis, with a mirror image generated to reflect the negative
#     imaginary axis.  Poles on or near the imaginary axis are avoided using a
#     small indentation.  The portion of the Nyquist contour at infinity is not
#     explicitly computed (since it maps to a constant value for any system with
#     a proper transfer function).

#     Parameters
#     ----------
#     syslist : list of LTI
#         List of linear input/output systems (single system is OK). Nyquist
#         curves for each system are plotted on the same graph.

#     NO -> omega : array_like, optional
#         Set of frequencies to be evaluated, in rad/sec.

#     omega_limits : array_like of two values, optional
#         Limits to the range of frequencies. Ignored if omega is provided, and
#         auto-generated if omitted.

#     omega_num : int, optional
#         Number of frequency samples to plot.  Defaults to
#         config.defaults['freqplot.number_of_samples'].

#     NO -> plot : boolean, optional
#         If True (default), plot the Nyquist plot.

#     NO -> color : string, optional
#         Used to specify the color of the line and arrowhead.

#     NO -> return_contour : bool, optional
#         If 'True', return the contour used to evaluate the Nyquist plot.

#     **kwargs : :func:`matplotlib.pyplot.plot` keyword properties, optional
#         Additional keywords (passed to `matplotlib`)

#     Returns
#     -------
#     count : int (or list of int if len(syslist) > 1)
#         Number of encirclements of the point -1 by the Nyquist curve.  If
#         multiple systems are given, an array of counts is returned.

#     NO -> contour : ndarray (or list of ndarray if len(syslist) > 1)), optional
#         The contour used to create the primary Nyquist curve segment, returned
#         if `return_contour` is Tue.  To obtain the Nyquist curve values,
#         evaluate system(s) along contour.

#     Other Parameters
#     ----------------
#     arrows : int or 1D/2D array of floats, optional
#         Specify the number of arrows to plot on the Nyquist curve.  If an
#         integer is passed. that number of equally spaced arrows will be
#         plotted on each of the primary segment and the mirror image.  If a 1D
#         array is passed, it should consist of a sorted list of floats between
#         0 and 1, indicating the location along the curve to plot an arrow.  If
#         a 2D array is passed, the first row will be used to specify arrow
#         locations for the primary curve and the second row will be used for
#         the mirror image.

#     arrow_size : float, optional
#         Arrowhead width and length (in display coordinates).  Default value is
#         8 and can be set using config.defaults['nyquist.arrow_size'].

#     arrow_style : matplotlib.patches.ArrowStyle, optional
#         Define style used for Nyquist curve arrows (overrides `arrow_size`).

#     encirclement_threshold : float, optional
#         Define the threshold for generating a warning if the number of net
#         encirclements is a non-integer value.  Default value is 0.05 and can
#         be set using config.defaults['nyquist.encirclement_threshold'].

#     indent_direction : str, optional
#         For poles on the imaginary axis, set the direction of indentation to
#         be 'right' (default), 'left', or 'none'.

#     indent_points : int, optional
#         Number of points to insert in the Nyquist contour around poles that
#         are at or near the imaginary axis.

#     indent_radius : float, optional
#         Amount to indent the Nyquist contour around poles on or near the
#         imaginary axis. Portions of the Nyquist plot corresponding to indented
#         portions of the contour are plotted using a different line style.

#     label_freq : int, optiona
#         Label every nth frequency on the plot.  If not specified, no labels
#         are generated.

#     max_curve_magnitude : float, optional
#         Restrict the maximum magnitude of the Nyquist plot to this value.
#         Portions of the Nyquist plot whose magnitude is restricted are
#         plotted using a different line style.

#     max_curve_offset : float, optional
#         When plotting scaled portion of the Nyquist plot, increase/decrease
#         the magnitude by this fraction of the max_curve_magnitude to allow
#         any overlaps between the primary and mirror curves to be avoided.

#     mirror_style : [str, str] or False
#         Linestyles for mirror image of the Nyquist curve.  The first element
#         is used for unscaled portions of the Nyquist curve, the second element
#         is used for portions that are scaled (using max_curve_magnitude).  If
#         `False` then omit completely.  Default linestyle (['--', ':']) is
#         determined by config.defaults['nyquist.mirror_style'].

#     primary_style : [str, str], optional
#         Linestyles for primary image of the Nyquist curve.  The first
#         element is used for unscaled portions of the Nyquist curve,
#         the second element is used for portions that are scaled (using
#         max_curve_magnitude).  Default linestyle (['-', '-.']) is
#         determined by config.defaults['nyquist.mirror_style'].

#     start_marker : str, optional
#         Matplotlib marker to use to mark the starting point of the Nyquist
#         plot.  Defaults value is 'o' and can be set using
#         config.defaults['nyquist.start_marker'].

#     start_marker_size : float, optional
#         Start marker size (in display coordinates).  Default value is
#         4 and can be set using config.defaults['nyquist.start_marker_size'].

#     warn_nyquist : bool, optional
#         If set to 'False', turn off warnings about frequencies above Nyquist.

#     warn_encirclements : bool, optional
#         If set to 'False', turn off warnings about number of encirclements not
#         meeting the Nyquist criterion.

#     Notes
#     -----
#     1. If a discrete time model is given, the frequency response is computed
#        along the upper branch of the unit circle, using the mapping ``z =
#        exp(1j * omega * dt)`` where `omega` ranges from 0 to `pi/dt` and `dt`
#        is the discrete timebase.  If timebase not specified (``dt=True``),
#        `dt` is set to 1.

#     2. If a continuous-time system contains poles on or near the imaginary
#        axis, a small indentation will be used to avoid the pole.  The radius
#        of the indentation is given by `indent_radius` and it is taken to the
#        right of stable poles and the left of unstable poles.  If a pole is
#        exactly on the imaginary axis, the `indent_direction` parameter can be
#        used to set the direction of indentation.  Setting `indent_direction`
#        to `none` will turn off indentation.  If `return_contour` is True, the
#        exact contour used for evaluation is returned.

#     3. For those portions of the Nyquist plot in which the contour is
#        indented to avoid poles, resuling in a scaling of the Nyquist plot,
#        the line styles are according to the settings of the `primary_style`
#        and `mirror_style` keywords.  By default the scaled portions of the
#        primary curve use a dotted line style and the scaled portion of the
#        mirror image use a dashdot line style.

#     Examples
#     --------
#     >>> G = ct.zpk([], [-1, -2, -3], gain=100)
#     >>> ct.nyquist_plot(G)
#     2

#     """
#     # Check to see if legacy 'Plot' keyword was used
#     if 'Plot' in kwargs:
#         warnings.warn("'Plot' keyword is deprecated in nyquist_plot; "
#                       "use 'plot'", FutureWarning)
#         # Map 'Plot' keyword to 'plot' keyword
#         plot = kwargs.pop('Plot')

#     # Check to see if legacy 'labelFreq' keyword was used
#     if 'labelFreq' in kwargs:
#         warnings.warn("'labelFreq' keyword is deprecated in nyquist_plot; "
#                       "use 'label_freq'", FutureWarning)
#         # Map 'labelFreq' keyword to 'label_freq' keyword
#         label_freq = kwargs.pop('labelFreq')

#     # Check to see if legacy 'arrow_width' or 'arrow_length' were used
#     if 'arrow_width' in kwargs or 'arrow_length' in kwargs:
#         warnings.warn(
#             "'arrow_width' and 'arrow_length' keywords are deprecated in "
#             "nyquist_plot; use `arrow_size` instead", FutureWarning)
#         kwargs['arrow_size'] = \
#             (kwargs.get('arrow_width', 0) + kwargs.get('arrow_length', 0)) / 2
#         kwargs.pop('arrow_width', False)
#         kwargs.pop('arrow_length', False)

#     # Get values for params (and pop from list to allow keyword use in plot)
#     omega_num_given = omega_num is not None
#     # omega_num = kwargs.pop("omega_num", 1000)
#     arrows = kwargs.pop("arrows", 2)
#     arrow_size = 8
#     arrow_style = None
#     indent_radius = 1e-04
#     encirclement_threshold = 0.05
#     indent_direction = 'right'
#     indent_points = 50
#     max_curve_magnitude = 20
#     max_curve_offset = 0.02
#     start_marker = "o"
#     start_marker_size = 4
#     primary_style = ['-', '-.']
#     mirror_style = ['--', ':']

#     # print("omega_num", omega_num)

#     # Set line styles for the curves
#     # def _parse_linestyle(style_name, allow_false=False):
#     #     style = config._get_param(
#     #         'nyquist', style_name, kwargs, _nyquist_defaults, pop=True)
#     #     if isinstance(style, str):
#     #         # Only one style provided, use the default for the other
#     #         style = [style, _nyquist_defaults['nyquist.' + style_name][1]]
#     #         warnings.warn(
#     #             "use of a single string for linestyle will be deprecated "
#     #             " in a future release", PendingDeprecationWarning)
#     #     if (allow_false and style is False) or \
#     #        (isinstance(style, list) and len(style) == 2):
#     #         return style
#     #     else:
#     #         raise ValueError(f"invalid '{style_name}': {style}")

#     # primary_style = _parse_linestyle('primary_style')
#     # mirror_style = _parse_linestyle('mirror_style', allow_false=True)

#     # If argument was a singleton, turn it into a tuple
#     if not isinstance(syslist, (list, tuple)):
#         syslist = (syslist,)

#     # Determine the range of frequencies to use, based on args/features
#     omega, omega_range_given = _determine_omega_vector(
#         syslist, omega, omega_limits, omega_num, feature_periphery_decades=2)

#     print("casso", len(omega))
#     # print(omega)

#     # If omega was not specified explicitly, start at omega = 0
#     if not omega_range_given:
#         print("not omega_range_given", len(omega))
#         if omega_num_given:
#             print("\ta")
#             # Just reset the starting point
#             omega[0] = 0.0
#         else:
#             print("\tb")
#             # Insert points between the origin and the first frequency point
#             omega = np.concatenate((
#                 np.linspace(0, omega[0], indent_points), omega[1:]))
#         print("\t", len(omega))
#         # print("\t", omega[:100])

#     # Go through each system and keep track of the results
#     counts, contours = [], []
#     for sys in syslist:
#         # print("sys", sys)
#         if not sys.issiso():
#             # TODO: Add MIMO nyquist plots.
#             raise ControlMIMONotImplemented(
#                 "Nyquist plot currently only supports SISO systems.")

#         # Figure out the frequency range
#         omega_sys = np.asarray(omega)

#         # # Determine the contour used to evaluate the Nyquist curve
#         # if sys.isdtime(strict=True):
#         #     # Restrict frequencies for discrete-time systems
#         #     nyquistfrq = math.pi / sys.dt
#         #     if not omega_range_given:
#         #         # limit up to and including nyquist frequency
#         #         omega_sys = np.hstack((
#         #             omega_sys[omega_sys < nyquistfrq], nyquistfrq))

#         #     # Issue a warning if we are sampling above Nyquist
#         #     if np.any(omega_sys * sys.dt > np.pi) and warn_nyquist:
#         #         warnings.warn("evaluation above Nyquist frequency")

#         # do indentations in s-plane where it is more convenient
#         splane_contour = 1j * omega_sys

#         # Bend the contour around any poles on/near the imaginary axis
#         if isinstance(sys, (StateSpace, TransferFunction)) \
#                 and indent_direction != 'none':
#             if sys.isctime():
#                 splane_poles = sys.poles()
#                 splane_cl_poles = sys.feedback().poles()
#                 # print("splane_poles", splane_poles)
#                 # print("splane_cl_poles", splane_cl_poles)
#             else:
#                 # map z-plane poles to s-plane. We ignore any at the origin
#                 # to avoid numerical warnings because we know we
#                 # don't need to indent for them
#                 zplane_poles = sys.poles()
#                 zplane_poles = zplane_poles[~np.isclose(abs(zplane_poles), 0.)]
#                 splane_poles = np.log(zplane_poles) / sys.dt

#                 zplane_cl_poles = sys.feedback().poles()
#                 # eliminate z-plane poles at the origin to avoid warnings
#                 zplane_cl_poles = zplane_cl_poles[
#                     ~np.isclose(abs(zplane_cl_poles), 0.)]
#                 splane_cl_poles = np.log(zplane_cl_poles) / sys.dt

#             #
#             # Check to make sure indent radius is small enough
#             #
#             # If there is a closed loop pole that is near the imaginary axis
#             # at a point that is near an open loop pole, it is possible that
#             # indentation might skip or create an extraneous encirclement.
#             # We check for that situation here and generate a warning if that
#             # could happen.
#             #
#             for p_cl in splane_cl_poles:
#                 # See if any closed loop poles are near the imaginary axis
#                 if abs(p_cl.real) <= indent_radius:
#                     # See if any open loop poles are close to closed loop poles
#                     if len(splane_poles) > 0:
#                         p_ol = splane_poles[
#                             (np.abs(splane_poles - p_cl)).argmin()]

#                         if abs(p_ol - p_cl) <= indent_radius and \
#                                 warn_encirclements:
#                             warnings.warn(
#                                 "indented contour may miss closed loop pole; "
#                                 "consider reducing indent_radius to below "
#                                 f"{abs(p_ol - p_cl):5.2g}", stacklevel=2)

#             #
#             # See if we should add some frequency points near imaginary poles
#             #
#             for p in splane_poles:
#                 # See if we need to process this pole (skip if on the negative
#                 # imaginary axis or not near imaginary axis + user override)
#                 if p.imag < 0 or abs(p.real) > indent_radius or \
#                    omega_range_given:
#                     continue

#                 # Find the frequencies before the pole frequency
#                 below_points = np.argwhere(
#                     splane_contour.imag - abs(p.imag) < -indent_radius)
#                 if below_points.size > 0:
#                     first_point = below_points[-1].item()
#                     start_freq = p.imag - indent_radius
#                 else:
#                     # Add the points starting at the beginning of the contour
#                     assert splane_contour[0] == 0
#                     first_point = 0
#                     start_freq = 0

#                 # Find the frequencies after the pole frequency
#                 above_points = np.argwhere(
#                     splane_contour.imag - abs(p.imag) > indent_radius)
#                 last_point = above_points[0].item()
#                 # print("first_point", first_point)
#                 # print("last_point", last_point)

#                 # Add points for half/quarter circle around pole frequency
#                 # (these will get indented left or right below)
#                 splane_contour = np.concatenate((
#                     splane_contour[0:first_point+1],
#                     (1j * np.linspace(
#                         start_freq, p.imag + indent_radius, indent_points)),
#                     splane_contour[last_point:]))

#             # Indent points that are too close to a pole
#             if len(splane_poles) > 0: # accomodate no splane poles if dtime sys
#                 for i, s in enumerate(splane_contour):
#                     # Find the nearest pole
#                     p = splane_poles[(np.abs(splane_poles - s)).argmin()]

#                     # See if we need to indent around it
#                     if abs(s - p) < indent_radius:
#                         # Figure out how much to offset (simple trigonometry)
#                         offset = np.sqrt(indent_radius ** 2 - (s - p).imag ** 2) \
#                             - (s - p).real

#                         # Figure out which way to offset the contour point
#                         if p.real < 0 or (p.real == 0 and
#                                         indent_direction == 'right'):
#                             # Indent to the right
#                             splane_contour[i] += offset

#                         elif p.real > 0 or (p.real == 0 and
#                                             indent_direction == 'left'):
#                             # Indent to the left
#                             splane_contour[i] -= offset

#                         else:
#                             raise ValueError("unknown value for indent_direction")

#         # print("splane_contour", len(splane_contour), splane_contour)
#         # change contour to z-plane if necessary
#         if sys.isctime():
#             contour = splane_contour
#         else:
#             contour = np.exp(splane_contour * sys.dt)

#         # Compute the primary curve
#         resp = sys(contour)
#         # print("resp", resp)

#         # Compute CW encirclements of -1 by integrating the (unwrapped) angle
#         phase = -unwrap(np.angle(resp + 1))
#         encirclements = np.sum(np.diff(phase)) / np.pi
#         count = int(np.round(encirclements, 0))

#         print("phase", phase)
#         print("encirclements", encirclements)
#         print("count", count)
#         print("warn_encirclements", warn_encirclements)
#         print("encirclement_threshold", encirclement_threshold)
#         print(abs(encirclements - count))

#         # Let the user know if the count might not make sense
#         if abs(encirclements - count) > encirclement_threshold and \
#            warn_encirclements:
#             warnings.warn(
#                 "number of encirclements was a non-integer value; this can"
#                 " happen is contour is not closed, possibly based on a"
#                 " frequency range that does not include zero.")

#         #
#         # Make sure that the enciriclements match the Nyquist criterion
#         #
#         # If the user specifies the frequency points to use, it is possible
#         # to miss enciriclements, so we check here to make sure that the
#         # Nyquist criterion is actually satisfied.
#         #
#         if isinstance(sys, (StateSpace, TransferFunction)):
#             # Count the number of open/closed loop RHP poles
#             if sys.isctime():
#                 print("A")
#                 if indent_direction == 'right':
#                     print("A1")
#                     P = (sys.poles().real > 0).sum()
#                 else:
#                     print("A2")
#                     P = (sys.poles().real >= 0).sum()
#                 Z = (sys.feedback().poles().real >= 0).sum()
#                 print("sys.feedback()", sys.feedback())
#                 print("sys.feedback().poles()", sys.feedback().poles())
#                 print("sys.feedback().poles().real", sys.feedback().poles().real)
#             else:
#                 print("B")
#                 if indent_direction == 'right':
#                     print("B1")
#                     P = (np.abs(sys.poles()) > 1).sum()
#                 else:
#                     print("B2")
#                     P = (np.abs(sys.poles()) >= 1).sum()
#                 Z = (np.abs(sys.feedback().poles()) >= 1).sum()

#             print("P", P)
#             print("Z", Z)
#             print("Z != count + P", Z != count + P)

#             # Check to make sure the results make sense; warn if not
#             if Z != count + P and warn_encirclements:
#                 warnings.warn(
#                     "number of encirclements does not match Nyquist criterion;"
#                     " check frequency range and indent radius/direction",
#                     UserWarning, stacklevel=2)
#             elif indent_direction == 'none' and any(sys.poles().real == 0) and \
#                  warn_encirclements:
#                 warnings.warn(
#                     "system has pure imaginary poles but indentation is"
#                     " turned off; results may be meaningless",
#                     RuntimeWarning, stacklevel=2)

#         counts.append(count)
#         contours.append(contour)

#         if plot:
#             # Parse the arrows keyword
#             if not arrows:
#                 arrow_pos = []
#             elif isinstance(arrows, int):
#                 N = arrows
#                 # Space arrows out, starting midway along each "region"
#                 arrow_pos = np.linspace(0.5/N, 1 + 0.5/N, N, endpoint=False)
#             elif isinstance(arrows, (list, np.ndarray)):
#                 arrow_pos = np.sort(np.atleast_1d(arrows))
#             else:
#                 raise ValueError("unknown or unsupported arrow location")

#             # Set the arrow style
#             if arrow_style is None:
#                 arrow_style = mpl.patches.ArrowStyle(
#                     'simple', head_width=arrow_size, head_length=arrow_size)

#             # Find the different portions of the curve (with scaled pts marked)
#             reg_mask = np.logical_or(
#                 np.abs(resp) > max_curve_magnitude,
#                 splane_contour.real != 0)
#             # reg_mask = np.logical_or(
#             #     np.abs(resp.real) > max_curve_magnitude,
#             #     np.abs(resp.imag) > max_curve_magnitude)

#             scale_mask = ~reg_mask \
#                 & np.concatenate((~reg_mask[1:], ~reg_mask[-1:])) \
#                 & np.concatenate((~reg_mask[0:1], ~reg_mask[:-1]))

#             # Rescale the points with large magnitude
#             rescale = np.logical_and(
#                 reg_mask, abs(resp) > max_curve_magnitude)
#             resp[rescale] *= max_curve_magnitude / abs(resp[rescale])

#             # Plot the regular portions of the curve (and grab the color)
#             x_reg = np.ma.masked_where(reg_mask, resp.real)
#             y_reg = np.ma.masked_where(reg_mask, resp.imag)
#             p = plt.plot(
#                 x_reg, y_reg, primary_style[0], color=color, **kwargs)
#             c = p[0].get_color()

#             # Figure out how much to offset the curve: the offset goes from
#             # zero at the start of the scaled section to max_curve_offset as
#             # we move along the curve
#             curve_offset = _compute_curve_offset(
#                 resp, scale_mask, max_curve_offset)

#             # Plot the scaled sections of the curve (changing linestyle)
#             x_scl = np.ma.masked_where(scale_mask, resp.real)
#             y_scl = np.ma.masked_where(scale_mask, resp.imag)
#             if x_scl.count() >= 1 and y_scl.count() >= 1:
#                 plt.plot(
#                     x_scl * (1 + curve_offset),
#                     y_scl * (1 + curve_offset),
#                     primary_style[1], color=c, **kwargs)

#             # Plot the primary curve (invisible) for setting arrows
#             x, y = resp.real.copy(), resp.imag.copy()
#             x[reg_mask] *= (1 + curve_offset[reg_mask])
#             y[reg_mask] *= (1 + curve_offset[reg_mask])
#             p = plt.plot(x, y, linestyle='None', color=c, **kwargs)

#             # Add arrows
#             ax = plt.gca()
#             _add_arrows_to_line2D(
#                 ax, p[0], arrow_pos, arrowstyle=arrow_style, dir=1)

#             # Plot the mirror image
#             if mirror_style is not False:
#                 # Plot the regular and scaled segments
#                 plt.plot(
#                     x_reg, -y_reg, mirror_style[0], color=c, **kwargs)
#                 if x_scl.count() >= 1 and y_scl.count() >= 1:
#                     plt.plot(
#                         x_scl * (1 - curve_offset),
#                         -y_scl * (1 - curve_offset),
#                         mirror_style[1], color=c, **kwargs)

#                 # Add the arrows (on top of an invisible contour)
#                 x, y = resp.real.copy(), resp.imag.copy()
#                 x[reg_mask] *= (1 - curve_offset[reg_mask])
#                 y[reg_mask] *= (1 - curve_offset[reg_mask])
#                 p = plt.plot(x, -y, linestyle='None', color=c, **kwargs)
#                 _add_arrows_to_line2D(
#                     ax, p[0], arrow_pos, arrowstyle=arrow_style, dir=-1)

#             # Mark the start of the curve
#             if start_marker:
#                 plt.plot(resp[0].real, resp[0].imag, start_marker,
#                          color=c, markersize=start_marker_size)

#             # Mark the -1 point
#             plt.plot([-1], [0], 'r+')

#             # Label the frequencies of the points
#             if label_freq:
#                 ind = slice(None, None, label_freq)
#                 for xpt, ypt, omegapt in zip(x[ind], y[ind], omega_sys[ind]):
#                     # Convert to Hz
#                     f = omegapt / (2 * np.pi)

#                     # Factor out multiples of 1000 and limit the
#                     # result to the range [-8, 8].
#                     pow1000 = max(min(get_pow1000(f), 8), -8)

#                     # Get the SI prefix.
#                     prefix = gen_prefix(pow1000)

#                     # Apply the text. (Use a space before the text to
#                     # prevent overlap with the data.)
#                     #
#                     # np.round() is used because 0.99... appears
#                     # instead of 1.0, and this would otherwise be
#                     # truncated to 0.
#                     plt.text(xpt, ypt, ' ' +
#                              str(int(np.round(f / 1000 ** pow1000, 0))) + ' ' +
#                              prefix + 'Hz')

#     if plot:
#         ax = plt.gca()
#         ax.set_xlabel("Real axis")
#         ax.set_ylabel("Imaginary axis")
#         ax.grid(color="lightgray")

#     # "Squeeze" the results
#     if len(syslist) == 1:
#         counts, contours = counts[0], contours[0]

#     # Return counts and (optionally) the contour we used
#     return splane_contour, resp
#     return (counts, contours) if return_contour else counts


# # Internal function to add arrows to a curve
# def _add_arrows_to_line2D(
#         axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
#         arrowstyle='-|>', arrowsize=1, dir=1, transform=None):
#     """
#     Add arrows to a matplotlib.lines.Line2D at selected locations.

#     Parameters:
#     -----------
#     axes: Axes object as returned by axes command (or gca)
#     line: Line2D object as returned by plot command
#     arrow_locs: list of locations where to insert arrows, % of total length
#     arrowstyle: style of the arrow
#     arrowsize: size of the arrow
#     transform: a matplotlib transform instance, default to data coordinates

#     Returns:
#     --------
#     arrows: list of arrows

#     Based on https://stackoverflow.com/questions/26911898/

#     """
#     if not isinstance(line, mpl.lines.Line2D):
#         raise ValueError("expected a matplotlib.lines.Line2D object")
#     x, y = line.get_xdata(), line.get_ydata()

#     arrow_kw = {
#         "arrowstyle": arrowstyle,
#     }

#     color = line.get_color()
#     use_multicolor_lines = isinstance(color, np.ndarray)
#     if use_multicolor_lines:
#         raise NotImplementedError("multicolor lines not supported")
#     else:
#         arrow_kw['color'] = color

#     linewidth = line.get_linewidth()
#     if isinstance(linewidth, np.ndarray):
#         raise NotImplementedError("multiwidth lines not supported")
#     else:
#         arrow_kw['linewidth'] = linewidth

#     if transform is None:
#         transform = axes.transData

#     # Compute the arc length along the curve
#     s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

#     arrows = []
#     for loc in arrow_locs:
#         n = np.searchsorted(s, s[-1] * loc)

#         # Figure out what direction to paint the arrow
#         if dir == 1:
#             arrow_tail = (x[n], y[n])
#             arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
#         elif dir == -1:
#             # Orient the arrow in the other direction on the segment
#             arrow_tail = (x[n + 1], y[n + 1])
#             arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
#         else:
#             raise ValueError("unknown value for keyword 'dir'")

#         p = mpl.patches.FancyArrowPatch(
#             arrow_tail, arrow_head, transform=transform, lw=0,
#             **arrow_kw)
#         axes.add_patch(p)
#         arrows.append(p)
#     return arrows


# #
# # Function to compute Nyquist curve offsets
# #
# # This function computes a smoothly varying offset that starts and ends at
# # zero at the ends of a scaled segment.
# #
# def _compute_curve_offset(resp, mask, max_offset):
#     # Compute the arc length along the curve
#     s_curve = np.cumsum(
#         np.sqrt(np.diff(resp.real) ** 2 + np.diff(resp.imag) ** 2))

#     # Initialize the offset
#     offset = np.zeros(resp.size)
#     arclen = np.zeros(resp.size)

#     # Walk through the response and keep track of each continous component
#     i, nsegs = 0, 0
#     while i < resp.size:
#         # Skip the regular segment
#         while i < resp.size and mask[i]:
#             i += 1              # Increment the counter
#             if i == resp.size:
#                 break
#             # Keep track of the arclength
#             arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

#         nsegs += 0.5
#         if i == resp.size:
#             break

#         # Save the starting offset of this segment
#         seg_start = i

#         # Walk through the scaled segment
#         while i < resp.size and not mask[i]:
#             i += 1
#             if i == resp.size:  # See if we are done with this segment
#                 break
#             # Keep track of the arclength
#             arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

#         nsegs += 0.5
#         if i == resp.size:
#             break

#         # Save the ending offset of this segment
#         seg_end = i

#         # Now compute the scaling for this segment
#         s_segment = arclen[seg_end-1] - arclen[seg_start]
#         offset[seg_start:seg_end] = max_offset * s_segment/s_curve[-1] * \
#             np.sin(np.pi * (arclen[seg_start:seg_end]
#                             - arclen[seg_start])/s_segment)

#     return offset


# def _determine_omega_vector(syslist, omega_in, omega_limits, omega_num,
#                             Hz=None, feature_periphery_decades=None):
#     """Determine the frequency range for a frequency-domain plot
#     according to a standard logic.

#     If omega_in and omega_limits are both None, then omega_out is computed
#     on omega_num points according to a default logic defined by
#     _default_frequency_range and tailored for the list of systems syslist, and
#     omega_range_given is set to False.
#     If omega_in is None but omega_limits is an array-like of 2 elements, then
#     omega_out is computed with the function np.logspace on omega_num points
#     within the interval [min, max] =  [omega_limits[0], omega_limits[1]], and
#     omega_range_given is set to True.
#     If omega_in is not None, then omega_out is set to omega_in,
#     and omega_range_given is set to True

#     Parameters
#     ----------
#     syslist : list of LTI
#         List of linear input/output systems (single system is OK)
#     omega_in : 1D array_like or None
#         Frequency range specified by the user
#     omega_limits : 1D array_like or None
#         Frequency limits specified by the user
#     omega_num : int
#         Number of points to be used for the frequency
#         range (if the frequency range is not user-specified)
#     Hz : bool, optional
#         If True, the limits (first and last value) of the frequencies
#         are set to full decades in Hz so it fits plotting with logarithmic
#         scale in Hz otherwise in rad/s. Omega is always returned in rad/sec.

#     Returns
#     -------
#     omega_out : 1D array
#         Frequency range to be used
#     omega_range_given : bool
#         True if the frequency range was specified by the user, either through
#         omega_in or through omega_limits. False if both omega_in
#         and omega_limits are None.
#     """
#     omega_range_given = True

#     print("_determine_omega_vector", omega_num)

#     if omega_in is None:
#         if omega_limits is None:
#             print("\ta")
#             omega_range_given = False
#             # Select a default range if none is provided
#             omega_out = _default_frequency_range(
#                 syslist, number_of_samples=omega_num, Hz=Hz,
#                 feature_periphery_decades=feature_periphery_decades)
#             print("\t", len(omega_out))
#         else:
#             print("\tb")
#             omega_limits = np.asarray(omega_limits)
#             if len(omega_limits) != 2:
#                 raise ValueError("len(omega_limits) must be 2")
#             omega_out = np.logspace(np.log10(omega_limits[0]),
#                                     np.log10(omega_limits[1]),
#                                     num=omega_num, endpoint=True)
#     else:
#         print("\tc")
#         omega_out = np.copy(omega_in)

#     return omega_out, omega_range_given


# # Compute reasonable defaults for axes
# def _default_frequency_range(syslist, Hz=None, number_of_samples=None,
#                              feature_periphery_decades=None):
#     """Compute a default frequency range for frequency domain plots.

#     This code looks at the poles and zeros of all of the systems that
#     we are plotting and sets the frequency range to be one decade above
#     and below the min and max feature frequencies, rounded to the nearest
#     integer.  If no features are found, it returns logspace(-1, 1)

#     Parameters
#     ----------
#     syslist : list of LTI
#         List of linear input/output systems (single system is OK)
#     Hz : bool, optional
#         If True, the limits (first and last value) of the frequencies
#         are set to full decades in Hz so it fits plotting with logarithmic
#         scale in Hz otherwise in rad/s. Omega is always returned in rad/sec.
#     number_of_samples : int, optional
#         Number of samples to generate.  The default value is read from
#         ``config.defaults['freqplot.number_of_samples'].  If None, then the
#         default from `numpy.logspace` is used.
#     feature_periphery_decades : float, optional
#         Defines how many decades shall be included in the frequency range on
#         both sides of features (poles, zeros).  The default value is read from
#         ``config.defaults['freqplot.feature_periphery_decades']``.

#     Returns
#     -------
#     omega : array
#         Range of frequencies in rad/sec

#     Examples
#     --------
#     >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
#     >>> omega = ct._default_frequency_range(G)
#     >>> omega.min(), omega.max()
#     (0.1, 100.0)

#     """
#     # Set default values for options
#     if number_of_samples is None:
#         number_of_samples = 1000
#     feature_periphery_decades = 2

#     # Find the list of all poles and zeros in the systems
#     features = np.array(())
#     freq_interesting = []

#     # detect if single sys passed by checking if it is sequence-like
#     if not hasattr(syslist, '__iter__'):
#         syslist = (syslist,)

#     for sys in syslist:
#         try:
#             # Add new features to the list
#             if sys.isctime():
#                 features_ = np.concatenate(
#                     (np.abs(sys.poles()), np.abs(sys.zeros())))
#                 # Get rid of poles and zeros at the origin
#                 toreplace = np.isclose(features_, 0.0)
#                 if np.any(toreplace):
#                     features_ = features_[~toreplace]
#             elif sys.isdtime(strict=True):
#                 fn = math.pi * 1. / sys.dt
#                 # TODO: What distance to the Nyquist frequency is appropriate?
#                 freq_interesting.append(fn * 0.9)

#                 features_ = np.concatenate((sys.poles(), sys.zeros()))
#                 # Get rid of poles and zeros on the real axis (imag==0)
#                # * origin and real < 0
#                 # * at 1.: would result in omega=0. (logaritmic plot!)
#                 toreplace = np.isclose(features_.imag, 0.0) & (
#                                     (features_.real <= 0.) |
#                                     (np.abs(features_.real - 1.0) < 1.e-10))
#                 if np.any(toreplace):
#                     features_ = features_[~toreplace]
#                 # TODO: improve
#                 features_ = np.abs(np.log(features_) / (1.j * sys.dt))
#             else:
#                 # TODO
#                 raise NotImplementedError(
#                     "type of system in not implemented now")
#             features = np.concatenate((features, features_))
#         except NotImplementedError:
#             pass

#     # Make sure there is at least one point in the range
#     if features.shape[0] == 0:
#         features = np.array([1.])

#     if Hz:
#         features /= 2. * math.pi
#     features = np.log10(features)
#     lsp_min = np.rint(np.min(features) - feature_periphery_decades)
#     lsp_max = np.rint(np.max(features) + feature_periphery_decades)
#     if Hz:
#         lsp_min += np.log10(2. * math.pi)
#         lsp_max += np.log10(2. * math.pi)

#     if freq_interesting:
#         lsp_min = min(lsp_min, np.log10(min(freq_interesting)))
#         lsp_max = max(lsp_max, np.log10(max(freq_interesting)))

#     # TODO: Add a check in discrete case to make sure we don't get aliasing
#     # (Attention: there is a list of system but only one omega vector)

#     # Set the range to be an order of magnitude beyond any features
#     if number_of_samples:
#         omega = np.logspace(
#             lsp_min, lsp_max, num=number_of_samples, endpoint=True)
#     else:
#         omega = np.logspace(lsp_min, lsp_max, endpoint=True)
#     return omega
