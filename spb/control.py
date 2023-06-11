from spb import plotgrid
from spb.defaults import TWO_D_B
from spb.interactive import create_interactive_plot
from spb.series import List2DSeries, LineOver1DRangeSeries, HVLineSeries
from spb.utils import _instantiate_backend
import numpy as np
from sympy import (roots, exp, Poly, degree, re, im, latex, apart, Dummy,
    I, log, Abs, arg
)
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from mergedeep import merge


# TODO:
# if a series have label="something" and show_in_legend=False, MB still
# shows it on the legend...
# interactive widget plotgrid
# latex on title only if backend supports it... use lambda func for title
# interactive prange with lower_limit and upper_limit
# https://www.mathworks.com/help/control/ug/nichols-plot-design.html
# https://www.mathworks.com/help/control/ref/dynamicsystem.nicholsplot.html
# https://www.mathworks.com/matlabcentral/answers/22783-bode-diagram-to-nichols-curve
# https://www.mathworks.com/help/ident/ug/how-to-plot-bode-and-nyquist-plots-at-the-command-line.html
# https://www.mathworks.com/help/ident/ref/dynamicsystem.nyquist.html
# https://www.mathworks.com/help/ident/ref/dynamicsystem.nyquistplot.html
# https://www.mathworks.com/help/ident/ref/idlti.spectrum.html
# https://www.mathworks.com/help/ident/ref/dynamicsystem.bodeplot.html


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
    title = base
    if len(systems) == 1:
        label = systems[0][1]
        if label == "System 1":
            label = f"${latex(systems[0][0])}$"
        title = base + f" of {label}"
    return title


def _pole_zero_helper(system, label, multiple_systems,
    pole_markersize, zero_markersize, **kwargs):
    _check_system(system)
    system = system.doit()  # Get the equivalent TransferFunction object.

    s = system.var
    num_poly = Poly(system.num, s)
    den_poly = Poly(system.den, s)

    if len(system.free_symbols) == 1:
        num_poly = np.array(num_poly.all_coeffs(), dtype=np.complex128)
        den_poly = np.array(den_poly.all_coeffs(), dtype=np.complex128)
        zeros = np.roots(num_poly)
        poles = np.roots(den_poly)
        zeros_re, zeros_im = np.real(zeros), np.imag(zeros)
        poles_re, poles_im = np.real(poles), np.imag(poles)
    else:
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
       :small-size: 800, 625

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
    
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
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
    

    Interactive-widgets plot of multiple systems, one of which is parametric:

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy.abc import a, b, c, d, e, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_step_response
       tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
       tf2 = TransferFunction(s**2 + a*s + b, s**3 + c*s**2 + d*s + e, s)
       plot_step_response(
           (tf1, "A"), (tf2, "B"),
           params={
               a: (3.7, 0, 5),
               b: (10, 0, 20),
               c: (7, 0, 8),
               d: (6, 0, 25),
               e: (16, 0, 25),
           })

    See Also
    ========

    impulse_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    
    if lower_limit < 0:
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

    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
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
    
    Interactive-widgets plot of multiple systems, one of which is parametric:

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy.abc import a, b, c, d, e, f, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_impulse_response
       tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
       tf2 = TransferFunction(a*s**2 + b*s + c, s**3 + d*s**2 + e*s + f, s)
       plot_impulse_response(
           (tf1, "A"), (tf2, "B"),
           params={
               a: (4, 0, 10),
               b: (24, 0, 40),
               c: (50, 0, 50),
               d: (3, 0, 25),
               e: (12.5, 0, 25),
               f: (17.5, 0, 50),
           }) 
       

    See Also
    ========

    step_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.impulse.html

    """
    if lower_limit < 0:
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

    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
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

    Interactive-widgets plot of multiple systems, one of which is parametric:

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy.abc import a, b, s
       from sympy.physics.control.lti import TransferFunction
       from spb.control import plot_ramp_response
       tf1 = TransferFunction(s, (s+4)*(s+8), s)
       tf2 = TransferFunction(s, (s+a)*(s+b), s)
       plot_ramp_response(
           (tf1, "A"), (tf2, "B"), upper_limit=2,
           params={
               a: (6, 0, 10),
               b: (7, 0, 10)
           })

    See Also
    ========

    step_response_plot, impulse_response_plot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    if slope < 0:
        raise ValueError("Slope must be greater than or equal"
            " to zero.")
    if lower_limit < 0:
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
    return LineOver1DRangeSeries(mag, (_w, 10**initial_exp, 10**final_exp),
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

    return LineOver1DRangeSeries(phase, (_w, 10**initial_exp, 10**final_exp),
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
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.

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
