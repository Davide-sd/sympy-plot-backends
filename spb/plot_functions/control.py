from spb.defaults import TWO_D_B
from spb.graphics.control import (
    _preprocess_system, _pole_zero_helper,
    _nichols_helper, _nyquist_helper, _step_response_helper,
    _ramp_response_helper, _impulse_response_helper,
    _bode_magnitude_helper, _bode_phase_helper
)
from spb.interactive import create_interactive_plot
from spb.plotgrid import plotgrid
from spb.series import HVLineSeries
from spb.utils import _instantiate_backend
from sympy import exp, latex, sympify, Expr
from sympy.physics.control.lti import SISOLinearTimeInvariant


__all__ = [
    'pole_zero_plot', 'plot_pole_zero',
    'step_response_plot', 'plot_step_response',
    'impulse_response_plot', 'plot_impulse_response',
    'ramp_response_plot', 'plot_ramp_response',
    'bode_magnitude_plot', 'plot_bode_magnitude',
    'bode_phase_plot', 'plot_bode_phase',
    'bode_plot', 'plot_bode',
    'nyquist_plot', 'plot_nyquist',
    'nichols_plot', 'plot_nichols'
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
    elif (
        all(isinstance(t, (list, tuple)) for t in systems) and
        all(len(t) >= 2 for t in systems) and
        all(all(
            isinstance(e, (int, float, Expr)) for e in t[:2]
        ) for t in systems)
    ):
        # list of tuples of the form: (num, den, gen [opt], label [opt])
        new_systems = []
        for s in systems:
            ns = _preprocess_system(
                s if not isinstance(s[-1], str) else s[:-1]
            )
            new_systems.append(
                ns if not isinstance(s[-1], str) else (ns, s[-1]))
        systems = new_systems.copy()
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


def plot_pole_zero(
    *systems, pole_markersize=10, zero_markersize=7, show_axes=False,
    **kwargs
):
    """
    Returns the [Pole-Zero]_ plot (also known as PZ Plot or PZ Map) of
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
    z_rendering_kw : dict
        A dictionary of keyword arguments to further customize the appearance
        of zeros.
    p_rendering_kw : dict
        A dictionary of keyword arguments to further customize the appearance
        of poles.
    **kwargs : dict
        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import plot_pole_zero
        >>> tf1 = TransferFunction(
        ...     s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        >>> plot_pole_zero(tf1)
        Plot object containing:
        [0]: 2D list plot
        [1]: 2D list plot

    Interactive-widgets plot of multiple systems, one of which is parametric:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, d, s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_pole_zero
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

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot
    """
    systems = _unpack_systems(systems)
    ms = len(systems) > 1
    series = []
    for system, label in systems:
        series.extend(_pole_zero_helper(
            system, label, ms,
            pole_markersize, zero_markersize, **kwargs.copy()
        ))

    kwargs.setdefault("xlabel", "Real axis")
    kwargs.setdefault("ylabel", "Imaginary axis")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Poles and Zeros"))
    return _create_plot_helper(series, show_axes, **kwargs)


pole_zero_plot = plot_pole_zero


def plot_step_response(
    *systems, lower_limit=0, upper_limit=10,
    prec=8, show_axes=False, **kwargs
):
    """
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Step Response is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
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
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.step_response` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import plot_step_response
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> plot_step_response(tf1)   # doctest: +SKIP


    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit`` and ``upper_limit``.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy.abc import a, b, c, d, e, f, g, s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_step_response
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
           }, use_latex=False)

    See Also
    ========

    plot_impulse_response, plot_ramp_response

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

    systems = _unpack_systems(systems)
    series = [
        _step_response_helper(
            s, l, lower_limit, upper_limit, prec, **kwargs
        ) for s, l in systems
    ]

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Unit Response"))
    return _create_plot_helper(series, show_axes, **kwargs)


step_response_plot = plot_step_response


def plot_impulse_response(
    *systems, prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, **kwargs
):
    """
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Impulse Response is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
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
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.impulse_response` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import plot_impulse_response
        >>> tf1 = TransferFunction(
        ...     8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> plot_impulse_response(tf1)   # doctest: +SKIP

    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit`` and ``upper_limit``.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy.abc import a, b, c, d, e, f, g, h, s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_impulse_response
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
           }, use_latex=False)


    See Also
    ========

    plot_step_response, plot_ramp_response

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

    systems = _unpack_systems(systems)
    series = [
        _impulse_response_helper(
            s, l, lower_limit, upper_limit, prec, **kwargs
        ) for s, l in systems
    ]

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Impulse Response"))
    return _create_plot_helper(series, show_axes, **kwargs)


impulse_response_plot = plot_impulse_response


def plot_ramp_response(
    *systems, slope=1, prec=8,
    lower_limit=0, upper_limit=10, show_axes=False, **kwargs
):
    """
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line passing through origin
    ($f(x) = mx$). The slope of the ramp function can be varied by the user
    and the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Ramp Response is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
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
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.ramp_response` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import plot_ramp_response
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> plot_ramp_response(tf1, upper_limit=2)   # doctest: +SKIP

    Interactive-widgets plot of multiple systems, one of which is parametric.
    Note the use of parametric ``lower_limit``, ``upper_limit`` and ``slope``.

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy.abc import a, b, c, d, e, s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_ramp_response
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
           }, use_latex=False)

    See Also
    ========

    plot_step_response, plot_impulse_response

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    # allows parametric slope
    slope = sympify(slope)
    if slope.is_Number and slope < 0:
        raise ValueError(
            "Slope must be greater than or equal to zero."
        )
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError(
            "Lower limit of time must be greater than or equal to zero."
        )

    systems = _unpack_systems(systems)
    series = [
        _ramp_response_helper(
            s, l, lower_limit, upper_limit, prec, slope, **kwargs
        ) for s, l in systems
    ]

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Ramp Response"))
    return _create_plot_helper(series, show_axes, **kwargs)


ramp_response_plot = plot_ramp_response


def plot_bode_magnitude(
    *systems, initial_exp=-5, final_exp=5,
    freq_unit='rad/sec', show_axes=False, **kwargs
):
    """
    Returns the Bode magnitude plot of a continuous-time system.

    See ``plot_bode`` for all the parameters.
    """
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    systems = _unpack_systems(systems)
    series = [
        _bode_magnitude_helper(
            s, l, initial_exp, final_exp,
            freq_unit, **kwargs
        ) for s, l in systems
    ]

    kwargs.setdefault("xlabel", 'Frequency [%s]' % freq_unit)
    kwargs.setdefault("ylabel", 'Magnitude (dB)')
    kwargs.setdefault("title", _create_title_helper(
        systems, "Bode Plot (Magnitude)"))
    kwargs.setdefault("xscale", "log")
    return _create_plot_helper(series, show_axes, **kwargs)


bode_magnitude_plot = plot_bode_magnitude


def plot_bode_phase(
    *systems, initial_exp=-5, final_exp=5,
    freq_unit='rad/sec', phase_unit='rad', show_axes=False, **kwargs
):
    """
    Returns the Bode phase plot of a continuous-time system.

    See ``plot_bode`` for all the parameters.
    """
    freq_units = ('rad/sec', 'Hz')
    phase_units = ('rad', 'deg')
    if freq_unit not in freq_units:
        raise ValueError(
            'Only "rad/sec" and "Hz" are accepted frequency units.'
        )
    if phase_unit not in phase_units:
        raise ValueError('Only "rad" and "deg" are accepted phase units.')

    systems = _unpack_systems(systems)
    series = [
        _bode_phase_helper(
            s, l, initial_exp, final_exp,
            freq_unit, phase_unit, **kwargs
        ) for s, l in systems
    ]

    kwargs.setdefault("xlabel", 'Frequency [%s]' % freq_unit)
    kwargs.setdefault("ylabel", 'Phase [%s]' % phase_unit)
    kwargs.setdefault("title", _create_title_helper(
        systems, "Bode Plot (Phase)"))
    kwargs.setdefault("xscale", "log")
    return _create_plot_helper(series, show_axes, **kwargs)


bode_phase_plot = plot_bode_phase


def plot_bode(
    *systems, initial_exp=-5, final_exp=5,
    freq_unit='rad/sec', phase_unit='rad', show_axes=False, **kwargs
):
    """
    Returns the Bode phase and magnitude plots of a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Bode Plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
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
    unwrap : bool, optional
        Depending on the transfer function, there could be discontinuities in
        the phase plot. Set ``unwrap=True`` to get a continuous phase.
        Default to False.
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.bode_magnitude` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from spb import plot_bode, plot_bode_phase, plotgrid
        >>> tf1 = TransferFunction(
        ...     1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        >>> plot_bode(tf1, initial_exp=0.2, final_exp=0.7)   # doctest: +SKIP

    In this example it is necessary to unwrap the phase:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> tf = TransferFunction(1, s**3 + 2*s**2 + s, s)
        >>> p1 = plot_bode_phase(
        ...     tf, unwrap=False, show=False, title="unwrap=False")
        >>> p2 = plot_bode_phase(
        ...     tf, unwrap=True, show=False, title="unwrap=True")
        >>> plotgrid(p1, p2)

    Interactive-widget plot:

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy.abc import a, b, c, d, e, f, s
       from sympy.physics.control.lti import TransferFunction
       from spb import *

       tf1 = TransferFunction(a*s**2 + b*s + c, d*s**4 + e*s**3 + f*s**2, s)
       plot_bode(
           tf1, initial_exp=-2, final_exp=2,
           params={
               a: (0.5, -10, 10),
               b: (0.1, -1, 1),
               c: (8, -10, 10),
               d: (10, -10, 10),
               e: (0.1, -1, 1),
               f: (1, -10, 10),
           },
           imodule="panel", ncols=3, use_latex=False
       )

    See Also
    ========

    plot_bode_magnitude, plot_bode_phase, plot_nyquist, plot_nichols

    """
    show = kwargs.pop("show", True)
    kwargs["show"] = False
    p1 = plot_bode_magnitude(
        *systems, show_axes=show_axes,
        initial_exp=initial_exp, final_exp=final_exp,
        freq_unit=freq_unit, **kwargs.copy()
    )
    p2 = plot_bode_phase(
        *systems, show_axes=show_axes,
        initial_exp=initial_exp, final_exp=final_exp,
        freq_unit=freq_unit, phase_unit=phase_unit,
        title="", **kwargs.copy()
    )

    systems = _unpack_systems(systems)
    title = "Bode Plot"
    if len(systems) == 1:
        title = f'Bode Plot of ${latex(systems[0][0])}$'
    p1.title = title
    p = plotgrid(p1, p2, **kwargs)

    if show:
        if kwargs.get("params", None):
            return p.show()
        p.show()
    return p


bode_plot = plot_bode


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
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
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
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.nyquist` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hall_circles

    See Also
    ========

    plot_bode, plot_nichols

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
       >>> from spb import plot_nyquist
       >>> tf1 = TransferFunction(
       ...     4 * s**2 + 5 * s + 1, 3 * s**2 + 2 * s + 5, s)
       >>> plot_nyquist(tf1)                                # doctest: +SKIP

    Visualizing M-circles:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

    >>> plot_nyquist(tf1, m_circles=True)                   # doctest: +SKIP

    Plotting multiple transfer functions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> tf2 = TransferFunction(1, s + Rational(1, 3), s)
       >>> plot_nyquist(tf1, tf2)                           # doctest: +SKIP

    Interactive-widgets plot of a systems:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, d, e, f, s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_nyquist
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


def plot_nichols(*systems, **kwargs):
    """Nichols plot for a system over a (optional) frequency range.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Bode Plot is to be computed.
        It can be:

        * a single LTI SISO system.
        * a symbolic expression, which will be converted to an object of
          type :class:`~sympy.physics.control.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :class:`~sympy.physics.control.TransferFunction`.
        * a sequence of LTI SISO systems.
        * a sequence of 2-tuples ``(LTI SISO system, label)``.
        * a dict mapping LTI SISO systems to labels.
    ngrid : bool, optional
        Turn on/off the [Nichols]_ grid lines.
    omega_limits : array_like of two values, optional
        Limits to the range of frequencies.
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.nichols` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

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
       >>> from spb import plot_nichols
       >>> tf = TransferFunction(50*s**2 - 20*s + 15, -10*s**2 + 40*s + 30, s)
       >>> plot_nichols(tf)                                 # doctest: +SKIP

    Turning off the Nichols grid lines:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_nichols(tf, ngrid=False)                    # doctest: +SKIP

    Plotting multiple transfer functions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> tf1 = TransferFunction(1, s**2 + 2*s + 1, s)
       >>> tf2 = TransferFunction(1, s**2 - 2*s + 1, s)
       >>> plot_nichols(tf1, tf2, xlim=(-360, 360))         # doctest: +SKIP

    Interactive-widgets plot of a systems. For these kind of plots, it is
    recommended to set both ``omega_limits`` and ``xlim``:

    .. panel-screenshot::
       :small-size: 800, 650

       from sympy.abc import a, b, c, s
       from spb import plot_nichols
       from sympy.physics.control.lti import TransferFunction
       tf = TransferFunction(a*s**2 + b*s + c, s**3 + 10*s**2 + 5 * s + 1, s)
       plot_nichols(
           tf, omega_limits=[1e-03, 1e03], n=1e04,
           params={
               a: (-25, -100, 100),
               b: (60, -300, 300),
               c: (-100, -1000, 1000),
           },
           xlim=(-360, 360)
       )

    See Also
    ========

    plot_bode, plot_nyquist

    """
    systems = _unpack_systems(systems)
    series = [_nichols_helper(s, l, **kwargs.copy()) for s, l in systems]
    kwargs.setdefault("ngrid", True)
    kwargs.setdefault("xlabel", "Open-Loop Phase [deg]")
    kwargs.setdefault("ylabel", "Open-Loop Magnitude [dB]")
    kwargs.setdefault("title", _create_title_helper(
        systems, "Nichols Plot"))
    kwargs.setdefault("grid", not kwargs.get("ngrid", False))
    return _create_plot_helper(series, False, **kwargs)


nichols_plot = plot_nichols
