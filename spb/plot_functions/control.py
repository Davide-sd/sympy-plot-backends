from spb.defaults import TWO_D_B, cfg
from spb.graphics.control import (
    _preprocess_system, pole_zero,
    nichols, nyquist, step_response,
    ramp_response, impulse_response,
    bode_magnitude, bode_phase,
    control_axis, root_locus, ngrid as ngrid_function,
    _get_grid_series
)
from spb.interactive import create_interactive_plot
from spb.plotgrid import plotgrid
from spb.series import HVLineSeries
from spb.utils import _instantiate_backend
from sympy import exp, latex, sympify, Expr


__all__ = [
    'pole_zero_plot', 'plot_pole_zero',
    'step_response_plot', 'plot_step_response',
    'impulse_response_plot', 'plot_impulse_response',
    'ramp_response_plot', 'plot_ramp_response',
    'bode_magnitude_plot', 'plot_bode_magnitude',
    'bode_phase_plot', 'plot_bode_phase',
    'bode_plot', 'plot_bode',
    'nyquist_plot', 'plot_nyquist',
    'nichols_plot', 'plot_nichols', 'plot_root_locus'
]


def _create_plot_helper(series, show_axes, **kwargs):
    if show_axes:
        series = control_axis() + series

    Backend = kwargs.pop("backend", TWO_D_B)
    if kwargs.get("params", None):
        return create_interactive_plot(*series, backend=Backend, **kwargs)

    return _instantiate_backend(Backend, *series, **kwargs)


def _unpack_systems(systems, **kwargs):
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
        provided_label = kwargs.get("label", None)
        if isinstance(provided_label, str):
            labels = [provided_label] * len(systems)
        else:
            labels = [f"System {i+1}" for i in range(len(systems))]
        systems = [(system, label) for system, label in zip(systems, labels)]
    return systems


def plot_pole_zero(
    *systems, pole_markersize=10, zero_markersize=7, show_axes=False,
    sgrid=False, zgrid=False, control=True, input=None, output=None, **kwargs
):
    """
    Returns the [Pole-Zero]_ plot (also known as PZ Plot or PZ Map) of
    a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    systems : one or more LTI system type
        The system for which the pole-zero plot is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
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
    sgrid : bool, optional
        Generates a grid of constant damping ratios and natural frequencies
        on the s-plane. Default to False.
    zgrid : bool, optional
        Generates a grid of constant damping ratios and natural frequencies
        on the z-plane. Default to False.
    control : bool, optional
        If True, computes the poles/zeros with the ``control`` module,
        which uses numerical integration. If False, computes them
        with ``sympy``. Default to True.
    input : int, optional
        Only compute the poles/zeros for the listed input. If not specified,
        the poles/zeros for each independent input are computed (as
        separate traces).
    output : int, optional
        Only compute the poles/zeros for the listed output.
        If not specified, all outputs are reported.
    **kwargs : dict
        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    Plotting poles and zeros on the s-plane:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s
        from sympy import I
        from sympy.physics.control.lti import TransferFunction
        from spb import plot_pole_zero
        tf1 = TransferFunction(
            s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        plot_pole_zero(tf1, sgrid=True)

    Plotting poles and zeros on the z-plane:

    .. plot::
        :context: close-figs
        :include-source: True

        plot_pole_zero(tf1, zgrid=True)

    If a transfer function has complex coefficients, make sure to request
    the evaluation using ``sympy`` instead of the ``control`` module:

    .. plot::
        :context: close-figs
        :include-source: True

        tf = TransferFunction(s + 2, s**2 + (2+I)*s + 10, s)
        plot_pole_zero(tf, control=False, grid=False, show_axes=True)

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
    series = []
    for system, label in systems:
        series.extend(pole_zero(
            system, label=label,
            pole_markersize=pole_markersize,
            zero_markersize=zero_markersize,
            control=control, **kwargs.copy()
        ))

    grid = _get_grid_series(sgrid, zgrid)
    if sgrid or zgrid:
        kwargs.setdefault("grid", False)
    kwargs.setdefault("xlabel", "Real axis")
    kwargs.setdefault("ylabel", "Imaginary axis")
    kwargs.setdefault("title", "Poles and Zeros")
    return _create_plot_helper(grid + series, show_axes, **kwargs)


pole_zero_plot = plot_pole_zero


def plot_step_response(
    *systems, lower_limit=0, upper_limit=10,
    prec=8, show_axes=False, control=True,
    input=None, output=None, **kwargs
):
    """
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    systems : LTI system type
        The LTI system for which the step response is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0. If a different value
        is to be used, also set ``control=False`` (see examples in order to
        understand why).
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    control : bool
        If True, computes the step response with the ``control``
        module, which uses numerical integration. If False, computes the
        step response with ``sympy``, which uses the inverse Laplace transform.
        Default to True.
    control_kw : dict
        A dictionary of keyword arguments passed to
        :py:func:`control.step_response`.
    input : int, optional
        Only compute the step response for the listed input. If not
        specified, the step responses for each independent input are
        computed (as separate traces).
    output : int, optional
        Only compute the step response for the listed output. If not
        specified, all outputs are reported.
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.step_response` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    Plotting a SISO system:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s
        from sympy.physics.control.lti import TransferFunction
        from spb import plot_step_response
        tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        plot_step_response(tf1)

    Plotting a MIMO system:

    .. plot::
        :context: close-figs
        :include-source: True

        from sympy.physics.control.lti import TransferFunctionMatrix
        tf1 = TransferFunction(1, s + 2, s)
        tf2 = TransferFunction(s + 1, s**2 + s + 1, s)
        tf3 = TransferFunction(s + 1, s**2 + s + 1.5, s)
        tfm = TransferFunctionMatrix(
            [[tf1, -tf1], [tf2, -tf2], [tf3, -tf3]])
        plot_step_response(tfm)

    Plotting a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        G = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)
        plot_step_response(G, upper_limit=15)

    Interactive-widgets plot of multiple systems, one of which is parametric.
    A few observations:

    1. Both systems are evaluated with the ``control`` module.
    2. Note the use of parametric ``lower_limit`` and ``upper_limit``.
    3. By moving the "lower limit" slider, both systems always start from
       zero amplitude. That's because the numerical integration's initial
       condition is 0. Hence, if ``lower_limit`` is to be used, please
       set ``control=False``.

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
    systems = _unpack_systems(systems, **kwargs)
    series = []
    for s, l in systems:
        kw = kwargs.copy()
        kw["label"] = l
        series.extend(
            step_response(
                s, lower_limit, upper_limit, prec, control=control,
                input=input, output=output, **kw
            )
        )

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", "Step Response")
    return _create_plot_helper(series, show_axes, **kwargs)


step_response_plot = plot_step_response


def plot_impulse_response(
    *systems, prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, control=True,
    input=None, output=None, **kwargs
):
    """
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system.

    Parameters
    ==========

    systems : LTI system type
        The LTI system for which the impulse response is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0. If a different value
        is to be used, also set ``control=False`` (see examples in order to
        understand why).
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    control : bool
        If True, computes the step response with the ``control``
        module, which uses numerical integration. If False, computes the
        step response with ``sympy``, which uses the inverse Laplace transform.
        Default to True.
    control_kw : dict
        A dictionary of keyword arguments passed to
        :py:func:`control.impulse_response`.
    input : int, optional
        Only compute the impulse response for the listed input.  If not
        specified, the impulse responses for each independent input are
        computed (as separate traces).
    output : int, optional
        Only compute the impulse response for the listed output. If not
        specified, all outputs are reported.
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.impulse_response` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    Plotting a SISO system:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s
        from sympy.physics.control.lti import TransferFunction
        from spb import plot_impulse_response
        tf1 = TransferFunction(
            8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        plot_impulse_response(tf1)

    Plotting a MIMO system:

    .. plot::
        :context: close-figs
        :include-source: True

        from sympy.physics.control.lti import TransferFunctionMatrix
        tf1 = TransferFunction(1, s + 2, s)
        tf2 = TransferFunction(s + 1, s**2 + s + 1, s)
        tf3 = TransferFunction(s + 1, s**2 + s + 1.5, s)
        tfm = TransferFunctionMatrix(
            [[tf1, -tf1], [tf2, -tf2], [tf3, -tf3]])
        plot_impulse_response(tfm)

    Plotting a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        G = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)
        plot_impulse_response(G, upper_limit=15)

    Interactive-widgets plot of multiple systems, one of which is parametric.
    A few observations:

    1. Both systems are evaluated with the ``control`` module.
    2. Note the use of parametric ``lower_limit`` and ``upper_limit``.
    3. By moving the "lower limit" slider, both systems always start from
       zero amplitude. That's because the numerical integration's initial
       condition is 0. Hence, if ``lower_limit`` is to be used, please
       set ``control=False``.

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
    systems = _unpack_systems(systems, **kwargs)
    series = []
    for s, l in systems:
        kw = kwargs.copy()
        kw["label"] = l
        series.extend(
            impulse_response(
                s, prec, lower_limit, upper_limit, control=control,
                input=input, output=output, **kw
            )
        )

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", "Impulse Response")
    return _create_plot_helper(series, show_axes, **kwargs)


impulse_response_plot = plot_impulse_response


def plot_ramp_response(
    *systems, slope=1, prec=8,
    lower_limit=0, upper_limit=10, show_axes=False, control=True,
    input=None, output=None, **kwargs
):
    """
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line passing through origin
    ($f(x) = mx$). The slope of the ramp function can be varied by the user
    and the default value is 1.

    Parameters
    ==========

    systems : LTI system type
        The LTI system for which the ramp response is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0. If a different value
        is to be used, also set ``control=False`` (see examples in order to
        understand why).
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    control : bool
        If True, computes the step response with the ``control``
        module, which uses numerical integration. If False, computes the
        step response with ``sympy``, which uses the inverse Laplace transform.
        Default to True.
    control_kw : dict
        A dictionary of keyword arguments passed to
        :py:func:`control.forced_response`.
    input : int, optional
        Only compute the ramp response for the listed input.  If not
        specified, the ramp responses for each independent input are
        computed (as separate traces).
    output : int, optional
        Only compute the ramp response for the listed output. If not
        specified, all outputs are reported.
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.ramp_response` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    Plotting a SISO system:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s
        from sympy.physics.control.lti import TransferFunction
        from spb import plot_ramp_response
        tf1 = TransferFunction(1, (s+1), s)
        plot_ramp_response(tf1)

    Plotting a MIMO system:

    .. plot::
        :context: close-figs
        :include-source: True

        from sympy.physics.control.lti import TransferFunctionMatrix
        tf1 = TransferFunction(1, s + 2, s)
        tf2 = TransferFunction(s + 1, s**2 + s + 1, s)
        tf3 = TransferFunction(s + 1, s**2 + s + 1.5, s)
        tfm = TransferFunctionMatrix(
            [[tf1, -tf1], [tf2, -tf2], [tf3, -tf3]])
        plot_ramp_response(tfm)

    Plotting a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        G = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)
        plot_ramp_response(G, upper_limit=15)

    Interactive-widgets plot of multiple systems, one of which is parametric.
    A few observations:

    1. Both systems are evaluated with the ``control`` module.
    2. Note the use of parametric ``lower_limit`` and ``upper_limit``.
    3. By moving the "lower limit" slider, both systems always start from
       zero amplitude. That's because the numerical integration's initial
       condition is 0. Hence, if ``lower_limit`` is to be used, please
       set ``control=False``.

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy import symbols
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_ramp_response
       a, b, c, xi, wn, s, t = symbols("a, b, c, xi, omega_n, s, t")
       tf1 = TransferFunction(25, s**2 + 10*s + 25, s)
       tf2 = TransferFunction(wn**2, s**2 + 2*xi*wn*s + wn**2, s)
       params = {
           xi: (6, 0, 10),
           wn: (25, 0, 50),
           # NOTE: remove `None` if using ipywidgets
           a: (1, 0, 10, 50, None, "slope"),
           b: (0, 0, 5, 50, None, "lower limit"),
           c: (5, 2, 10, 50, None, "upper limit"),
       }
       plot_ramp_response(
           (tf1, "A"), (tf2, "B"),
           slope=a, lower_limit=b, upper_limit=c,
           params=params, use_latex=False)

    See Also
    ========

    plot_step_response, plot_impulse_response

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    systems = _unpack_systems(systems, **kwargs)
    series = []
    for s, l in systems:
        kw = kwargs.copy()
        kw["label"] = l
        series.extend(
                ramp_response(
                s, prec, slope, lower_limit, upper_limit,
                control=control, input=input, output=output, **kw
            )
        )

    kwargs.setdefault("xlabel", "Time [s]")
    kwargs.setdefault("ylabel", "Amplitude")
    kwargs.setdefault("title", "Ramp Response")
    return _create_plot_helper(series, show_axes, **kwargs)


ramp_response_plot = plot_ramp_response


def plot_bode_magnitude(
    *systems, initial_exp=None, final_exp=None,
    freq_unit=None, show_axes=False, **kwargs
):
    """
    Returns the Bode magnitude plot of a continuous-time system.

    See ``plot_bode`` for all the parameters.
    """
    freq_units = ('rad/sec', 'Hz')
    freq_unit = cfg["bode"]["freq_unit"] if freq_unit is None else freq_unit
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    systems = _unpack_systems(systems)
    series = []
    for s, l in systems:
        series.extend(
            bode_magnitude(
                s, label=l, initial_exp=initial_exp, final_exp=final_exp,
                freq_unit=freq_unit, **kwargs
            )
        )

    kwargs.setdefault("xlabel", 'Frequency [%s]' % freq_unit)
    kwargs.setdefault("ylabel", 'Magnitude (dB)')
    kwargs.setdefault("title", "Bode Plot (Magnitude)")
    kwargs.setdefault("xscale", "log")
    return _create_plot_helper(series, show_axes, **kwargs)


bode_magnitude_plot = plot_bode_magnitude


def plot_bode_phase(
    *systems, initial_exp=None, final_exp=None,
    freq_unit=None, phase_unit=None, show_axes=False,
    unwrap=True, **kwargs
):
    """
    Returns the Bode phase plot of a continuous-time system.

    See ``plot_bode`` for all the parameters.
    """
    freq_units = ('rad/sec', 'Hz')
    phase_units = ('rad', 'deg')
    freq_unit = cfg["bode"]["freq_unit"] if freq_unit is None else freq_unit
    phase_unit = cfg["bode"]["phase_unit"] if phase_unit is None else phase_unit
    if freq_unit not in freq_units:
        raise ValueError(
            'Only "rad/sec" and "Hz" are accepted frequency units.'
        )
    if phase_unit not in phase_units:
        raise ValueError('Only "rad" and "deg" are accepted phase units.')

    systems = _unpack_systems(systems)
    series = []
    for s, l in systems:
        series.extend(
                bode_phase(
                s, label=l, initial_exp=initial_exp, final_exp=final_exp,
                freq_unit=freq_unit, phase_unit=phase_unit,
                unwrap=unwrap, **kwargs
            )
        )

    kwargs.setdefault("xlabel", 'Frequency [%s]' % freq_unit)
    kwargs.setdefault("ylabel", 'Phase [%s]' % phase_unit)
    kwargs.setdefault("title", "Bode Plot (Phase)")
    kwargs.setdefault("xscale", "log")
    return _create_plot_helper(series, show_axes, **kwargs)


bode_phase_plot = plot_bode_phase


def plot_bode(
    *systems, initial_exp=None, final_exp=None,
    freq_unit=None, phase_unit=None, show_axes=False,
    unwrap=True, **kwargs
):
    """
    Returns the Bode phase and magnitude plots of a continuous-time system.

    Parameters
    ==========

    systems : LTI system type
        The LTI system for which the ramp response is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Default to None, which
        will autocompute the appropriate value.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Default to None, which
        will autocompute the appropriate value.
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
        Depending on the transfer function, the computed phase could contain
        discontinuities of 2*pi. ``unwrap=True`` post-process the numerical
        data in order to get a continuous phase.
    input : int, optional
        Only compute the poles/zeros for the listed input. If not specified,
        the poles/zeros for each independent input are computed (as
        separate traces).
    output : int, optional
        Only compute the poles/zeros for the listed output.
        If not specified, all outputs are reported.
        Default to True.
    **kwargs : dict
        Refer to :func:`~spb.graphics.control.bode_magnitude` for a full list
        of keyword arguments to customize the appearances of lines.

        Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
        keyword arguments to customize the appearances of the figure (title,
        axis labels, ...).

    Examples
    ========

    .. plot::
       :context: reset
       :include-source: True

       from sympy.abc import s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_bode, plot_bode_phase, plotgrid
       tf1 = TransferFunction(
           1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
       plot_bode(tf1)

    This example shows how the phase is actually computed (with
    ``unwrap=False``) and how it is post-processed (with ``unwrap=True``).

    .. plot::
       :context: close-figs
       :include-source: True

       tf = TransferFunction(1, s**3 + 2*s**2 + s, s)
       p1 = plot_bode_phase(
           tf, unwrap=False, show=False, title="unwrap=False")
       p2 = plot_bode_phase(
           tf, unwrap=True, show=False, title="unwrap=True")
       plotgrid(p1, p2)

    ``plot_bode`` also works with time delays. However, for the post-processing
    of the phase to work as expected, the frequency range must be sufficiently
    small, and the number of discretization points must be sufficiently high.

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import symbols, exp
       s = symbols("s")
       G1 = 1 / (s * (s + 1) * (s + 10))
       G2 = G1 * exp(-5*s)
       plot_bode(G1, G2, phase_unit="deg", n=1e04)

    Bode plot of a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        tf = ct.tf([1], [1, 2, 3], dt=0.05)
        plot_bode(tf)

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
    title = kwargs.get("title", None)
    kwargs["title"] = ""
    p1 = plot_bode_magnitude(
        *systems, show_axes=show_axes,
        initial_exp=initial_exp, final_exp=final_exp,
        freq_unit=freq_unit, **kwargs.copy()
    )
    p2 = plot_bode_phase(
        *systems, show_axes=show_axes,
        initial_exp=initial_exp, final_exp=final_exp,
        freq_unit=freq_unit, phase_unit=phase_unit,
        **kwargs.copy()
    )

    systems = _unpack_systems(systems)
    if title is None:
        title = 'Bode Plot'
    p1.title = title
    p = plotgrid(p1, p2, **kwargs)

    if show:
        if kwargs.get("params", None):
            return p.show()
        p.show()
    return p


bode_plot = plot_bode


def plot_nyquist(*systems, **kwargs):
    """Plots a Nyquist plot for the system over a (optional) frequency range.
    The curve is computed by evaluating the Nyquist segment along the positive
    imaginary axis, with a mirror image generated to reflect the negative
    imaginary axis. Poles on or near the imaginary axis are avoided using a
    small indentation. The portion of the Nyquist contour at infinity is not
    explicitly computed (since it maps to a constant value for any system with
    a proper transfer function).

    Parameters
    ==========

    systems : LTI system type
        The LTI system for which the ramp response is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
    label : str, optional
        The label to be shown on the legend.
    arrows : int or 1D/2D array of floats, optional
        Specify the number of arrows to plot on the Nyquist curve.  If an
        integer is passed, that number of equally spaced arrows will be
        plotted on each of the primary segment and the mirror image.  If a 1D
        array is passed, it should consist of a sorted list of floats between
        0 and 1, indicating the location along the curve to plot an arrow.
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
    m_circles : bool or float or iterable, optional
        Turn on/off M-circles, which are circles of constant closed loop
        magnitude. If float or iterable (of floats), represents specific
        magnitudes in dB.
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
    control_kw : dict, optional
        A dictionary of keyword arguments passed to
        :py:func:`control.nyquist_plot`
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
       :include-source: True

       from sympy import Rational
       from sympy.abc import s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_nyquist
       tf1 = TransferFunction(
           4 * s**2 + 5 * s + 1, 3 * s**2 + 2 * s + 5, s)
       plot_nyquist(tf1)

    Plotting multiple transfer functions and visualizing M-circles:

    .. plot::
       :context: close-figs
       :include-source: True

       tf2 = TransferFunction(1, s + Rational(1, 3), s)
       plot_nyquist(tf1, tf2,
           m_circles=[-20, -10, -6, -4, -2, 0])


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
    series = []
    for s, l in systems:
        series.extend(nyquist(s, label=l, **kwargs.copy()))
    kwargs.setdefault("xlabel", "Real axis")
    kwargs.setdefault("ylabel", "Imaginary axis")
    kwargs.setdefault("title", "Nyquist Plot")
    kwargs.setdefault("grid", not kwargs.get("m_circles", False))
    return _create_plot_helper(series, False, **kwargs)


nyquist_plot = plot_nyquist


def plot_nichols(*systems, **kwargs):
    """Nichols plot for a system over a (optional) frequency range.

    Parameters
    ==========

    systems : LTI system type
        The LTI system for which the ramp response is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
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
       :include-source: True

       from sympy.abc import s
       from sympy.physics.control.lti import TransferFunction
       from spb import plot_nichols
       tf = TransferFunction(50*s**2 - 20*s + 15, -10*s**2 + 40*s + 30, s)
       plot_nichols(tf)

    Turning off the Nichols grid lines:

    .. plot::
       :context: close-figs
       :include-source: True

       plot_nichols(tf, ngrid=False)

    Plotting multiple transfer functions:

    .. plot::
       :context: close-figs
       :include-source: True

       tf1 = TransferFunction(1, s**2 + 2*s + 1, s)
       tf2 = TransferFunction(1, s**2 - 2*s + 1, s)
       plot_nichols(tf1, tf2, xlim=(-360, 360))

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
    series, grid = [], []
    show_ngrid = kwargs.get("ngrid", True)
    kw = kwargs.copy()
    kw["ngrid"] = False
    for s, l in systems:
        series.extend(nichols(s, l, **kw.copy()))
    if show_ngrid:
        grid.extend(ngrid_function())
    kwargs.setdefault("xlabel", "Open-Loop Phase [deg]")
    kwargs.setdefault("ylabel", "Open-Loop Magnitude [dB]")
    kwargs.setdefault("title", "Nichols Plot")
    kwargs.setdefault("grid", not show_ngrid)
    return _create_plot_helper(grid + series, False, **kwargs)


nichols_plot = plot_nichols


def plot_root_locus(*systems, sgrid=True, zgrid=False, **kwargs):
    """Root Locus plot for one or multiple systems.

    Notes
    =====

    This function uses the ``python-control`` module to generate the numerical
    data.

    Parameters
    ==========

    systems : LTI system type
        The LTI system for which the ramp response is to be computed.
        It can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a sequence of LTI systems.
        * a sequence of 2-tuples ``(LTI system, label)``.
        * a dict mapping LTI systems to labels.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    control_kw : dict
        A dictionary of keyword arguments to be passed to
        :py:func:`control.root_locus`.
    sgrid : bool, optional
        Generates a grid of constant damping ratios and natural frequencies
        on the s-plane. Default to True.
    zgrid : bool, optional
        Generates a grid of constant damping ratios and natural frequencies
        on the z-plane. Default to False. If ``zgrid=True``, then it will
        automatically sets ``sgrid=False``.
    input : int, optional
        Only compute the poles/zeros for the listed input. If not specified,
        the poles/zeros for each independent input are computed (as
        separate traces).
    output : int, optional
        Only compute the poles/zeros for the listed output.
        If not specified, all outputs are reported.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Examples
    ========

    Plotting a single transfer function on the s-plane:

    .. plot::
       :context: reset
       :include-source: True

       from sympy.abc import s, z
       from spb import plot_root_locus
       G1 = (s**2 - 4) / (s**3 + 2*s - 3)
       plot_root_locus(G1)

    Plotting a discrete transfer function on the z-plane:

    .. plot::
       :context: close-figs
       :include-source: True

       G2 = (0.038*z + 0.031)/(9.11*z**2 - 13.77*z + 5.0)
       plot_root_locus(G2, zgrid=True, aspect="equal")

    Plotting multiple transfer functions:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy.abc import s
       from spb import plot_root_locus
       G3 = (s**2 + 1) / (s**3 + 2*s**2 + 3*s + 4)
       plot_root_locus(G1, G3)

    Interactive-widgets root locus plot:

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy import symbols
       from spb import *
       a, s = symbols("a, s")
       G = (s**2 + a) / (s**3 + 2*s**2 + 3*s + 4)
       params={a: (-0.5, -4, 4)}
       plot_root_locus(G, params=params, xlim=(-4, 1))

    """
    systems = _unpack_systems(systems)
    kwargs.setdefault("grid", False)
    kwargs.setdefault("xlabel", "Real")
    kwargs.setdefault("ylabel", "Imaginary")
    rls = [root_locus(s, l, sgrid=False, zgrid=False, **kwargs)[0]
        for s, l in systems]
    if sgrid and zgrid:
        # user has explicetly types zgrid=True. Disable sgrid.
        sgrid = False
    grid = _get_grid_series(sgrid, zgrid)
    if sgrid or zgrid:
        kwargs.setdefault("grid", False)
    return _create_plot_helper(grid + rls, False, **kwargs)
