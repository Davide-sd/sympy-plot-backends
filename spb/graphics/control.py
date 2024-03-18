from spb.defaults import TWO_D_B
from spb.series import (
    List2DSeries, LineOver1DRangeSeries, HVLineSeries, NyquistLineSeries,
    NicholsLineSeries, RootLocusSeries, SGridLineSeries, ZGridLineSeries,
    SystemResponseSeries, PoleZeroSeries, NGridLineSeries, MCirclesSeries
)
from spb.utils import (
    prange, is_number, tf_to_sympy, tf_to_control, _get_initial_params,
    is_discrete_time, tf_find_time_delay
)
import numpy as np
from sympy import (
    roots, exp, Poly, degree, re, im, apart, Dummy, symbols,
    I, log, Abs, arg, sympify, S, Min, Max, Piecewise, sqrt, cos, acos, sin,
    floor, ceiling, frac, pi, fraction, Expr, Tuple, inverse_laplace_transform,
    Integer, Float
)
from sympy.external import import_module
from mergedeep import merge
import warnings

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
    'nichols',
    'root_locus',
    'sgrid',
    'zgrid',
    'ngrid',
    'mcircles'
]


def _check_if_control_is_installed(use_control=None, force_stop=False):
    ct = import_module("control")
    if use_control is not None:
        if ct is None:
            warnings.warn(
                "``control=True`` was provided, but the ``control`` module "
                "is not installed. The evaluation will be performed by SymPy."
            )
            return False
    if force_stop and (ct is None):
        raise RuntimeError(
            "The ``control`` module is not installed. Can't proceed with "
            "the evaluation."
        )
    return use_control


def _preprocess_system(system, **kwargs):
    """Allow users to provide a transfer function with the following form:

    1. instance of ``sympy.physics.control.TransferFunction``.
    2. symbolic expression in fractional form.
    3. tuple of 2/3 elements:

       * ``(num, den)``: where ``num``, ``den`` can be symbolic expressions
         or list of coefficients of a polynomial.
       * ``(num, den, generator)``: where ``num``, ``den`` are symbolic
         expressions (possibly multivariate) and ``gen`` represents the
         s-variable or z-variable.

    4. instance of ``control.TransferFunction``.
    5. instance of ``scipy.signal.TransferFunction``.

    Returns
    =======
    system : TransferFunction
    """
    ct = import_module("control")
    sp = import_module("scipy")
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})

    if isinstance(system, (
        sm.control.lti.Series,
        sm.control.lti.Parallel
    )):
        return system.doit()

    allowed_types = [
        sm.control.lti.SISOLinearTimeInvariant,
        sm.control.lti.TransferFunctionMatrix,
    ]
    if ct is not None:
        allowed_types.append(ct.TransferFunction)
    if sp is not None:
        allowed_types.append(sp.signal.TransferFunction)

    if isinstance(system, tuple(allowed_types)):
        return system

    if isinstance(system, (list, tuple)):
        if len(system) == 2:
            if all(isinstance(e, Expr) for e in system):
                num, den = system
                fs = Tuple(num, den).free_symbols.pop()
                return sm.control.lti.TransferFunction(num, den, fs)
            else:
                num, den = system
                num = [float(t) for t in num]
                den = [float(t) for t in den]
                return ct.tf(num, den)
        elif len(system) == 3:
            num, den, fs = system
            return sm.control.lti.TransferFunction(num, den, fs)
        else:
            raise ValueError(
                "If a tuple/list is provided, it must have "
                "two or three elements: (num, den, free_symbol [opt]). "
                f"Received len(system) = {len(system)}"
            )

    if isinstance(system, Expr):
        params = kwargs.get("params", dict())
        fs = system.free_symbols.difference(params.keys())
        if len(fs) > 1:
            raise ValueError(f"Too many free symbols: {fs}")
        elif len(fs) == 0:
            raise ValueError(
                "An expression with one free symbol is required.")
        return sm.control.lti.TransferFunction.from_rational_expression(system, fs.pop())

    raise TypeError(f"type(system) = {type(system)} not recognized.")


def _is_siso(system):
    ct = import_module("control")
    sp = import_module("scipy")
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    if isinstance(system, sm.control.lti.SISOLinearTimeInvariant):
        return True
    if isinstance(system, sp.signal.TransferFunction):
        return True
    if (
        isinstance(system, ct.TransferFunction) and
        (system.ninputs == 1) and (system.noutputs == 1)
    ):
        return True
    return False


def _check_system(system, bypass_delay_check=False):
    """Function to check whether the dynamical system passed for plots is
    compatible or not."""

    if not _is_siso(system):
        raise NotImplementedError(
            "Only SISO LTI systems are currently supported.")

    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})

    if isinstance(system, sm.control.lti.TransferFunction):
        sys = system.to_expr()
        if not bypass_delay_check and sys.has(exp):
            # Should test that exp is not part of a constant, in which case
            # no exception is required, compare exp(s) with s*exp(1)
            raise NotImplementedError("Time delay terms are not supported.")


def _unpack_mimo_systems(system, label, input, output):
    """Unpack MIMO `system` into `[(sys1, label1), (sys2, label2), ...]`.
    """
    ct = import_module("control")
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    systems = []
    pre = " - " if len(label) > 0 else ""

    def _check_condition(i, o):
        if (input is None) and (output is None):
            return True
        if (input is None) and (output == o):
            return True
        if (output is None) and (input == i):
            return True
        if (input == i) and (output == o):
            return True
        return False

    if isinstance(system, sm.control.lti.TransferFunctionMatrix):
        for i in range(system.num_inputs):
            for o in range(system.num_outputs):
                if _check_condition(i, o):
                    lbl = label + pre + f"inp{i} -> out{o}"
                    systems.append([system[o, i], lbl])
        return systems

    elif (ct is not None) and isinstance(system, ct.TransferFunction):
        if (system.ninputs == 1) and (system.noutputs == 1):
            return [[system, label]]
        systems = []
        for i in range(system.ninputs):
            for o in range(system.noutputs):
                if _check_condition(i, o):
                    n = system.num[o][i]
                    d = system.den[o][i]
                    lbl = label + pre + f"inp{i} -> out{o}"
                    systems.append([ct.tf(n, d, dt=system.dt), lbl])
        return systems

    return [[system, label]]


def _get_zeros_poles_from_symbolic_tf(system):
    s = system.var
    num_poly = Poly(system.num, s)
    den_poly = Poly(system.den, s)
    zeros = roots(num_poly, s, multiple=True)
    if len(zeros) != degree(num_poly, s):
        raise ValueError(
            "Coult not compute all the roots of the numerator of the "
            "transfer function.")
    poles = roots(den_poly, s, multiple=True)
    if len(poles) != degree(den_poly, s):
        raise ValueError(
            "Coult not compute all the roots of the denominator of the "
            "transfer function.")
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
    system, pole_markersize, zero_markersize,
    **kwargs
):
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    if not isinstance(system, sm.control.lti.TransferFunction):
        system = tf_to_sympy(system)

    zeros, poles = _get_zeros_poles_from_symbolic_tf(system)
    zeros_re, zeros_im = [re(z) for z in zeros], [im(z) for z in zeros]
    poles_re, poles_im = [re(p) for p in poles], [im(p) for p in poles]

    prk, zrk, p_label, z_label = _pole_zero_common_keyword_arguments(
        pole_markersize, zero_markersize, **kwargs
    )
    z_series = List2DSeries(
        zeros_re, zeros_im, z_label,
        scatter=True, is_filled=True, rendering_kw=zrk,
        **kwargs
    )
    p_series = List2DSeries(
        poles_re, poles_im, p_label,
        scatter=True, is_filled=True, rendering_kw=prk,
        **kwargs
    )
    return [p_series, z_series]


def _pole_zero_with_control_helper(
    system, pole_markersize, zero_markersize,
    **kwargs
):
    params = kwargs.get("params", {})
    prk, zrk, p_label, z_label = _pole_zero_common_keyword_arguments(
        pole_markersize, zero_markersize, **kwargs
    )
    return [
        PoleZeroSeries(system, p_label, return_poles=True,
            rendering_kw=prk, params=params),
        PoleZeroSeries(system, z_label, return_poles=False,
            rendering_kw=zrk, params=params),
    ]


def _pole_zero_common_keyword_arguments(
    pole_markersize, zero_markersize, **kwargs
):
    label = kwargs.get("label", None)
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
    get_label = lambda t: t + " of " + label if label else t
    z_label = get_label("zeros")
    p_label = get_label("poles")
    return p_rendering_kw, z_rendering_kw, p_label, z_label


def pole_zero(
    system, pole_markersize=10, zero_markersize=7, show_axes=False,
    label=None, sgrid=False, zgrid=False, control=True,
    input=None, output=None, **kwargs
):
    """
    Computes the [Pole-Zero]_ plot (also known as PZ Plot or PZ Map) of
    a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : LTI system type
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
    **kwargs :
        See ``plot`` for a list of keyword arguments to further customize
        the resulting figure.

    Returns
    =======

    A list containing:

    * one instance of ``SGridLineSeries`` if ``sgrid=True``.
    * one instance of ``ZGridLineSeries`` if ``zgrid=True``.
    * one or more instances of ``List2DSeries`` if ``control=False``.
    * one or more instances of ``PoleZeroSeries`` if ``control=True``.

    Examples
    ========

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s
        from sympy import I
        from sympy.physics.control.lti import TransferFunction
        from spb import *
        tf1 = TransferFunction(
            s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        graphics(
            pole_zero(tf1, sgrid=True),
            grid=False, xlabel="Real", ylabel="Imaginary"
        )

    Plotting poles and zeros on the z-plane:

    .. plot::
        :context: close-figs
        :include-source: True

        graphics(
            pole_zero(tf1, zgrid=True),
            grid=False, xlabel="Real", ylabel="Imaginary"
        )

    If a transfer function has complex coefficients, make sure to request
    the evaluation using ``sympy`` instead of the ``control`` module:

    .. plot::
        :context: close-figs
        :include-source: True

        tf = TransferFunction(s + 2, s**2 + (2+I)*s + 10, s)
        graphics(
            control_axis(),
            pole_zero(tf, control=False),
            grid=False, xlabel="Real", ylabel="Imaginary"
        )

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
           grid=False, xlim=(-4, 1), ylim=(-4, 4),
           xlabel="Real", ylabel="Imaginary")

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot

    See Also
    ========

    sgrid, zgrid

    """
    control = _check_if_control_is_installed(use_control=control)
    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )
    func = _pole_zero_with_control_helper if control else _pole_zero_helper
    series = []
    for s, l in systems:
        kw = kwargs.copy()
        kw["label"] = l
        s = _preprocess_system(s, **kw)
        _check_system(s)
        series.extend(
            func(s, pole_markersize, zero_markersize, **kwargs.copy())
        )

    grid = _get_grid_series(sgrid, zgrid)
    return grid + series


def _get_grid_series(sgrid, zgrid):
    grid = []
    if sgrid and zgrid:
        warnings.warn(
            "``sgrid=True`` and ``zgrid=True`` is not supported. "
            "Automatically setting ``zgrid=False``.")
        zgrid = False
    if sgrid:
        grid = sgrid_function(auto=True)
    if zgrid:
        grid = zgrid_function()
    return grid


def _ilt(expr, s, t):
    """_fast_inverse_laplace and inverse_laplace_transform are not on-par
    feature wise. First attempt the fast approach, if it fails go for the
    slow approach.
    """
    try:
        y = _fast_inverse_laplace(expr, s, t)
    except NotImplementedError:
        y = inverse_laplace_transform(expr, s, t)
    return y


def _step_response_helper(
    system, label, lower_limit, upper_limit, prec, **kwargs
):
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    if not isinstance(system, sm.control.lti.TransferFunction):
        system = tf_to_sympy(system)

    expr = system.to_expr() / system.var
    expr = apart(expr, system.var, full=True)

    _x = Dummy("x")
    _y = _ilt(expr, system.var, _x).evalf(prec)
    # if `params` is given, _y might contain RootSum, which is not implemented
    # in Numpy. `doit()` is going to expand it, so that Numpy can be used.
    _y = _y.doit()

    return LineOver1DRangeSeries(
        _y, prange(_x, lower_limit, upper_limit),
        label, **kwargs
    )


def _step_response_with_control_helper(
    system, label, lower_limit, upper_limit, prec, **kwargs
):
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    expr = (system.to_expr() if isinstance(
        system, sm.control.lti.TransferFunction) else system)
    _x = Dummy("x")
    return SystemResponseSeries(
        expr, prange(_x, lower_limit, upper_limit),
        label, force_real_eval=True, response_type="step", **kwargs
    )


def _check_lower_limit_and_control(lower_limit, control):
    # allows parametric lower_limit
    lower_limit = sympify(lower_limit)
    if lower_limit.is_Number and lower_limit < 0:
        raise ValueError(
            "Lower limit of time must be greater than or equal to zero."
        )
    if (
        (lower_limit.is_Number and lower_limit > 0) or
        (len(lower_limit.free_symbols) > 0)
    ):
        warnings.warn(
            "You are evaluating a transfer function using the ``control`` "
            "module, but you also set ``lower_limit != 0``. This is likely "
            "going to produce incorrect results. Please, consider "
            "setting ``control=False`` when using ``lower_limit != 0``."
        )


def step_response(
    system, lower_limit=0, upper_limit=10, prec=8,
    label=None, rendering_kw=None, control=True,
    input=None, output=None, **kwargs
):
    """
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    system : LTI system type
        The system for which the step response plot is to be computed.
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
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0. If a different value
        is to be used, also set ``control=False`` (see examples in order to
        understand why).
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
    control : bool, optional
        If True, computes the step response with the ``control``
        module, which uses numerical integration. If False, computes the
        step response with ``sympy``, which uses the inverse Laplace transform.
        Default to True.
    control_kw : dict, optional
        A dictionary of keyword arguments passed to
        :py:func:`control.step_response`
    input : int, optional
        Only compute the step response for the listed input.  If not
        specified, the step responses for each independent input are
        computed (as separate traces).
    output : int, optional
        Only compute the step response for the listed output. If not
        specified, all outputs are reported.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one or more instances of:

    * ``LineOver1DRangeSeries`` if ``control=False``.
    * ``SystemResponseSeries`` if ``control=True``.

    Examples
    ========

    Plotting a SISO system:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s, t
        from sympy import Heaviside
        from sympy.physics.control.lti import TransferFunction
        from spb import *
        tf1 = TransferFunction(
            8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        graphics(
            line(Heaviside(t), (t, -1, 10), label="step"),
            step_response(tf1, label="response"),
            xlabel="Time [s]", ylabel="Amplitude"
        )

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
        graphics(
            step_response(tfm),
            xlabel="Time [s]", ylabel="Amplitude"
        )

    Plotting a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        G = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)
        graphics(
            step_response(G, upper_limit=15),
            xlabel="Time [s]", ylabel="Amplitude"
        )

    Interactive-widgets plot of multiple systems, one of which is parametric.
    A few observations:

    1. The first system's response will be computed with SymPy because
       ``control=False`` was set.
    2. The second system's response will be computed with the ``control``
       module, because ``control=True`` was set.
    3. Note the use of parametric ``lower_limit`` and ``upper_limit``.
    4. By moving the "lower limit" slider, the first system (evaluated with
       SymPy) will start from some amplitude value. However, on the second
       system (evaluated with the ``control`` module), the amplitude always
       starts from 0. That's because the numerical integration's initial
       condition is 0. Hence, if ``lower_limit`` is to be used, please
       set ``control=False``.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy.abc import a, b, c, d, e, f, g, s, t
       from sympy import Heaviside
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
           line(Heaviside(t), (t, -1, 10), label="step"),
           step_response(
               tf1, label="A", lower_limit=f, upper_limit=g, params=params,
               control=False),
           step_response(
               tf2, label="B", lower_limit=f, upper_limit=g, params=params,
               control=True),
           xlabel="Time [s]", ylabel="Amplitude"
       )

    See Also
    ========

    impulse_response, ramp_response

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    control = _check_if_control_is_installed(use_control=control)
    _check_lower_limit_and_control(lower_limit, control)
    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )

    func = _step_response_with_control_helper if control else _step_response_helper
    series = []
    for sys, lbl in systems:
        series.append(
            func(
                sys, lbl, lower_limit, upper_limit, prec,
                rendering_kw=rendering_kw, **kwargs)
        )

    return series


def _impulse_response_helper(
    system, label, lower_limit, upper_limit, prec, **kwargs
):
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    if not isinstance(system, sm.control.lti.TransferFunction):
        system = tf_to_sympy(system)

    _x = Dummy("x")
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    _y = _ilt(expr, system.var, _x).evalf(prec)
    _y = _y.doit()

    return LineOver1DRangeSeries(
        _y, prange(_x, lower_limit, upper_limit),
        label, **kwargs
    )


def _impulse_response_with_control_helper(
    system, label, lower_limit, upper_limit, prec, **kwargs
):
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    _x = Dummy("x")
    expr = (system.to_expr() if isinstance(
        system, sm.control.lti.TransferFunction) else system)

    return SystemResponseSeries(
        expr, prange(_x, lower_limit, upper_limit),
        label, force_real_eval=True, response_type="impulse", **kwargs
    )


def impulse_response(
    system, prec=8, lower_limit=0, upper_limit=10,
    label=None, rendering_kw=None, control=True,
    input=None, output=None, **kwargs
):
    """
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system

    Parameters
    ==========

    system : LTI system type
        The system for which the impulse response plot is to be computed.
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
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0. If a different value
        is to be used, also set ``control=False`` (see examples in order to
        understand why).
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
    control : bool, optional
        If True, computes the impulse response with the ``control``
        module, which uses numerical integration. If False, computes the
        impulse response with ``sympy``, which uses the inverse Laplace
        transform. Default to True.
    control_kw : dict, optional
        A dictionary of keyword arguments passed to
        :py:func:`control.impulse_response`
    input : int, optional
        Only compute the impulse response for the listed input.  If not
        specified, the impulse responses for each independent input are
        computed (as separate traces).
    output : int, optional
        Only compute the impulse response for the listed output. If not
        specified, all outputs are reported.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one or more instances of:

    * ``LineOver1DRangeSeries`` if ``control=False``.
    * ``SystemResponseSeries`` if ``control=True``.

    Examples
    ========

    Plotting a SISO system:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s
        from sympy.physics.control.lti import TransferFunction
        from spb import *
        tf1 = TransferFunction(
            8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        graphics(
            impulse_response(tf1),
            xlabel="Time [s]", ylabel="Amplitude"
        )

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
        graphics(
            impulse_response(tfm),
            xlabel="Time [s]", ylabel="Amplitude"
        )

    Plotting a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        G = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)
        graphics(
            impulse_response(G, upper_limit=15),
            xlabel="Time [s]", ylabel="Amplitude"
        )

    Interactive-widgets plot of multiple systems, one of which is parametric.
    A few observations:

    1. The first system's response will be computed with SymPy because
       ``control=False`` was set.
    2. The second system's response will be computed with the ``control``
       module, because ``control=True`` was set.
    3. Note the use of parametric ``lower_limit`` and ``upper_limit``.
    4. By moving the "lower limit" slider, the first system (evaluated with
       SymPy) will start from some amplitude value. However, on the second
       system (evaluated with the ``control`` module), the amplitude always
       starts from 0. That's because the numerical integration's initial
       condition is 0. Hence, if ``lower_limit`` is to be used, please
       set ``control=False``.

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
               tf1, label="A", lower_limit=g, upper_limit=h, params=params,
               control=True),
           impulse_response(
               tf2, label="B", lower_limit=g, upper_limit=h, params=params,
               control=False),
           xlabel="Time [s]", ylabel="Amplitude"
       )


    See Also
    ========

    step_response, ramp_response

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.impulse.html

    """
    control = _check_if_control_is_installed(use_control=control)
    _check_lower_limit_and_control(lower_limit, control)
    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )

    func = _impulse_response_with_control_helper if control else _impulse_response_helper
    series = []
    for sys, lbl in systems:
        series.append(
            func(
                sys, lbl, lower_limit, upper_limit, prec,
                rendering_kw=rendering_kw, **kwargs)
        )

    return series


def _ramp_response_helper(
    system, label, lower_limit, upper_limit, prec, slope, **kwargs
):
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    system = _preprocess_system(system, **kwargs)
    _check_system(system)

    if not isinstance(system, sm.control.lti.TransferFunction):
        system = tf_to_sympy(system)

    _x = Dummy("x")
    expr = (slope*system.to_expr()) / ((system.var)**2)
    expr = apart(expr, system.var, full=True)
    _y = _ilt(expr, system.var, _x).evalf(prec)
    _y = _y.doit()

    return LineOver1DRangeSeries(
        _y, prange(_x, lower_limit, upper_limit),
        label, **kwargs
    )


def _ramp_response_with_control_helper(
    system, label, lower_limit, upper_limit, prec, slope, **kwargs
):
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    system = _preprocess_system(system, **kwargs)
    _check_system(system)
    sp = import_module("scipy")

    if isinstance(system, sp.signal.TransferFunction):
        n, d = system.num, system.den
        n = [slope*t for t in n]
        kw = {}
        if system.dt is not None:
            kw["dt"] = system.dt
        expr = sp.signal.TransferFunction(n, d, **kw)
    elif isinstance(system, sm.control.lti.TransferFunction):
        expr = slope * system.to_expr()
    else:
        expr = slope * system

    _x = Dummy("x")
    return SystemResponseSeries(
        expr, prange(_x, lower_limit, upper_limit),
        label, force_real_eval=True, response_type="ramp", **kwargs
    )


def ramp_response(
    system, prec=8, slope=1, lower_limit=0, upper_limit=10,
    label=None, rendering_kw=None, control=True,
    input=None, output=None, **kwargs
):
    """
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line passing through origin
    ($f(x) = mx$). The slope of the ramp function can be varied by the
    user and the default value is 1.

    Parameters
    ==========

    system : LTI system type
        The system for which the ramp response plot is to be computed.
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
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0. If a different value
        is to be used, also set ``control=False`` (see examples in order to
        understand why).
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    control : bool, optional
        If True, computes the ramp response with the ``control``
        module, which uses numerical integration. If False, computes the
        ramp response with ``sympy``, which uses the inverse Laplace transform.
        Default to True.
    control_kw : dict, optional
        A dictionary of keyword arguments passed to
        :py:func:`control.forced_response`
    input : int, optional
        Only compute the ramp response for the listed input.  If not
        specified, the ramp responses for each independent input are
        computed (as separate traces).
    output : int, optional
        Only compute the ramp response for the listed output. If not
        specified, all outputs are reported.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing one or more instances of:

    * ``LineOver1DRangeSeries`` if ``control=False``.
    * ``SystemResponseSeries`` if ``control=True``.

    Examples
    ========

    Plotting a SISO system:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s, t
        from sympy.physics.control.lti import TransferFunction
        from spb import *
        tf1 = TransferFunction(1, (s+1), s)
        ul = 10
        graphics(
            line(t, (t, 0, ul), label="ramp"),
            ramp_response(tf1, upper_limit=ul, label="response"),
            xlabel="Time [s]", ylabel="Amplitude"
        )

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
        graphics(
            ramp_response(tfm),
            xlabel="Time [s]", ylabel="Amplitude"
        )

    Plotting a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        G = ct.tf([0.0244, 0.0236], [1.1052, -2.0807, 1.0236], dt=0.2)
        graphics(
            ramp_response(G, upper_limit=15),
            xlabel="Time [s]", ylabel="Amplitude"
        )

    Interactive-widgets plot of multiple systems, one of which is parametric.
    A few observations:

    1. The first system's response will be computed with SymPy because
       ``control=False`` was set.
    2. The second system's response will be computed with the ``control``
       module, because ``control=True`` was set.
    3. Note the use of parametric ``lower_limit`` and ``upper_limit``.
    4. By moving the "lower limit" slider, the first system (evaluated with
       SymPy) will start from some amplitude value. However, on the second
       system (evaluated with the ``control`` module), the amplitude always
       starts from 0. That's because the numerical integration's initial
       condition is 0. Hence, if ``lower_limit`` is to be used, please
       set ``control=False``.

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy import symbols
       from sympy.physics.control.lti import TransferFunction
       from spb import *
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
       graphics(
           line(a*t, (t, 0, c), params=params, label="ramp"),
           ramp_response(
               tf1, label="A", slope=a, lower_limit=b, upper_limit=c,
               params=params, control=False),
           ramp_response(
               tf2, label="B", slope=a, lower_limit=b, upper_limit=c,
               params=params, control=True),
           xlabel="Time [s]", ylabel="Amplitude", imodule="panel")

    See Also
    ========

    plot_step_response, plot_impulse_response

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    control = _check_if_control_is_installed(use_control=control)
    _check_lower_limit_and_control(lower_limit, control)
    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )

    non_symbolic_systems = any([
        not isinstance(s[0], (
            Expr, sm.control.lti.SISOLinearTimeInvariant)
        ) for s in systems])
    if (
        isinstance(slope, Expr) and
        (len(slope.free_symbols) > 0) and
        non_symbolic_systems
    ):
        raise ValueError(
            "You are using a symbolic `slope` with a non-symbolic "
            "transfer function. This mode of operation is not supported. "
            "Please, consider converting the transfer function to a "
            "transfer function from `sympy.physics.control`."
        )

    func = _ramp_response_with_control_helper if control else _ramp_response_helper
    series = []
    for sys, lbl in systems:
        series.append(
            func(
                sys, lbl, lower_limit, upper_limit, prec, slope,
                rendering_kw=rendering_kw, **kwargs)
        )

    return series


def _bode_common(system, label, initial_exp, final_exp, freq_unit, **kwargs):
    # NOTE: why use ``sympy`` and not ``control`` to compute bode plots?
    # Because I can easily deal with time delays.

    original_system = _preprocess_system(system, **kwargs)
    system = tf_to_sympy(original_system, skip_check_dt=True)
    _check_system(system, bypass_delay_check=True)

    expr = system.to_expr()
    _w = Dummy("w", real=True)
    params = kwargs.get("params", None)

    nyquistfrq = None
    if is_discrete_time(original_system):
        if freq_unit == 'Hz':
            repl = exp(I * _w * 2*pi * original_system.dt)
        else:
            repl = exp(I * _w * original_system.dt)

        nyquistfrq = pi / original_system.dt
    else:
        if freq_unit == 'Hz':
            repl = I*_w*2*pi
        else:
            repl = I*_w

    if (initial_exp is None) or (final_exp is None):
        if params:
            initial_params = _get_initial_params(params)
            new_system = system.subs(initial_params)
            # assume any time delay is in cascade, so that only the phase is
            # affected by it.
            for d in tf_find_time_delay(new_system):
                new_system = new_system.subs({d: 1})
            tf = tf_to_control(new_system)
        else:
            new_system = system
            # assume any time delay is in cascade, so that only the phase is
            # affected by it.
            for d in tf_find_time_delay(new_system):
                new_system = new_system.subs({d: 1})
            tf = tf_to_control(new_system)
        i, f = _default_frequency_exponent_range(tf, freq_unit == 'Hz', 1)
        initial_exp = i if initial_exp is None else initial_exp
        final_exp = f if final_exp is None else final_exp

    w_expr = expr.subs({system.var: repl})
    _range = prange(
        _w,
        10**initial_exp,
        10**final_exp if nyquistfrq is None else nyquistfrq
    )
    return w_expr, _range


def _bode_magnitude_helper(
    system, label, initial_exp, final_exp, freq_unit, **kwargs
):
    w_expr, _range = _bode_common(
        system, label, initial_exp, final_exp, freq_unit, **kwargs)
    mag = 20*log(Abs(w_expr), 10)
    return LineOver1DRangeSeries(
        mag, _range, label, xscale='log', **kwargs)


def bode_magnitude(
    system, initial_exp=None, final_exp=None, freq_unit='rad/sec',
    phase_unit='rad', label=None, rendering_kw=None,
    input=None, output=None, **kwargs
):
    """
    Returns the Bode magnitude plot of a continuous-time system.

    Parameters
    ==========

    system : LTI system type
        The system for which the step response plot is to be computed.
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
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Default to None, which
        will autocompute the appropriate value.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Default to None, which
        will autocompute the appropriate value.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'``
        (Hertz) as frequency units.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
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

    Bode magnitude plot of a continuous-time system:

    .. plot::
        :context: reset
        :include-source: True

        from sympy.abc import s
        from sympy.physics.control.lti import TransferFunction
        from spb import *
        tf1 = TransferFunction(
            1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        graphics(
            bode_magnitude(tf1),
            xscale="log", xlabel="Frequency [rad/s]",
            ylabel="Magnitude [dB]"
        )

    Bode magnitude plot of a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        tf2 = ct.tf([1], [1, 2, 3], dt=0.05)
        graphics(
            bode_magnitude(tf2),
            xscale="log", xlabel="Frequency [rad/s]",
            ylabel="Magnitude [dB]"
        )

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
           imodule="panel", ncols=3,
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

    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )

    series = []
    for sys, lbl in systems:
        series.append(
            _bode_magnitude_helper(
                sys, lbl, initial_exp, final_exp,
                freq_unit, rendering_kw=rendering_kw, **kwargs
            )
        )

    return series


def _bode_phase_helper(
    system, label, initial_exp, final_exp, freq_unit, phase_unit,
    unwrap, **kwargs
):
    w_expr, _range = _bode_common(
        system, label, initial_exp, final_exp, freq_unit, **kwargs)

    if phase_unit == 'deg':
        phase = arg(w_expr)*180/pi
        if unwrap is True:
            unwrap = {"period": 360}
    else:
        phase = arg(w_expr)

    return LineOver1DRangeSeries(
        phase, _range, label, xscale='log', unwrap=unwrap, **kwargs)


def bode_phase(
    system, initial_exp=None, final_exp=None, freq_unit='rad/sec',
    phase_unit='rad', label=None, rendering_kw=None, unwrap=True,
    input=None, output=None, **kwargs
):
    """
    Returns the Bode phase plot of a continuous-time system.

    Parameters
    ==========

    system : LTI system type
        The system for which the step response plot is to be computed.
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
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Default to None, which
        will autocompute the appropriate value.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Default to None, which
        will autocompute the appropriate value.
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
        Depending on the transfer function, the computed phase could contain
        discontinuities of 2*pi. ``unwrap=True`` post-process the numerical
        data in order to get a continuous phase.
        Default to True.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
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

    Bode phase plot of a continuous-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        from sympy.abc import s
        from sympy.physics.control.lti import TransferFunction
        from spb import *
        tf1 = TransferFunction(
            1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        graphics(
            bode_phase(tf1, initial_exp=0.2, final_exp=0.7),
            xscale="log", xlabel="Frequency [rad/s]",
            ylabel="Magnitude [dB]"
        )

    Bode phase plot of a discrete-time system:

    .. plot::
        :context: close-figs
        :include-source: True

        import control as ct
        tf2 = ct.tf([1], [1, 2, 3], dt=0.05)
        graphics(
            bode_phase(tf2),
            xscale="log", xlabel="Frequency [rad/s]",
            ylabel="Magnitude [dB]"
        )

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
           imodule="panel", ncols=3,
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

    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )

    series = []
    for sys, lbl in systems:
        series.append(
            _bode_phase_helper(
                sys, lbl, initial_exp, final_exp,
                freq_unit, phase_unit, rendering_kw=rendering_kw,
                unwrap=unwrap, **kwargs
            )
        )

    return series


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


def nyquist(system, omega_limits=None, input=None, output=None,
    label=None, rendering_kw=None, m_circles=False, **kwargs):
    """Plots a Nyquist plot for the system over a (optional) frequency range.
    The curve is computed by evaluating the Nyquist segment along the positive
    imaginary axis, with a mirror image generated to reflect the negative
    imaginary axis. Poles on or near the imaginary axis are avoided using a
    small indentation. The portion of the Nyquist contour at infinity is not
    explicitly computed (since it maps to a constant value for any system with
    a proper transfer function).

    Parameters
    ==========

    system : LTI system type
        The system for which the step response plot is to be computed.
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

    Returns
    =======

    A list containing:

    * one instance of ``MCirclesSeries`` if ``mcircles=True``.
    * one instance of ``NyquistLineSeries``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hall_circles

    See Also
    ========

    bode, nichols, mcircles

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
       from spb import *
       tf1 = TransferFunction(
           4 * s**2 + 5 * s + 1, 3 * s**2 + 2 * s + 5, s)
       graphics(
           nyquist(tf1, m_circles=True),
           xlabel="Real", ylabel="Imaginary",
           grid=False, aspect="equal"
       )

    Visualizing M-circles:

    .. plot::
       :context: close-figs
       :include-source: True

       graphics(
           nyquist(tf1, m_circles=True),
           xlabel="Real", ylabel="Imaginary"
       )

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
           xlabel="Real", ylabel="Imaginary",
           xlim=(-1, 4), ylim=(-2.5, 2.5), aspect="equal"
       )

    """
    control = _check_if_control_is_installed(force_stop=True)
    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )

    params = kwargs.get("params", None)
    initial_params = None
    if params:
        initial_params = _get_initial_params(params)

    omega = symbols("omega")
    series = []
    for sys, lbl in systems:
        ol = omega_limits
        sys = _preprocess_system(sys, **kwargs)

        if params:
            ctrl_sys = tf_to_control(sys.subs(initial_params))
        else:
            ctrl_sys = tf_to_control(sys)

        if ol is None:
            # NOTE: I could use _default_frequency_exponent_range to compute
            # the omega_limits, however there is some bugs in
            # ``control.nyquist_plot`` that would make horrible plots in case
            # of discrete time systems. Hence, I set the exponents to be the
            # same: I can catch the case where they are different inside
            # NyquistLineSeries.
            ol = (omega, -1, -1)
        else:
            ol = prange(omega, *omega_limits)

        series.append(
            NyquistLineSeries(
                sys, ol, lbl, rendering_kw=rendering_kw, **kwargs
            )
        )
    grid = []
    if m_circles is True:
        grid = mcircles_func()
    elif hasattr(m_circles, "__iter__"):
        grid = mcircles_func(m_circles)

    return grid + series


def _nichols_helper(system, label, **kwargs):
    system = _preprocess_system(system, **kwargs)
    _check_system(system)
    system = tf_to_sympy(system)
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


def nichols(system, label=None, rendering_kw=None, ngrid=True,
    input=None, output=None, **kwargs):
    """Nichols plot for a system over a (optional) frequency range.

    Parameters
    ==========

    system : LTI system type
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
    input : int, optional
        Only compute the poles/zeros for the listed input. If not specified,
        the poles/zeros for each independent input are computed (as
        separate traces).
    output : int, optional
        Only compute the poles/zeros for the listed output.
        If not specified, all outputs are reported.
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_2d.line_parametric_2d`.
        Refer to its documentation for a for a full list of keyword arguments.

    Returns
    =======

    A list containing:

    * one instance of ``NGridLineSeries`` if ``ngrid=True``.
    * one instance of ``NicholsLineSeries``.

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
       from spb import *
       tf = TransferFunction(50*s**2 - 20*s + 15, -10*s**2 + 40*s + 30, s)
       graphics(
           nichols(tf),
           xlabel="Open-Loop Phase [deg]",
           ylabel="Open-Loop Magnitude [dB]",
           grid=False
       )

    Turning off the Nichols grid lines:

    .. plot::
       :context: close-figs
       :include-source: True

       graphics(
           nichols(tf, ngrid=False),
           xlabel="Open-Loop Phase [deg]",
           ylabel="Open-Loop Magnitude [dB]",
           grid=False
       )

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
           xlim=(-360, 360), grid=False,
       )

    See Also
    ========

    bode_magnitude, bode_phase, nyquist, ngrid

    """
    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )

    series = []
    for s, l in systems:
        s = _preprocess_system(s, **kwargs)
        _check_system(s)
        series.append(
            _nichols_helper(s, l, rendering_kw=rendering_kw, **kwargs.copy())
        )

    grid = []
    if ngrid:
        grid = ngrid_function()
    return grid + series


def root_locus(system, label=None, rendering_kw=None, rl_kw={},
    sgrid=True, zgrid=False, input=None, output=None, **kwargs):
    """Root Locus plot for a system.

    Parameters
    ==========

    system : LTI system type
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

    Returns
    =======

    A list containing:

    * one instance of ``SGridLineSeries`` if ``sgrid=True``.
    * one instance of ``ZGridLineSeries`` if ``zgrid=True``.
    * one or more instances of ``RootLocusSeries``.

    Examples
    ========

    Plot the root locus of a system on the s-plane, also showing a custom
    damping ratio line.

    .. plot::
       :context: reset
       :include-source: True

       from sympy.abc import s, z
       from spb import *
       G = (s**2 - 4) / (s**3 + 2*s - 3)
       graphics(
           root_locus(G),
           sgrid(xi=0.92, wn=False, rendering_kw={"color": "r"}),
           grid=False, xlabel="Real", ylabel="Imaginary")

    Plot the root locus of a discrete system on the z-plane:

    .. plot::
       :context: close-figs
       :include-source: True

       G = (0.038*z + 0.031) / (9.11*z**2 - 13.77*z + 5.0)
       graphics(
           root_locus(G, sgrid=False, aspect="equal"),
           zgrid(T=0.2),
           grid=False, xlabel="Real", ylabel="Imaginary")

    Interactive-widgets root locus plot:

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy import symbols
       from spb import *
       a, s, xi = symbols("a, s, xi")
       G = (s**2 + a) / (s**3 + 2*s**2 + 3*s + 4)
       params={a: (-0.5, -4, 4), xi: (0.8, 0, 1)}
       graphics(
           sgrid(xi, wn=False, params=params, rendering_kw={"color": "r"}),
           root_locus(G, params=params),
           grid=False, xlim=(-4, 1), ylim=(-2.5, 2.5),
           xlabel="Real", ylabel="Imaginary")

    See Also
    ========

    sgrid, zgrid
    """
    control = _check_if_control_is_installed(force_stop=True)
    systems = _unpack_mimo_systems(
        system,
        "" if label is None else label,
        input, output
    )
    series = []
    for s, l in systems:
        s = _preprocess_system(s, **kwargs)
        _check_system(s)
        series.append(
            RootLocusSeries(
                s, label=l, rendering_kw=rendering_kw, rl_kw=rl_kw,
                **kwargs.copy())
        )

    grid = _get_grid_series(sgrid, zgrid)
    return grid + series


def sgrid(xi=None, wn=None, tp=None, ts=None, xlim=None, ylim=None,
    show_control_axis=True, rendering_kw=None, auto=False, **kwargs):
    """Create the s-grid of constant damping ratios and natural frequencies.

    Parameters
    ==========

    xi : iterable or float, optional
        Damping ratios. Must be ``0 <= xi <= 1``.
        If ``None``, default damping ratios will be used. If ``False``,
        no damping ratios will be visualized.
    wn : iterable or float, optional
        Natural frequencies.
        If ``None``, default natural frequencies will be used. If ``False``,
        no natural frequencies will be visualized.
    tp : iterable or float, optional
        Peak times.
    ts : iterable or float, optional
        Settling times.
    auto : bool, optional
        If True, automatically compute damping ratio and natural frequencies
        in order to obtain a "evenly" distributed grid.
    show_control_axis : bool, optional
        Shows an horizontal and vertical grid lines crossing at the origin.
        Default to True.
    xlim, ylim : 2-elements tuple
        If provided, compute damping ratios and natural frequencies in order
        to display "evenly" distributed grid lines on the plane.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

    Examples
    ========

    Shows the default grid lines, as well as a custom damping ratio line,
    a custom natural frequency line, a custom peak time line and a custom
    settling time line.

    .. plot::
       :context: reset
       :include-source: True

       from spb import *
       graphics(
           sgrid(),
           sgrid(xi=0.85, wn=False,
               rendering_kw={"color": "r", "linestyle": "-"},
               show_control_axis=False),
           sgrid(xi=False, wn=4.5,
               rendering_kw={"color": "g", "linestyle": "-"},
               show_control_axis=False),
           sgrid(xi=False, wn=False, tp=1,
               rendering_kw={"color": "b", "linestyle": "-"},
               show_control_axis=False),
           sgrid(xi=False, wn=False, ts=1,
               rendering_kw={"color": "m", "linestyle": "-"},
               show_control_axis=False),
           grid=False, xlim=(-8.5, 1), ylim=(-5, 5))

    Auto-generate grid lines over a specified area of of the s-plane:

    .. plot::
       :context: close-figs
       :include-source: True

       xlim = (-5.5, 1)
       ylim = (-3, 3)
       graphics(
           sgrid(auto=True),
           grid=False, xlim=xlim, ylim=ylim
       )

    Interactive-widgets plot of custom s-grid lines:

    .. panel-screenshot::
       :small-size: 800, 750

       from sympy import symbols
       from spb import *
       xi, wn, Tp, Ts = symbols("xi omega_n T_p T_s")
       params = {
           xi: (0.85, 0, 1),
           wn: (6.5, 2, 8),
           Tp: (1, 0.2, 4),
           Ts: (1, 0.2, 4),
       }
       graphics(
           sgrid(auto=True),
           sgrid(xi=xi, wn=False, params=params,
               rendering_kw={"color": "r", "linestyle": "-"},
               show_control_axis=False),
           sgrid(xi=False, wn=wn, params=params,
               rendering_kw={"color": "g", "linestyle": "-"},
               show_control_axis=False),
           sgrid(xi=False, wn=False, tp=Tp, params=params,
               rendering_kw={"color": "b", "linestyle": "-"},
               show_control_axis=False),
           sgrid(xi=False, wn=False, ts=Ts, params=params,
               rendering_kw={"color": "m", "linestyle": "-"},
               show_control_axis=False),
           grid=False, xlim=(-8.5, 1), ylim=(-5, 5))

    See Also
    ========
    zgrid, pole_zero, root_locus

    """
    if (xi is not None) and (wn is False):
        # If we are showing a specific value of damping ratio, don't show
        # control axis. They are likely already shown by some other
        # SGridLineSeries.
        show_control_axis = False

    if xi is None:
        xi = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .96, .99, 1]
        if show_control_axis:
            # `xi` and `wn` lines also show an annotation with their
            # respective values. When `show_control_axis=True` it is easier
            # to plot horizontal and vertical lines (xi=1 and xi=0,
            # respectively). Also, this allows me not to show annotations
            # for these two values which would make things confusing.
            # For example, the annotation for `xi=1` could be confused
            # with an `wn` annotation.
            xi = xi[1:-1]
    elif xi is False:
        xi = []
    elif not hasattr(xi, "__iter__"):
        xi = [xi]

    if wn is None:
        wn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif wn is False:
        wn = []
    elif not hasattr(wn, "__iter__"):
        wn = [wn]

    if not tp:
        tp = []
    elif not hasattr(tp, "__iter__"):
        tp = [tp]

    if not ts:
        ts = []
    elif not hasattr(ts, "__iter__"):
        ts = [ts]

    params = kwargs.get("params", None)
    if (
        any(isinstance(t, Expr) and (not is_number(t)) for t in xi+wn+tp+ts)
        and (params is None)
    ):
        raise ValueError(
            "The provided natural frequencies or damping ratios "
            "contains symbolic expressions, but ``params`` was not "
            "provided. Cannot continue."
        )

    return [
        SGridLineSeries(xi, wn, tp, ts, xlim=xlim, ylim=ylim,
            show_control_axis=show_control_axis,
            rendering_kw=rendering_kw, auto=auto, **kwargs)
    ]


sgrid_function = sgrid


def zgrid(xi=None, wn=None, tp=None, ts=None, T=None,
    show_control_axis=True, rendering_kw=None, **kwargs):
    """Create the s-grid of constant damping ratios and natural frequencies.

    Parameters
    ==========

    xi : iterable or float, optional
        Damping ratios. Must be ``0 <= xi <= 1``.
        If ``None``, default damping ratios will be used. If ``False``,
        no damping ratios will be visualized.
    wn : iterable or float, optional
        Normalized natural frequencies.
        If ``None``, default natural frequencies will be used. If ``False``,
        no natural frequencies will be visualized.
    tp : iterable or float, optional
        Normalized peak times.
    ts : iterable or float, optional
        Normalized settling times.
    T : float or None, optional
        Sampling period.
    show_control_axis : bool, optional
        Shows an horizontal and vertical grid lines crossing at the origin.
        Default to True.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

    Examples
    ========

    Shows the default grid lines, as well as a custom damping ratio line and
    natural frequency line.

    .. plot::
       :context: reset
       :include-source: True

       from spb import *
       graphics(
           zgrid(),
           zgrid(xi=0.05, wn=False, rendering_kw={"color": "r", "linestyle": "-"}),
           zgrid(xi=False, wn=0.25, rendering_kw={"color": "b", "linestyle": "-"}),
           grid=False, aspect="equal", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

    Shows a grid of settling times and peak times:

    .. plot::
       :context: close-figs
       :include-source: True

       graphics(
           zgrid(xi=False, wn=False, tp=[3, 5, 7, 10, 20], ts=[2, 3, 5, 10, 20]),
           zgrid(xi=False, wn=False, tp=4, rendering_kw={"color": "r"}),
           zgrid(xi=False, wn=False, ts=7, rendering_kw={"color": "b"}),
           grid=False, aspect="equal", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

    Interactive-widgets plot of z-grid lines:

    .. panel-screenshot::
       :small-size: 800, 725

       from sympy import symbols
       from spb import *
       xi, wn, Tp, Ts = symbols("xi, omega_n, T_p, T_s")
       graphics(
           zgrid(),
           zgrid(xi=xi, wn=False, rendering_kw={"color": "r", "linestyle": "-"}, params={xi: (0.05, 0, 1)}),
           zgrid(wn=wn, xi=False, rendering_kw={"color": "g", "linestyle": "-"}, params={wn: (0.45, 0, 1)}),
           zgrid(wn=False, xi=False, tp=Tp, rendering_kw={"color": "b", "linestyle": "-"}, params={Tp: (3, 0, 20)}),
           zgrid(wn=False, xi=False, ts=Ts, rendering_kw={"color": "m", "linestyle": "-"}, params={Ts: (5, 0, 20)}),
           grid=False, aspect="equal", xlabel="Real", ylabel="Imaginary")

    See Also
    ========
    sgrid, pole_zero, root_locus

    """
    if (
        ((xi is not None) and (wn is False)) or
        ((wn is not None) and (xi is False))
    ):
        # If we are showing a specific value of damping ratio or natural
        # frequency, don't show control axis. They are likely already shown
        # by some other ZGridLineSeries.
        show_control_axis = False

    if xi is None:
        xi = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    elif xi is False:
        xi = []
    elif not hasattr(xi, "__iter__"):
        xi = [xi]
        if any(is_number(t) and t > 1 for t in xi):
            raise ValueError(
                "Damping ratios must be: 0 <= xi <= 1."
            )

    if wn is None:
        wn = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    elif wn is False:
        wn = []
    elif not hasattr(wn, "__iter__"):
        wn = [wn]
        if any(is_number(t) and t > 1 for t in wn):
            raise ValueError(
                "Natural frequencies must be normalized: 0 <= wn <= 1."
            )

    if not tp:
        tp = []
    elif not hasattr(tp, "__iter__"):
        tp = [tp]

    if not ts:
        ts = []
    elif not hasattr(ts, "__iter__"):
        ts = [ts]

    params = kwargs.get("params", None)
    if (
        any(isinstance(t, Expr) and (not is_number(t)) for t in xi+wn+tp+ts)
        and (params is None)
    ):
        raise ValueError(
            "The provided natural frequencies or damping ratios "
            "contains symbolic expressions, but ``params`` was not "
            "provided. Cannot continue."
        )

    return [
        ZGridLineSeries(xi, wn, tp, ts, T=T, rendering_kw=rendering_kw,
            show_control_axis=show_control_axis, **kwargs)
    ]


zgrid_function = zgrid


def ngrid(
    cl_mags=None, cl_phases=None, label_cl_phases=False,
    rendering_kw=None, **kwargs
):
    """Create the n-grid (Nichols grid) of constant closed-loop magnitudes
    and phases.

    Parameters
    ==========

    cl_mags : float or array-like (dB), optional
        Array of closed-loop magnitudes defining the iso-gain lines.
        If False, hide closed-loop magnitude lines.
    cl_phases : float or array-like (degrees), optional
        Array of closed-loop phases defining the iso-phase lines.
        Must be in the range -360 < cl_phases < 0.
        If False, hide closed-loop phase lines.
    label_cl_phases: bool, optional
        If True, closed-loop phase lines will be labelled. Default to False.
    rendering_kw : dict or None, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

    Returns
    =======

    A list containing one instance of ``NGridLineSeries``.

    Examples
    ========

    Default N-grid:

    .. plot::
       :context: reset
       :include-source: True

       from spb import *
       graphics(
           ngrid(),
           grid=False
       )

    Highlight specific values of closed-loop magnitude and closed-loop phase:

    .. plot::
       :context: close-figs
       :include-source: True

       graphics(
           ngrid(label_cl_phases=False),
           ngrid(cl_mags=-30, cl_phases=False, rendering_kw={"color": "r", "linestyle": "-"}),
           ngrid(cl_mags=False, cl_phases=-200, rendering_kw={"color": "g", "linestyle": "-"}),
           grid=False
       )

    See Also
    ========

    nichols

    """

    show_cl_mags = True
    show_cl_phases = True
    if cl_mags is False:
        cl_mags = None
        show_cl_mags = False
    elif (cl_mags is not None) and (not hasattr(cl_mags, "__iter__")):
        cl_mags = [cl_mags]

    if cl_phases is False:
        cl_phases = None
        show_cl_phases = False
    elif (cl_phases is not None) and (not hasattr(cl_phases, "__iter__")):
        cl_phases = [cl_phases]

    return [
        NGridLineSeries(cl_mags, cl_phases, label_cl_phases,
            rendering_kw=rendering_kw,
            show_cl_mags=show_cl_mags, show_cl_phases=show_cl_phases,
            **kwargs)
    ]


ngrid_function = ngrid


def mcircles(magnitudes_db=None, rendering_kw=None, **kwargs):
    """Draw M-circles of constant closed-loop magnitude.

    Parameters
    ==========
    magnitudes_db : float, iterable or None
        Specify the magnitudes in dB.
        If None, a list of default magnitudes will be used.
    rendering_kw : dict or None, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

    Returns
    =======

    A list containing one instance of ``MCirclesSeries``.

    Examples
    ========

    .. plot::
       :context: close-figs
       :include-source: True

       from spb import *
       graphics(
           mcircles(),
           mcircles(-3, rendering_kw={"color": "r"}),
           grid=False, aspect="equal")

    Interactive-widgets plot of m-circles:

    .. panel-screenshot::
       :small-size: 800, 675

       from spb import *
       from sympy.abc import m
       graphics(
           mcircles(),
           mcircles(m, rendering_kw={"color": "r"}, params={m: (-3, -15, 15)}),
           grid=False, aspect="equal")

    """
    math = import_module("math")
    if magnitudes_db is None:
        dbs = [-20, -10, -6, -4, -2, 0, 2, 4, 6, 10, 20]
    elif not hasattr(magnitudes_db, "__iter__"):
        dbs = [magnitudes_db]
    else:
        dbs = magnitudes_db

    magnitudes = [10**(t / 20) for t in dbs]
    return [
        MCirclesSeries(dbs, magnitudes,
            rendering_kw=rendering_kw, **kwargs)
    ]

mcircles_func = mcircles

def _default_frequency_exponent_range(
    syslist, Hz=None, feature_periphery_decades=None
):
    """Compute the exponents to be used with ``numpy.logspace`` in order
    to get a reasonable default frequency range for frequency domain plots.

    This code looks at the poles and zeros of all of the systems that
    we are plotting and sets the frequency range to be one decade above
    and below the min and max feature frequencies, rounded to the nearest
    integer.  If no features are found, it returns logspace(-1, 1).

    This function is a modified form of
    ``control.freqplot._default_frequency_range``.

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK)
    Hz : bool, optional
        If True, the limits (first and last value) of the frequencies
        are set to full decades in Hz so it fits plotting with logarithmic
        scale in Hz otherwise in rad/s. Omega is always returned in rad/sec.
    feature_periphery_decades : float, optional
        Defines how many decades shall be included in the frequency range on
        both sides of features (poles, zeros).  The default value is read from
        ``config.defaults['freqplot.feature_periphery_decades']``.

    Returns
    -------
    lsp_min, lsp_max : int
        Lower and upper exponents to be used with numpy.logspace.

    Examples
    --------
    G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    omega_range = _default_frequency_exponent_range(G)
    omega_range
    (-1.0, 2.0)

    """
    np = import_module("numpy")

    if feature_periphery_decades is None:
        feature_periphery_decades = 1

    # Find the list of all poles and zeros in the systems
    features = np.array(())
    freq_interesting = []

    # detect if single sys passed by checking if it is sequence-like
    if not hasattr(syslist, '__iter__'):
        syslist = (syslist,)

    for sys in syslist:
        try:
            # Add new features to the list
            if sys.isctime():
                features_ = np.concatenate(
                    (np.abs(sys.poles()), np.abs(sys.zeros())))
                # Get rid of poles and zeros at the origin
                toreplace = np.isclose(features_, 0.0)
                if np.any(toreplace):
                    features_ = features_[~toreplace]
            elif sys.isdtime(strict=True):
                fn = np.pi * 1. / sys.dt
                # TODO: What distance to the Nyquist frequency is appropriate?
                freq_interesting.append(fn * 0.9)

                features_ = np.concatenate((sys.poles(), sys.zeros()))
                # Get rid of poles and zeros on the real axis (imag==0)
               # * origin and real < 0
                # * at 1.: would result in omega=0. (logaritmic plot!)
                toreplace = np.isclose(features_.imag, 0.0) & (
                                    (features_.real <= 0.) |
                                    (np.abs(features_.real - 1.0) < 1.e-10))
                if np.any(toreplace):
                    features_ = features_[~toreplace]
                # TODO: improve
                features_ = np.abs(np.log(features_) / (1.j * sys.dt))
            else:
                # TODO
                raise NotImplementedError(
                    "type of system in not implemented now")
            features = np.concatenate((features, features_))
        except NotImplementedError:
            pass

    # Make sure there is at least one point in the range
    if features.shape[0] == 0:
        features = np.array([1.])

    if Hz:
        features /= 2. * np.pi
    features = np.log10(features)
    lsp_min = np.rint(np.min(features) - feature_periphery_decades)
    lsp_max = np.rint(np.max(features) + feature_periphery_decades)
    if Hz:
        lsp_min += np.log10(2. * np.pi)
        lsp_max += np.log10(2. * np.pi)

    if freq_interesting:
        lsp_min = min(lsp_min, np.log10(min(freq_interesting)))
        lsp_max = max(lsp_max, np.log10(max(freq_interesting)))

    return lsp_min, lsp_max
