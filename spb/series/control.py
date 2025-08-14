import math
import numpy as np
import param
from numbers import Number
from inspect import signature
from spb.wegert import wegert
from spb.defaults import cfg
from spb.utils import (
    _get_free_symbols, unwrap, extract_solution, tf_to_control
)
# import sympy
from sympy import latex, Tuple, symbols, Expr, Poly
from sympy.external import import_module
import warnings
from spb.series.evaluator import (
    _discretize,
    _GridEvaluationParameters,
    GridEvaluator,
    _update_range_value
)
from spb.series.base import BaseSeries, _get_wrapper_for_expr
from spb.series.series_2d_3d import Line2DBaseSeries, List2DSeries


class GridBase(param.Parameterized):
    """
    *GridLineSeries may cover the entire visible area. Hence, they need to
    know the axis limits.

    Axis limits can be:
    1. provided by the user in the plot function call. For example:
       ``plot(..., xlim=(a, b), ylim=(c, d))``
    2. computed from the data that was already plotted.
    3. provided in some function call that generates data series. For example:
       ``graphics(sgrid(xlim=(a, b), ylim=(c, d)))``

    Either way, the appropriate renderer will:

    1. figure it out the axis limits.
    2. Let the grid series knows about this limits by calling
       ``series.set_axis_limits(xlim, ylim)``.
    3. Compute the numerical data for the specified grid that cover the
       specified area, with ``series.get_data()``.

    """
    is_grid = True

    xlim = param.Tuple(default=None, length=2, doc="""
        Axis limits along the x-direction.""")
    ylim = param.Tuple(default=None, length=2, doc="""
        Axis limits along the x-direction.""")

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("show_in_legend", False)
        xlim = kwargs.get("xlim", None)
        ylim = kwargs.get("ylim", None)
        # The algorithm expectes the elements to be `float`.
        kwargs["xlim"] = tuple([float(t) for t in xlim]) if xlim else None
        kwargs["ylim"] = tuple([float(t) for t in ylim]) if ylim else None
        super().__init__(*args, **kwargs)


    def set_axis_limits(self, xlim, ylim):
        self.xlim = tuple(xlim)
        self.ylim = tuple(ylim)


class _NaturalFrequencyDampingRatioGrid(param.Parameterized):
    auto = param.Boolean(True, doc="""
        If True, automatically compute damping ratio and natural frequencies
        in order to obtain a "evenly" distributed grid.""")
    show_control_axis = param.Boolean(True, doc="""
        Shows an horizontal and vertical grid lines crossing at the origin.""")
    xi = param.List([], doc="""
        List of damping values where to draw a grid line.""")
    wn = param.List([], doc="""
        List of natural frequencies where to draw a grid line.""")
    tp = param.List([], doc="""
        List of peak times where to draw a grid line.""")
    ts = param.List([], doc="""
        List of settling times where to draw a grid line.""")


class SGridLineSeries(_NaturalFrequencyDampingRatioGrid, GridBase, BaseSeries):
    """Represent a grid of damping ratio lines and natural frequency lines
    on the s-plane. This data series implements two modes of operation:

    1. User can provide xi, wn.
    2. User can provide dummy xi, wn, and a list of associated RootLocusSeries.
       When ``get_data()`` will be called, it first loops over the associated
       root locus series in order to determine the axis limits of the visible
       area. Then, it computes new values of xi, wn in order to get grid lines
       "evenly" distributed on the available space.
    """

    def __init__(self, xi, wn, tp, ts, series=[], **kwargs):
        super().__init__(xi=xi, wn=wn, tp=tp, ts=ts, **kwargs)

    def __str__(self):
        return "s-grid"

    def _sgrid_default_xi(self, xlim, ylim):
        """Return default list of damping coefficients

        This function computes a list of damping coefficients based on the limits
        of the graph.  A set of 4 damping coefficients are computed for the x-axis
        and a set of three damping coefficients are computed for the y-axis
        (corresponding to the normal 4:3 plot aspect ratio in `matplotlib`?).

        Parameters
        ----------
        xlim : array_like
            List of x-axis limits [min, max]
        ylim : array_like
            List of y-axis limits [min, max]

        Returns
        -------
        zeta : list
            List of default damping coefficients for the plot

        """
        np = import_module("numpy")

        x_lower_lim = xlim[0] if xlim else -10
        y_upper_lim = ylim[1] if ylim else 10

        # Damping coefficient lines that intersect the x-axis
        sep1 = -x_lower_lim / 4
        ang1 = [np.arctan((sep1*i)/y_upper_lim) for i in np.arange(1, 4, 1)]

        # Damping coefficient lines that intersection the y-axis
        sep2 = y_upper_lim / 3
        ang2 = [
            np.arctan(-x_lower_lim/(y_upper_lim-sep2*i))
            for i in np.arange(1, 3, 1)
        ]

        # Put the lines together and add one at -pi/2 (negative real axis)
        angles = np.concatenate((ang1, ang2))

        # Return the damping coefficients corresponding to these angles
        zeta = np.sin(angles).tolist()
        if not self.show_control_axis:
            zeta += [0, 1]
        return zeta

    def _sgrid_default_wn(self, xlim, ylim, max_lines=7):
        """Return default wn for root locus plot

        This function computes a list of natural frequencies based on the grid
        parameters of the graph.

        Parameters
        ----------
        xloc : array_like
            List of x-axis tick values
        ylim : array_like
            List of y-axis limits [min, max]
        max_lines : int, optional
            Maximum number of frequencies to generate (default = 7)

        Returns
        -------
        wn : list
            List of default natural frequencies for the plot

        """
        lower_lim = xlim[0] if xlim else -10
        np = import_module("numpy")
        available_width = 0 - lower_lim
        wn = np.linspace(0, abs(lower_lim), max_lines)[1:-1]
        return wn

    def get_data(self):
        """
        Returns
        =======
        xi_dict : dict
        wn_dict : dict
        y_tp : np.ndarray
        x_ts : np.ndarray
        """
        np = import_module("numpy")

        if self.auto:
            xi = self._sgrid_default_xi(self.xlim, self.ylim)
            wn = self._sgrid_default_wn(self.xlim, self.ylim)
        else:
            xi = np.array([
                t.evalf(subs=self.params) if isinstance(t, Expr) else t
                for t in self.xi], dtype=float)
            if any(xi > 1) or any(xi < 0):
                # Enforce this condition
                raise ValueError("It must be ``0 <= xi <= 1. "
                    "Computed: %s" % xi)
            wn = np.array([
                t.evalf(subs=self.params) if isinstance(t, Expr) else t
                for t in self.wn], dtype=float)
        tp = np.array([
            t.evalf(subs=self.params) if isinstance(t, Expr) else t
            for t in self.tp], dtype=float)
        ts = np.array([
            t.evalf(subs=self.params) if isinstance(t, Expr) else t
            for t in self.ts], dtype=float)

        angles = np.pi - np.arccos(xi)
        y_over_x = np.tan(angles)
        r = max(1000, max(wn)) if len(wn) > 0 else 1000

        xi_dict = {k: {} for k in zip(xi, angles, y_over_x)}
        wn_dict = {k: {} for k in wn}
        tp_dict = {k: {} for k in tp}
        ts_dict = {k: {} for k in ts}

        # damping ratio lines
        for k in zip(xi, angles, y_over_x):
            x, a, yp = k
            xi_dict[k]["x"] = np.array([0, r * np.cos(a)])
            xi_dict[k]["y"] = np.array([0, r * np.sin(a)])
            xi_dict[k]["label"] = "%.2f" % x

        # natural frequency lines
        t = np.linspace(np.pi/2, 3*np.pi/2, 100)
        ct = np.cos(t)
        st = np.sin(t)
        ylim = self.ylim
        y_offset = 0 if ylim is None else 0.015 * abs(ylim[1] - ylim[0])
        for w in wn:
            wn_dict[w]["x"] = w * ct
            wn_dict[w]["y"] = w * st
            wn_dict[w]["label"] = "%.2f" % w
            wn_dict[w]["lx"] = -w
            wn_dict[w]["ly"] = y_offset

        # peak time lines
        y_tp = np.pi / tp
        # settling time lines
        x_ts = -4 / ts

        return xi_dict, wn_dict, y_tp, x_ts


class ZGridLineSeries(_NaturalFrequencyDampingRatioGrid, GridBase, BaseSeries):
    """Represent a grid of damping ratio lines and natural frequency lines
    on the z-plane.
    """
    sampling_period = param.Number(doc="""Sampling period.""")

    def __init__(self, xi, wn, tp, ts, **kwargs):
        super().__init__(xi=xi, wn=wn, tp=tp, ts=ts, **kwargs)

    def __str__(self):
        return "z-grid"

    def get_data(self):
        """
        Returns
        =======
        xi, wn, tp, ts : dict
            Dictionaries containing the required numerical data to create
            lines and annotations.
        """
        np = import_module("numpy")
        if self.is_interactive:
            xi = np.array([
                t.evalf(subs=self.params) if isinstance(t, Expr) else t
                for t in self.xi], dtype=float)
            wn = np.array([
                t.evalf(subs=self.params) if isinstance(t, Expr) else t
                for t in self.wn], dtype=float)
            tp = np.array([
                t.evalf(subs=self.params) if isinstance(t, Expr) else t
                for t in self.tp], dtype=float)
            ts = np.array([
                t.evalf(subs=self.params) if isinstance(t, Expr) else t
                for t in self.ts], dtype=float)
        else:
            xi = np.array(self.xi, dtype=float)
            wn = np.array(self.wn, dtype=float)
            tp = np.array(self.tp, dtype=float)
            ts = np.array(self.ts, dtype=float)

        T = self.sampling_period
        xi_dict = {k: {} for k in xi}
        wn_dict = {k: {} for k in wn}
        tp_dict = {k: {} for k in tp}
        ts_dict = {k: {} for k in ts}

        # damping ratio lines
        for zeta in xi:
            # Calculate in polar coordinates
            factor = zeta/np.sqrt(1-zeta**2)
            x = np.linspace(0, np.sqrt(1-zeta**2), 200)
            ang = np.pi*x
            mag = np.exp(-np.pi*factor*x)
            # Draw upper part in retangular coordinates
            xret = mag*np.cos(ang)
            yret = mag*np.sin(ang)
            xi_dict[zeta]["x1"] = xret
            xi_dict[zeta]["y1"] = yret
            # Draw lower part in retangular coordinates
            xret = mag*np.cos(-ang)
            yret = mag*np.sin(-ang)
            xi_dict[zeta]["x2"] = xret
            xi_dict[zeta]["y2"] = yret
            # Annotation
            an_i = int(len(xret)/2.5)
            an_x = xret[an_i]
            an_y = yret[an_i]
            xi_dict[zeta]["lx"] = xret[an_i]
            xi_dict[zeta]["ly"] = yret[an_i]
            xi_dict[zeta]["label"] = str(round(zeta, 2))

        # natural frequency lines
        r_an = 1.075
        fmt = '{:1.1f}' if len(wn) > 1 else '{:1.2f}'
        def get_label(num):
            def func(use_latex=True):
                if use_latex:
                    return r"$\frac{"+num+r"\pi}{T}$"
                return str(num) + " π/T"
            return func
        for a in wn:
            # Calculate in polar coordinates
            x = np.linspace(-np.pi/2, np.pi/2, 200)
            ang = np.pi*a*np.sin(x)
            mag = np.exp(-np.pi*a*np.cos(x))
            # Draw in retangular coordinates
            xret = mag*np.cos(ang)
            yret = mag*np.sin(ang)
            wn_dict[a]["x"] = xret
            wn_dict[a]["y"] = yret
            # Annotation
            angle = np.arctan2(yret[-1], xret[-1])
            wn_dict[a]["lx"] = r_an * np.cos(angle)
            wn_dict[a]["ly"] = r_an * np.sin(angle)
            if T is None:
                num = fmt.format(a)
                an = r"$\frac{"+num+r"\pi}{T}$"
                an = get_label(num)
            else:
                func = lambda a, T: lambda use_latex: "%.2f" % (a * np.pi * T)
                an = func(a, T)
            wn_dict[a]["label"] = an

        # peak time lines
        angles = np.pi / tp
        for _tp, a in zip(tp, angles):
            tp_dict[_tp]["x"] = [0, np.cos(a)]
            tp_dict[_tp]["y"] = [0, np.sin(a)]
            # Annotation
            tp_dict[_tp]["lx"] = r_an * np.cos(a)
            tp_dict[_tp]["ly"] = r_an * np.sin(a)
            an = _tp if not T else _tp * T
            an = "%.2f" % an if not T else "%.2f s" % an
            tp_dict[_tp]["label"] = an

        # settling time lines
        radius = np.exp(-4 / ts)
        theta = np.linspace(0, 2*np.pi, 400)
        ct = np.cos(theta)
        st = np.sin(theta)
        for _ts, r in zip(ts, radius):
            ts_dict[_ts]["x"] = r * ct
            ts_dict[_ts]["y"] = r * st
            # Annotation
            an_i = int(len(theta)*0.75)
            ts_dict[_ts]["lx"] = ts_dict[_ts]["x"][an_i]
            ts_dict[_ts]["ly"] = ts_dict[_ts]["y"][an_i]
            an = _ts if not T else _ts * T
            an = "%.2f" % an if not T else "%.2f s" % an
            ts_dict[_ts]["label"] = an

        return xi_dict, wn_dict, tp_dict, ts_dict


class ArrowsMixin(param.Parameterized):
    arrows = param.Parameter(default=3, doc="""
        Specify the number of arrows to plot on the Nyquist/Nichols curve.
        It can be:

        * an integer, representing the number of equally spaced arrows that
          will be plotted on each of the primary segment and the mirror image.
        * If a 1D array is passed, it should consist of a sorted list of
          floats between 0 and 1, indicating the location along the curve
          to plot an arrow.""")
    arrow_locs = param.Parameter(doc="""
        Location of the arrows along the curve.""")

    @param.depends("arrows", watch=True, on_init=True)
    def _set_arrow_locs(self):
        # Parse the arrows keyword
        np = import_module("numpy")
        arrow_locs = []

        if not self.arrows:
            self.arrow_locs = []
        elif isinstance(self.arrows, int):
            N = 3 if self.arrows is True else self.arrows
            # Space arrows out, starting midway along each "region"
            self.arrow_locs = np.linspace(0.5/N, 1 + 0.5/N, N, endpoint=False)
        elif isinstance(self.arrows, (list, np.ndarray)):
            self.arrow_locs = np.sort(np.atleast_1d(self.arrows))
        else:
            raise ValueError("unknown or unsupported arrow location")


class _TfParameter(param.Parameterized):
    _tf = param.Parameter(doc="""
        The LTI system, which can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form, which will be converted to
          an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.
        * a tuple of two or three elements: ``(num, den, generator [opt])``,
          which will be converted to an object of type
          :py:class:`sympy.physics.control.lti.TransferFunction`.""")

class NicholsLineSeries(
    ArrowsMixin,
    _TfParameter,
    _GridEvaluationParameters,
    Line2DBaseSeries
):
    """Represent a Nichols line in control system plotting.
    """
    _allowed_keys = ["arrows"]
    xscale = param.Selector(
        default="log", objects=["linear", "log"], doc="""
        Discretization strategy along the pulsation.""")

    def __init__(
        self, tf, ol_phase, ol_mag, cl_phase, cl_mag, omega_range,
        label="", **kwargs
    ):
        kwargs["force_real_eval"] = True
        kwargs["label"] = label
        kwargs["_tf"] = tf
        super().__init__(**kwargs)
        self.expr = Tuple(ol_phase, ol_mag, cl_phase, cl_mag)
        self.ranges = [omega_range]
        self.evaluator = GridEvaluator(series=self)

    def __str__(self):
        return self._str_helper("nichols line of %s" % self._tf)

    def get_data(self):
        """
        Returns
        =======
        omega : np.ndarray
        ol_phase : np.ndarray
        ol_mag : np.ndarray
        cl_phase : np.ndarray
        cl_mag : np.ndarray
        """
        np = import_module('numpy')

        results = self.evaluator._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        omega, ol_phase, ol_mag, cl_phase, cl_mag = results
        ol_mag = 20 * np.log10(ol_mag)
        ol_phase = np.degrees(unwrap(ol_phase))
        cl_mag = 20 * np.log10(cl_mag)
        # TODO: if the nichols line passes through the point (-180 deg, 0 dB)
        # (in open loop), then the resulting closed-loop phase is wrong. Why?
        # For example, test with this system:
        # tf = (5 * (s - 1)) / (s**2 * (s**2 + s + 4))
        cl_phase = np.degrees(unwrap(cl_phase))
        return omega, ol_phase, ol_mag, cl_phase, cl_mag


class ControlBaseSeries(Line2DBaseSeries):
    """A base series for classes that are going to produce numerical
    data using the ``control`` module for control-system plotting.
    Those series represent a SISO system.
    """

    _allowed_keys = ["control_kw"]

    expr = param.Parameter(doc="""
        Store a symbolic transfer function.""")
    _control_tf = param.Parameter(doc="""
        Store a transfer function from the control module.""")
    control_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to
        the function of the ``control`` module that will generate
        numerical data starting from the transfer function stored
        in ``_control_tf``.""")

    def __init__(self, *args, **kwargs):
        label = kwargs.pop("label", "")
        super().__init__(**kwargs)
        sp = import_module(
            "sympy.physics", import_kwargs={'fromlist':['control']})
        TransferFunction = sp.control.lti.TransferFunction
        np = import_module('numpy')
        sp = import_module('scipy')
        ct = import_module('control')
        tf = args[0]

        if isinstance(tf, (Expr, TransferFunction)):
            if isinstance(tf, Expr):
                params_fs = set(self.params.keys())
                fs = tf.free_symbols.difference(params_fs)
                fs = fs.pop() if len(fs) > 0 else symbols("s")
                tf = TransferFunction.from_rational_expression(tf, fs)
            self.expr = tf
            self._control_tf = None
            if not self.is_interactive:
                self._control_tf = tf_to_control(tf)
            self._label_str = str(self.expr) if label is None else label
            self._label_latex = latex(self.expr) if label is None else label
        elif isinstance(tf, (sp.signal.TransferFunction, ct.TransferFunction)):
            self.expr = None
            self._label_str = label
            self._label_latex = label
            if label is None:
                s = symbols("s" if tf.dt is None else "z")
                n = tf.num[0][0] if isinstance(ct.TransferFunction) else tf.num
                d = tf.den[0][0] if isinstance(ct.TransferFunction) else tf.den
                expr = Poly.from_list(n, s) / Poly.from_list(d, s)
                self._label_str = str(expr)
                self._label_latex = latex(expr)
            if isinstance(tf, sp.signal.TransferFunction):
                self._control_tf = tf_to_control(tf)
            else:
                self._control_tf = tf
        else:
            raise TypeError(
                "Transfer function's type not recognized. "
                "Received: " + str(type(tf))
            )

    def _check_fs(self):
        """ Checks if there are enogh parameters and free symbols.
        """
        fs = set()
        if self.expr:
            fs = {self.expr.var}
        ranges, params = self.ranges, self.params

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = fs.difference(params.keys())
        if ranges is not None:
            fs = fs.difference([r[0] for r in ranges])

        if len(fs) > 1:
            raise ValueError(
                "Incompatible expression and parameters.\n"
                + "Specify what these symbols represent: {}\n".format(fs)
                + "Are they ranges or parameters?"
            )

        # verify that all symbols are known (they either represent plotting
        # ranges or parameters)
        range_symbols = [r[0] for r in ranges]
        for r in ranges:
            fs = set().union(*[e.free_symbols for e in r[1:]])
            if any(t in fs for t in range_symbols):
                raise ValueError("Range symbols can't be included into "
                    "minimum and maximum of a range. "
                    "Received range: %s" % str(r))
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError(
                    "Unkown symbols found in plotting range: %s. " % (r,) +
                    "Are the following parameters? %s" % remaining_fs)


class NyquistLineSeries(ArrowsMixin, ControlBaseSeries):
    """Generates numerical data for Nyquist plot using the ``control``
    module.
    """

    _allowed_keys = [
        "arrows", "max_curve_magnitude", "max_curve_offset",
        "start_marker", "primary_style", "mirror_style"
    ]

    def _copy_from_dict(self, d, k):
        if k in d.keys():
            setattr(self, k, d[k])

    def __init__(self, tf, var_start_end, label="", **kwargs):
        super().__init__(tf, label=label, **kwargs)
        self.ranges = [var_start_end]
        self._check_fs()

        # these attributes are used by ``control`` in the rendering step,
        # not in the data generation step. I need them here in order to
        # control the rendering in each backend.
        self.max_curve_magnitude = kwargs.get("max_curve_magnitude", 20)
        self.max_curve_offset = kwargs.get("max_curve_offset", 0.02)
        self.start_marker = kwargs.get("start_marker", True)
        self.primary_style = kwargs.get("primary_style", None)
        self.mirror_style = kwargs.get("mirror_style", None)
        for k in ["arrows", "max_curve_magnitude", "max_curve_offset",
            "start_marker", "primary_style", "mirror_style"]:
            self._copy_from_dict(self.control_kw, k)

    def __str__(self):
        return self._str_helper(
            "nyquist line of %s" % (
                self.expr if self.expr else self._control_tf))

    def get_data(self):
        """
        Returns
        =======
        x_reg, y_reg : np.ndarray
        x_scl, y_scl : np.ndarray
        x_inv1, y_inv1 : np.ndarray
        x_inv2, y_inv2 : np.ndarray
        curve_offset : np.ndarray
        """
        np = import_module("numpy")
        ct = import_module("control")
        mergedeep = import_module('mergedeep')

        if self.is_interactive:
            tf = self.expr.subs(self.params)
            self._control_tf = tf_to_control(tf)

        control_kw = {}
        sym, start, end = self.ranges[0]
        if (start != end) or self._parametric_ranges:
            start = _update_range_value(self, start).real
            end = _update_range_value(self, end).real
            control_kw["omega_limits"] = [10**start, 10**end]

        ckw = mergedeep.merge({}, control_kw, self.control_kw)
        response_obj = ct.nyquist_response(self._control_tf, **ckw)
        resp = response_obj.response
        if response_obj.dt in [0, None]:
            splane_contour = response_obj.contour
        else:
            splane_contour = np.log(response_obj.contour) / response_obj.dt

        max_curve_magnitude = self.max_curve_magnitude
        max_curve_offset = self.max_curve_offset

        reg_mask = np.logical_or(
            np.abs(resp) > max_curve_magnitude,
            splane_contour.real != 0)

        scale_mask = ~reg_mask \
            & np.concatenate((~reg_mask[1:], ~reg_mask[-1:])) \
            & np.concatenate((~reg_mask[0:1], ~reg_mask[:-1]))

        # Rescale the points with large magnitude
        rescale = np.logical_and(
            reg_mask, abs(resp) > max_curve_magnitude)
        resp[rescale] *= max_curve_magnitude / abs(resp[rescale])

        # Plot the regular portions of the curve (and grab the color)
        x_reg = np.ma.masked_where(reg_mask, resp.real)
        y_reg = np.ma.masked_where(reg_mask, resp.imag)

        # Figure out how much to offset the curve: the offset goes from
        # zero at the start of the scaled section to max_curve_offset as
        # we move along the curve
        curve_offset = self._compute_curve_offset(
            resp, scale_mask, max_curve_offset)

        # Plot the scaled sections of the curve (changing linestyle)
        x_scl = np.ma.masked_where(scale_mask, resp.real)
        y_scl = np.ma.masked_where(scale_mask, resp.imag)

        # the primary curve (invisible) for setting arrows
        x_inv1, y_inv1 = resp.real.copy(), resp.imag.copy()
        x_inv1[reg_mask] *= (1 + curve_offset[reg_mask])
        y_inv1[reg_mask] *= (1 + curve_offset[reg_mask])

        # Add the arrows to the mirror image (on top of an invisible contour)
        x_inv2, y_inv2 = resp.real.copy(), resp.imag.copy()
        x_inv2[reg_mask] *= (1 - curve_offset[reg_mask])
        y_inv2[reg_mask] *= (1 - curve_offset[reg_mask])

        return x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset

    @staticmethod
    def _compute_curve_offset(resp, mask, max_offset):
        """
            Function to compute Nyquist curve offsets

        This function computes a smoothly varying offset that starts and ends at
        zero at the ends of a scaled segment.

        This function comes from ``control/freqplot.py``.
        """
        np = import_module("numpy")

        # Compute the arc length along the curve
        s_curve = np.cumsum(
            np.sqrt(np.diff(resp.real) ** 2 + np.diff(resp.imag) ** 2))

        # Initialize the offset
        offset = np.zeros(resp.size)
        arclen = np.zeros(resp.size)

        # Walk through the response and keep track of each continous component
        i, nsegs = 0, 0
        while i < resp.size:
            # Skip the regular segment
            while i < resp.size and mask[i]:
                i += 1              # Increment the counter
                if i == resp.size:
                    break
                # Keep track of the arclength
                arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

            nsegs += 0.5
            if i == resp.size:
                break

            # Save the starting offset of this segment
            seg_start = i

            # Walk through the scaled segment
            while i < resp.size and not mask[i]:
                i += 1
                if i == resp.size:  # See if we are done with this segment
                    break
                # Keep track of the arclength
                arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

            nsegs += 0.5
            if i == resp.size:
                break

            # Save the ending offset of this segment
            seg_end = i

            # Now compute the scaling for this segment
            s_segment = arclen[seg_end-1] - arclen[seg_start]
            offset[seg_start:seg_end] = max_offset * s_segment/s_curve[-1] * \
                np.sin(np.pi * (arclen[seg_start:seg_end]
                                - arclen[seg_start])/s_segment)

        return offset


class RootLocusSeries(ControlBaseSeries):
    """
    Generates numerical data for root locus plot using the ``control``
    module.

    Symbolic expressions or SymPy's transfer functions are converted to
    ``control.TransferFunction``. If a interactive-widget plot is created,
    at each widget's state-change the updated symbolic transfer function
    will be converted to ``control.TransferFunction``.

    It has been shown that numpy.roots() produces inaccurate results in
    comparison to sympy.roots(). https://github.com/sympy/sympy/issues/25234
    However, we are dealing with a root locus plot, where branches start from
    poles and goes to zeros (or to infinity). Hence, these errors are
    likely to be irrelevant on a practical case. This data series uses
    ``control`` (hence numpy) for performace.

    References
    ==========

    https://github.com/python-control/python-control

    """

    def __init__(self, tf, label="", **kwargs):
        kwargs["label"] = label
        super().__init__(tf, **kwargs)
        self._check_fs()

        # compute appropriate axis limits from the transfer function
        # associated to this data series.
        self._xlim = None
        self._ylim = None
        # zeros and poles are necessary in order to show appropriate markers.
        self._zeros = None
        self._poles = None

        self._zeros_rk = kwargs.get("zeros_rk", dict())
        self._poles_rk = kwargs.get("poles_rk", dict())

    def __str__(self):
        expr = self.expr if self.expr else self._control_tf
        return "root locus of " + str(expr)

    @property
    def zeros(self):
        if self._zeros is None:
            self.get_data()
        return self._zeros

    @property
    def poles(self):
        if self._poles is None:
            self.get_data()
        return self._poles

    @property
    def xlim(self):
        return self._xlim

    @property
    def ylim(self):
        return self._ylim

    def get_data(self):
        """
        Returns
        =======
        roots : ndarray
            Closed-loop root locations, arranged in which each row corresponds
            to a gain in gains
        gains : ndarray
            Gains used.  Same as kvect keyword argument if provided.
        """
        ct = import_module("control")
        # TODO: open PR on control and implement a method inside the object
        # returned by root_locus_map, so that we don't have to deal with
        # private methods.
        from control.pzmap import _compute_root_locus_limits

        if self.is_interactive:
            tf = self.expr.subs(self.params)
            self._control_tf = tf_to_control(tf)

        self._zeros = self._control_tf.zeros()
        self._poles = self._control_tf.poles()
        data = ct.root_locus_map(
            self._control_tf, **self.control_kw)
        self._xlim, self._ylim = _compute_root_locus_limits(data)
        return data.loci, data.gains



class SystemResponseSeries(ControlBaseSeries):
    """Represent a system response computed with the ``control`` module.

    Computing the inverse laplace transform of a system with SymPy is not
    trivial: sometimes it works fine, other times it produces wrong results,
    other times it just consumes to much memory even for trivial transfer
    functions. This is true for both the public ``inverse_laplace_transform``
    as well as the private ``_fast_inverse_laplace`` used in
    ``spb.graphics.control``.

    In order to address these issues, let's evaluate the system with the
    ``control`` module. Sure, it relies on numerical integration, hence errors.
    But, at least it doesn't crash the machine and it is reliable.
    """

    only_integers = param.Boolean(False, doc="""
        Discretize the domain using only integer numbers.""")
    response_type = param.Selector(
        default="step", objects=["impulse", "step", "ramp"], doc="""
        The type of response to simulate.""")
    n = param.List([100, 100, 100], item_type=Number, bounds=(3, 3), doc="""
        Number of discretization points along the x, y, z directions,
        respectively. It can easily be set with ``n=number``, which will
        set ``number`` for each element of the list.
        For surface, contour, 2d vector field plots it can be set with
        ``n=[num1, num2]``. For 3D implicit plots it can be set with
        ``n=[num1, num2, num3]``.

        Alternatively, ``n1=num1, n2=num2, n3=num3`` can be indipendently
        set in order to modify the respective element of the ``n`` list.""")

    def __new__(cls, *args, **kwargs):
        cf = kwargs.get("color_func", None)
        lc = kwargs.get("line_color", None)
        if (callable(cf) or callable(lc)):
            return super().__new__(ColoredSystemResponseSeries)
        return object.__new__(cls)

    def __init__(self, tf, var_start_end, label="", **kwargs):
        super().__init__(tf, label=label, **kwargs)
        self.ranges = [var_start_end]
        self._check_fs()

        if self.expr is None:
            self.steps = self._control_tf.isdtime()

        # time values over which the evaluation will be performed
        self._time_array = None

    def __str__(self):
        return self._str_helper(
            "%s response of %s" % (
                self.response_type,
                self.expr if self.expr else self._control_tf))

    def _get_data_helper(self):
        ct = import_module("control")
        np = import_module("numpy")
        mergedeep = import_module('mergedeep')

        if self.is_interactive:
            tf = self.expr.subs(self.params)
            self._control_tf = tf_to_control(tf)

        # create (or update) the discretized domain
        _, start, end = self.ranges[0]
        if self._parametric_ranges:
            start = _update_range_value(self, start).real
            end = _update_range_value(self, end).real
        else:
            start, end = float(start), float(end)

        if (self._time_array is None) or self._parametric_ranges:
            if not self._control_tf.isdtime():
                n = self.n[0]
            else:
                n = int((end - start) / self._control_tf.dt) + 1
                end = (n - 1) * self._control_tf.dt
            self._time_array = _discretize(
                    start, end, n, self.scales[0], self.only_integers)

        control_kw = {"T": self._time_array, "squeeze": True}

        if self.response_type == "step":
            ckw = mergedeep.merge({}, control_kw, self.control_kw)
            response = ct.step_response(self._control_tf, **ckw)
        elif self.response_type == "impulse":
            ckw = mergedeep.merge({}, control_kw, self.control_kw)
            response = ct.impulse_response(self._control_tf, **ckw)
        elif self.response_type == "ramp":
            ramp = self._time_array
            control_kw["U"] = ramp
            ckw = mergedeep.merge({}, control_kw, self.control_kw)
            response = ct.forced_response(self._control_tf, **ckw)
        else:
            raise NotImplementedError

        return response.time, response.y.flatten()


class ColoredSystemResponseSeries(SystemResponseSeries):
    """
    Represent a system response computed with the ``control`` module,
    and colored according some color function.
    """
    is_parametric = True

    color_func = param.Callable(doc="""
        A numerical function of 2 variables, x, y (the points computed
        by the internal algorithm) supporting vectorization, returning
        the color value.""")

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_cm", True)
        super().__init__(*args, **kwargs)

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        x, y, p = args
        return t(x, self.tx), t(y, self.ty), p

    def _get_data_helper(self):
        x, y = super()._get_data_helper()
        return x, y, self.color_func(x, y)


class PoleZeroCommon(param.Parameterized):
    return_poles = param.Boolean(True, doc="""
        If True returns the poles of the transfer function, otherwise
        it returns the zeros.""")
    pole_color = param.ClassSelector(class_=(str, list, tuple), doc="""
        The color of the pole points on the plot.""")
    zero_color = param.ClassSelector(class_=(str, list, tuple), doc="""
        The color of the zero points on the plot.""")
    pole_markersize = param.Integer(10, bounds=(1, None), doc="""
        The size of the markers used to mark the poles in the plot.""")
    zero_markersize = param.Integer(10, bounds=(1, None), doc="""
        The size of the markers used to mark the zeros in the plot.""")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_point = True

    def __str__(self):
        return self._str_helper("poles" if self.return_poles else "zeros ")


class PoleZeroWithSympySeries(PoleZeroCommon, List2DSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PoleZeroSeries(PoleZeroCommon, ControlBaseSeries):
    """Represent a the pole-zero of an LTI SISO system computed
    with the ``control`` module.

    This series represents either poles or zeros, not both at the same time.
    In some sense, it behaves like a List2DSeries. So, to represents both
    poles and zeros of a transfer function, we need to instatiate two
    different series passing in the same transfer function.

    While computationally less efficient, this design choice have been made
    in order to reuse the existing BaseBackend architecture, that sets up
    the number of colors based on the number of data series, as well as the
    logic to show or hide the legend.
    """
    def __init__(self, tf, label="", **kwargs):
        super().__init__(tf, label=label, **kwargs)
        self._check_fs()

    def __str__(self):
        pre = "poles of " if self.return_poles else "zeros of "
        expr = self.expr if self.expr is not None else self._control_tf
        return pre + str(expr)

    def _get_data_helper(self):
        """
        Returns
        =======
        x : np.ndarray
        y : np.ndarray
        """
        np = import_module("numpy")
        if self.is_interactive:
            tf = self.expr.subs(self.params)
            self._control_tf = tf_to_control(tf)
        if self.return_poles:
            points = self._control_tf.poles()
        else:
            points = self._control_tf.zeros()
        return np.real(points), np.imag(points)


class NGridLineSeries(GridBase, BaseSeries):
    """ The code of this class comes from the ``control`` package, which has
    been rearranged to work with the architecture of this module.
    """

    show_cl_mags = param.Boolean(default=True, doc="""
        Toggle the visibility of the closed-loop magnitude grid lines.""")
    show_cl_phases = param.Boolean(default=True, doc="""
        Toggle the visibility of the closed-loop phase grid lines.""")
    label_cl_phases = param.Boolean(default=True, doc="""
        Toggle the visibility of the labels about the closed-loop phase.""")

    def __init__(self, cl_mags=None, cl_phases=None, label_cl_phases=False,
        **kwargs):
        kwargs.setdefault("show_in_legend", False)
        kwargs["label_cl_phases"] = label_cl_phases
        super().__init__(**kwargs)
        np = import_module("numpy")
        self.cl_mags = cl_mags if cl_mags is None else np.array(cl_mags)
        self.cl_phases = cl_phases if cl_phases is None else np.array(cl_phases)

    def __str__(self):
        return "n-grid"

    @staticmethod
    def closed_loop_contours(Gcl_mags, Gcl_phases):
        """Contours of the function Gcl = Gol/(1+Gol), where
        Gol is an open-loop transfer function, and Gcl is a corresponding
        closed-loop transfer function.

        Parameters
        ----------
        Gcl_mags : array-like
            Array of magnitudes of the contours
        Gcl_phases : array-like
            Array of phases in radians of the contours

        Returns
        -------
        contours : complex array
            Array of complex numbers corresponding to the contours.
        """
        # Compute the contours in Gcl-space. Since we're given closed-loop
        # magnitudes and phases, this is just a case of converting them into
        # a complex number.
        np = import_module("numpy")
        Gcl = Gcl_mags*np.exp(1.j*Gcl_phases)

        # Invert Gcl = Gol/(1+Gol) to map the contours into the open-loop space
        return Gcl/(1.0 - Gcl)

    @staticmethod
    def m_circles(mags, phase_min=-359.75, phase_max=-0.25):
        """Constant-magnitude contours of the function Gcl = Gol/(1+Gol), where
        Gol is an open-loop transfer function, and Gcl is a corresponding
        closed-loop transfer function.

        Parameters
        ----------
        mags : array-like
            Array of magnitudes in dB of the M-circles
        phase_min : degrees
            Minimum phase in degrees of the N-circles
        phase_max : degrees
            Maximum phase in degrees of the N-circles

        Returns
        -------
        contours : complex array
            Array of complex numbers corresponding to the contours.
        """
        # Convert magnitudes and phase range into a grid suitable for
        # building contours
        np = import_module("numpy")
        phases = np.radians(np.linspace(phase_min, phase_max, 2000))
        Gcl_mags, Gcl_phases = np.meshgrid(10.0**(mags/20.0), phases)
        return NGridLineSeries.closed_loop_contours(Gcl_mags, Gcl_phases)

    @staticmethod
    def n_circles(phases, mag_min=-40.0, mag_max=12.0):
        """Constant-phase contours of the function Gcl = Gol/(1+Gol), where
        Gol is an open-loop transfer function, and Gcl is a corresponding
        closed-loop transfer function.

        Parameters
        ----------
        phases : array-like
            Array of phases in degrees of the N-circles
        mag_min : dB
            Minimum magnitude in dB of the N-circles
        mag_max : dB
            Maximum magnitude in dB of the N-circles

        Returns
        -------
        contours : complex array
            Array of complex numbers corresponding to the contours.
        """
        # Convert phases and magnitude range into a grid suitable for
        # building contours
        np = import_module("numpy")
        mags = np.linspace(10**(mag_min/20.0), 10**(mag_max/20.0), 2000)
        Gcl_phases, Gcl_mags = np.meshgrid(np.radians(phases), mags)
        return NGridLineSeries.closed_loop_contours(Gcl_mags, Gcl_phases)

    def get_data(self):
        np = import_module("numpy")

        # Default chart size
        ol_phase_min = -359.99
        ol_phase_max = 0.0
        ol_mag_min = -40.0
        ol_mag_max = default_ol_mag_max = 50.0

        cl_mags = self.cl_mags
        cl_phases = self.cl_phases
        label_cl_phases = self.label_cl_phases

        # Find extent of intersection the current dataset or view
        ol_phase_min, ol_phase_max = self.xlim
        ol_mag_min, ol_mag_max = self.ylim

        # M-circle magnitudes.
        if cl_mags is None:
            # Default chart magnitudes
            # The key set of magnitudes are always generated, since this
            # guarantees a recognizable Nichols chart grid.
            key_cl_mags = np.array([
                -40.0, -20.0, -12.0, -6.0, -3.0, -1.0, -0.5,
                0.0, 0.25, 0.5, 1.0, 3.0, 6.0, 12.0
            ])

            # Extend the range of magnitudes if necessary. The extended arange
            # will end up empty if no extension is required. Assumes that
            # closed-loop magnitudes are approximately aligned with open-loop
            # magnitudes beyond the value of np.min(key_cl_mags)
            cl_mag_step = -20.0  # dB
            extended_cl_mags = np.arange(
                np.min(key_cl_mags), ol_mag_min + cl_mag_step, cl_mag_step)
            cl_mags = np.concatenate((extended_cl_mags, key_cl_mags))

        # a minimum 360deg extent containing the phases
        phase_round_max = 360.0*np.ceil(ol_phase_max/360.0)
        phase_round_min = min(phase_round_max-360,
                            360.0*np.floor(ol_phase_min/360.0))

        # N-circle phases (should be in the range -360 to 0)
        if cl_phases is None:
            # aim for 9 lines, but always show (-360+eps, -180, -eps)
            # smallest spacing is 45, biggest is 180
            phase_span = phase_round_max - phase_round_min
            spacing = np.clip(round(phase_span / 8 / 45) * 45, 45, 180)
            key_cl_phases = np.array([-0.25, -359.75])
            other_cl_phases = np.arange(-spacing, -360.0, -spacing)
            cl_phases = np.unique(np.concatenate((key_cl_phases, other_cl_phases)))
        elif not ((-360 < np.min(cl_phases)) and (np.max(cl_phases) < 0.0)):
            raise ValueError('cl_phases must between -360 and 0, exclusive')

        self.cl_mags = cl_mags
        self.cl_phases = cl_phases

        # Find the M-contours
        m = self.m_circles(
            cl_mags, phase_min=np.min(cl_phases), phase_max=np.max(cl_phases))
        m_mag = 20*np.log10(np.abs(m))
        m_phase = np.mod(np.degrees(np.angle(m)), -360.0)  # Unwrap

        # Find the N-contours
        n = self.n_circles(cl_phases, mag_min=np.min(cl_mags), mag_max=np.max(cl_mags))
        n_mag = 20*np.log10(np.abs(n))
        n_phase = np.mod(np.degrees(np.angle(n)), -360.0)  # Unwrap

        # Plot the contours behind other plot elements.
        # The "phase offset" is used to produce copies of the chart that cover
        # the entire range of the plotted data, starting from a base chart computed
        # over the range -360 < phase < 0. Given the range
        # the base chart is computed over, the phase offset should be 0
        # for -360 < ol_phase_min < 0.
        phase_offsets = 360 + np.arange(phase_round_min, phase_round_max, 360.0)
        return m_mag, m_phase, n_mag, n_phase, phase_offsets


class MCirclesSeries(GridBase, BaseSeries):
    def __init__(self, magnitudes_db, magnitudes, **kwargs):
        super().__init__(**kwargs)
        self.magnitudes_db = Tuple(*magnitudes_db)
        self.magnitudes = self.expr = Tuple(*magnitudes)
        self.show_minus_one = kwargs.get("show_minus_one", False)

    def get_data(self):
        """
        Returns
        =======

        data : list
            Each element of the list has the form:
            ``[magnitude_db, x_coords, y_coords]``.
        """
        np = import_module("numpy")
        data = []
        magnitudes = self.magnitudes
        magnitudes_db = self.magnitudes_db
        if self.is_interactive:
            magnitudes = magnitudes.subs(self.params)
            magnitudes_db = magnitudes_db.subs(self.params)
        magnitudes = np.array(magnitudes, dtype=float)
        magnitudes_db = np.array(magnitudes_db, dtype=float)

        theta = np.linspace(0, 2*np.pi, 400)
        ct = np.cos(theta)
        st = np.sin(theta)
        for mdb, m in zip(magnitudes_db, magnitudes):
            if not np.isclose(mdb, 0):
                r = m / (1 - m**2)
                x = m**2 / (1 - m**2) + r * ct
                y = r * st
            else:
                x = [-0.5]
                y = [0]
            data.append([mdb, x, y])
        return data
