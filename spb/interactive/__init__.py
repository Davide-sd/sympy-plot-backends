from collections import Counter
import math
from spb.defaults import cfg
from spb.utils import is_number
from sympy import latex
from sympy.external import import_module


def _tuple_to_dict(k, v, use_latex=False, latex_wrapper="$%s$"):
    """Create a dictionary of keyword arguments to be later used to
    instantiate sliders.

    Parameters
    ==========

    k : Symbol
        Symbolic parameter

    v : tuple/list
        A variable length tuple/list containing:

        (default, min, max, N [optional], label [optional], spacing [optional])

        where:
            default : float
                Default value of the slider
            min : float
                Minimum value of the slider.
            max : float
                Maximum value of the slider.
            N : int
                Number of increments in the slider.
                (start - end) / N represents the step increment. Default to 40.
                Set N=-1 to have unit step increments.
            label : str
                Label of the slider. Default to None. If None, the string or
                latex representation will be used. See use_latex for more
                information.
            spacing : str
                Discretization spacing. Can be "linear" or "log".
                Default to "linear".
    """
    bokeh = import_module("bokeh", import_kwargs={'fromlist': ['models']})
    TickFormatter = bokeh.models.formatters.TickFormatter if bokeh else None

    if not hasattr(v, "__iter__"):
        raise TypeError(
            "Provide a tuple or list for the parameter {}".format(k))
    if len(v) < 3:
        raise ValueError(
            "The parameter-tuple must have at least 3 elements: "
            "value, min, max. Received: %s" % list(v))
    if any(not is_number(t) for t in v[:3]):
        raise TypeError(
            "The first three elements of the parameter-tuple must be numbers. "
            "Received: %s" % list(v[:3]))

    defaults_keys = [
        "value", "min", "max", "step", "formatter", "description", "type"]
    defaults_values = [1, 0, 2, 0.1, None,
        latex_wrapper % latex(k) if use_latex else str(k),
        "linear",
    ]

    # find min, max, value from what the user provided.
    n1, n2, n3 = [float(t) for t in v[:3]]
    _min = min(n1, n2, n3)
    _max = max(n1, n2, n3)
    if math.isclose(_min, _max):
        raise ValueError(
            "The minimum value of the slider must be different from the "
            "maximum value.")
    c = Counter([n1, n2, n3])
    if c[_min] == 2:
        _val = _min
    elif c[_max] == 2:
        _val = _max
    else:
        _val = set([n1, n2, n3]).difference([_min, _max]).pop()

    N, formatter, label, spacing = None, None, None, None
    for a in v[3:]:
        if is_number(a):
            N = int(a)
        elif TickFormatter and isinstance(a, TickFormatter):
            formatter = a
        elif isinstance(a, str):
            b = a.lower()
            if b in ["linear", "log"]:
                spacing = b
            elif b[0] in [".", "%"]:
                formatter = a
            else:
                label = a

    N = N if N else 40
    step = (_max - _min) / N

    values = defaults_values.copy()
    values[:4] = [_val, _min, _max, step]
    if formatter:
        values[4] = formatter
    if label:
        values[5] = label
    if spacing:
        values[6] = spacing

    return {k: v for k, v in zip(defaults_keys, values)}


def create_interactive_plot(*series, **kwargs):
    """Select which interactive module to use.
    """
    imodule = kwargs.pop("imodule", cfg["interactive"]["module"])
    if imodule is None:
        imodule = cfg["interactive"]["module"]
    imodule = imodule.lower()

    animation = kwargs.get("animation", False)

    # NOTE: Holoviz's Panel is really slow to load, so let's load it only when
    # it is necessary
    if not animation:
        if imodule == "panel":
            from spb.interactive.panel import iplot
            return iplot(*series, **kwargs)
        elif imodule == "ipywidgets":
            from spb.interactive.ipywidgets import iplot
            return iplot(*series, **kwargs)
    else:
        if imodule == "panel":
            from spb.animation.panel import animation
            return animation(*series, **kwargs)
        elif imodule == "ipywidgets":
            from spb.animation.ipywidgets import animation
            return animation(*series, **kwargs)

    raise ValueError("`%s` is not a valid interactive module" % imodule)


class IPlot:
    """Mixin class for interactive plots containing common attributes and
    methods.
    """

    @property
    def fig(self):
        """Return the plot object"""
        return self._backend.fig

    @property
    def backend(self):
        """Return the backend"""
        return self._backend

    def save(self, *args, **kwargs):
        """Save the current figure.
        This is a wrapper to the backend's `save` function. Refer to the
        backend's documentation to learn more about arguments and keyword
        arguments.
        """
        self._backend.save(*args, **kwargs)

    def __add__(self, other):
        return self._do_sum(other)

    def __radd__(self, other):
        return other._do_sum(self)

    def _do_sum(self, other):
        """Differently from Plot.extend, this method creates a new plot object,
        which uses the series of both plots and merges the _kwargs dictionary
        of `self` with the one of `other`.
        """
        from spb.backends.base_backend import Plot
        mergedeep = import_module('mergedeep')
        merge = mergedeep.merge

        if not isinstance(other, (Plot, IPlot)):
            raise TypeError(
                "Both sides of the `+` operator must be instances of the "
                "InteractivePlot or Plot class.\n"
                "Received: {} + {}".format(type(self), type(other)))

        series = self._backend.series
        if isinstance(other, Plot):
            series.extend(other.series)
        else:
            series.extend(other._backend.series)

        # check that the interactive series uses the same parameters
        symbols = []
        for s in series:
            if s.is_interactive:
                symbols.append(list(s.params.keys()))
        if not all(t == symbols[0] for t in symbols):
            raise ValueError(
                "The same parameters must be used when summing up multiple "
                "interactive plots.")

        backend_kw = self._backend._copy_kwargs()
        iplot_kw = self._get_iplot_kw()
        iplot_kw["show"] = False

        new_iplot = type(self)(*series, **merge({}, backend_kw, iplot_kw))
        return new_iplot
