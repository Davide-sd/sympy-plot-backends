from spb.defaults import cfg
from sympy import latex

def _tuple_to_dict(k, v, use_latex=False):
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
    if not hasattr(v, "__iter__"):
        raise TypeError(
            "Provide a tuple or list for the parameter {}".format(k))

    N = 40
    defaults_keys = ["value", "min", "max", "step", "description", "type"]
    defaults_values = [1, 0, 2, N,
        "$%s$" % latex(k) if use_latex else str(k),
        "linear",
    ]
    values = defaults_values.copy()
    values[: len(v)] = v
    values[:3] = [float(t) for t in values[:3]]
    # set the step increment for the slider
    _min, _max = float(values[1]), float(values[2])
    if values[3] > 0:
        N = int(values[3])
        values[3] = (_max - _min) / N
    else:
        values[3] = 1

    return {k: v for k, v in zip(defaults_keys, values)}


def create_interactive_plot(*series, **kwargs):
    """Select which interactive module to use.
    """
    imodule = kwargs.pop("imodule", cfg["interactive"]["module"])
    imodule = imodule.lower()

    if imodule == "panel":
        # NOTE: Holoviz's Panel is really slow to load, so let's load it only when
        # it is necessary
        from spb.interactive.panel import iplot
        return iplot(*series, **kwargs)
    elif imodule == "ipywidgets":
        from spb.interactive.ipywidgets import iplot
        return iplot(*series, **kwargs)

    raise ValueError("`%s` is not a valid interactive module" % imodule)
