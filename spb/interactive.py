import numpy as np
import param
import panel as pn
from sympy import latex, Tuple
from spb.series import (
    InteractiveSeries,
    _set_discretization_points
)
from spb.ccomplex.complex import _build_series as _build_complex_series
from spb.vectors import _preprocess, _build_series as _build_vector_series
from spb.utils import _plot_sympify, _unpack_args
from spb.defaults import TWO_D_B, THREE_D_B
import warnings

pn.extension("plotly")


class MyList(param.ObjectSelector):
    """Represent a list of numbers discretizing a log-spaced slider.
    This parameter will be rendered by pn.widgets.DiscreteSlider
    """

    pass


# explicitely ask panel to use DiscreteSlider when it encounters a MyList object
pn.Param._mapping[MyList] = pn.widgets.DiscreteSlider


class DynamicParam(param.Parameterized):
    """Dynamically add parameters based on the user-provided dictionary.
    Also, generate the lambda functions to be evaluated at a later stage.
    """

    # NOTE: why DynamicParam is a child class of param.Parameterized?
    # param is a full-python library, doesn't depend on anything else.
    # In theory, by using a parameterized class it should be possible to create
    # an InteractivePlotGUI class targeting a specific GUI.
    # At this moment, InteractivePlot is built on top of 'panel', so it only
    # works inside a Jupyter Notebook. Maybe it's possible to use PyQt or Tk...

    def _tuple_to_dict(self, k, v):
        """The user can provide a variable length tuple/list containing:
            (default, softbounds, N, label, spacing)
        where:
            default : float
                Default value of the slider
            softbounds : tuple
                Tuple of two float (or integer) numbers: (start, end).
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
        N = 40
        defaults_keys = ["default", "softbounds", "step", "label", "type"]
        defaults_values = [
            1,
            (0, 2),
            N,
            "$%s$" % latex(k) if self.use_latex else str(k),
            "linear",
        ]
        values = defaults_values.copy()
        values[: len(v)] = v
        # set the step increment for the slider
        _min, _max = values[1][0], values[1][1]
        if values[2] > 0:
            N = values[2]
            values[2] = (_max - _min) / N
        else:
            values[2] = 1

        if values[-1] == "log":
            # In case of a logarithm slider, we need to instantiate the custom
            # parameter MyList.

            # # divide the range in N steps evenly spaced in a log scale
            options = np.geomspace(_min, _max, N)
            # the provided default value may not be in the computed options.
            # If that's the case, I chose the closest value
            default = values[0]
            if default not in options:
                default = min(options, key=lambda x: abs(x - default))
            return MyList(default=default, objects=list(options), label=values[3])

        return {k: v for k, v in zip(defaults_keys, values)}

    def __init__(self, *args, name="", params=None, **kwargs):
        # remove the previous class attributes added by the previous instances
        cls_name = type(self).__name__
        setattr(type(self), "_" + cls_name + "__params", dict())
        prev_params = [k for k in type(self).__dict__.keys() if "dyn_param_" in k]
        for p in prev_params:
            delattr(type(self), p)

        # use latex on control labels and legends
        self.use_latex = kwargs.pop("use_latex", True)

        # this must be present in order to assure correct behaviour
        super().__init__(name=name, **kwargs)
        if not params:
            raise ValueError("`params` must be provided.")

        # The following dictionary will be used to create the appropriate
        # lambda function arguments:
        #    key: the provided symbol
        #    val: name of the associated parameter
        self.mapping = {}

        # create and attach the params to the class
        for i, (k, v) in enumerate(params.items()):
            if not isinstance(v, param.parameterized.Parameter):
                v = self._tuple_to_dict(k, v)
                # at this stage, v could be a dictionary representing a number,
                # or a MyList parameter, representing a log slider
                if not isinstance(v, param.parameterized.Parameter):
                    v.pop("type", None)
                    v = param.Number(**v)

            # TODO: using a private method: not the smartest thing to do
            self.param._add_parameter("dyn_param_{}".format(i), v)
            self.mapping[k] = "dyn_param_{}".format(i)

    def read_parameters(self):
        readout = dict()
        for k, v in self.mapping.items():
            readout[k] = getattr(self, v)
        return readout


def _new_class(cls, **kwargs):
    "Creates a new class which overrides parameter defaults."
    return type(type(cls).__name__, (cls,), kwargs)


class PanelLayout:
    """Mixin class to group together the layout functionalities related to
    the library panel.
    """

    def __init__(self, layout, ncols):
        """
        Parameters
        ==========
            layout : str
                The layout for the controls/plot. Possible values:
                    'tb': controls in the top bar.
                    'bb': controls in the bottom bar.
                    'sbl': controls in the left side bar.
                    'sbr': controls in the right side bar.
                Default layout to 'tb'.

            ncols : int
                Number of columns to lay out the widgets. Default to 2.
        """
        # NOTE: More often than not, the numerical evaluation is going to be
        # resource-intensive. By default, panel's sliders will force a recompute
        # at every step change. As a consequence, the user experience will be
        # laggy. To solve this problem, the update must be triggered on mouse-up
        # event, which is set using throttled=True.
        #
        # https://panel.holoviz.org/reference/panes/Param.html#disabling-continuous-updates-for-slider-widgets
        #
        # The procedure is not optimal, as we need to re-map again the
        # parameters with the widgets: for param.Number, param.Integer there is
        # no one-on-one mapping. For example, a bounded param.Integer
        # will create an IntegerSlider, whereas an unbounded param.Integer will
        # create a IntInput.

        layouts = ["tb", "bb", "sbl", "sbr"]
        layout = layout.lower()
        if layout not in layouts:
            warnings.warn(
                "`layout` must be one of the following: {}\n".format(layouts)
                + "Falling back to layout='tb'."
            )
            layout = "tb"
        self._layout = layout

        widgets = {}
        for k, v in self.mapping.items():
            t = getattr(self.param, v)
            widget = ""
            if isinstance(t, param.Integer):
                widget = pn.widgets.IntSlider
                if t.bounds and any([(b is None) for b in t.bounds]):
                    widget = pn.widgets.IntInput
            elif isinstance(t, param.Number):
                widget = pn.widgets.FloatSlider
                if t.bounds and any([(b is None) for b in t.bounds]):
                    widget = pn.widgets.FloatInput
            elif isinstance(t, MyList):
                # TODO: it seems like DiscreteSlider doesn't support throttling
                widget = pn.widgets.DiscreteSlider

            if isinstance(t, param.Number):
                widgets[v] = {
                    "type": widget,
                    "throttled": True,
                }

        self.controls = pn.Param(
            self,
            widgets=widgets,
            default_layout=_new_class(pn.GridBox, ncols=ncols),
            show_name=False,
            sizing_mode="stretch_width",
        )

    def layout_controls(self):
        return self.controls

    @pn.depends("controls")
    def view(self):
        params = self.read_parameters()
        self._backend._update_interactive(params)
        # TODO:
        # 1. for some reason, panel is going to set width=0 if K3D-Jupyter.
        # Temporary workaround: create a Pane with a default width.
        # Long term solution: create a PR on panel to create a K3DPane so that
        # panel will automatically deal with K3D, in the same way it does with
        # Bokeh, Plotly, Matplotlib, ...
        # 2. If the following import statement was located at the beginning of
        # the file, there would be a circular import.
        from spb.backends.k3d import KB

        if isinstance(self._backend, KB):
            return pn.pane.Pane(self._backend.fig, width=800)
        else:
            return self.fig

    def show(self):
        if self._layout == "tb":
            return pn.Column(self.layout_controls, self.view)
        elif self._layout == "bb":
            return pn.Column(self.view, self.layout_controls)
        elif self._layout == "sbl":
            return pn.Row(self.layout_controls, self.view)
        elif self._layout == "sbr":
            return pn.Row(self.view, self.layout_controls)


class InteractivePlot(DynamicParam, PanelLayout):
    """Contains all the logic to create parametric-interactive plots."""

    # NOTE: why isn't Plot a parent class for InteractivePlot?
    # If that was the case, we would need to create multiple subclasses of
    # InteractivePlot, each one targeting a different backend.
    # Instead, we keep the backend (the actual plot) as an instance attribute.

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, name="", params=None, aux=dict(), **kwargs):
        """
        Parameters
        ==========
            args : tuple
                The usual plot arguments
            name : str
                Unused parameter
            params : dict
                In the keys there will be the symbols, in the values there will
                be parameters to create the slider associated to a symbol.
            aux : dict
                Auxiliary dictionary containing keyword arguments to be passed
                to DynamicParam.
            kwargs : dict
                Usual keyword arguments to be used by the backends and series.
        """
        layout = kwargs.pop("layout", "tb")
        ncols = kwargs.pop("ncols", 2)

        aux.setdefault("use_latex", kwargs.pop("use_latex", True))

        args = list(map(_plot_sympify, args))
        super().__init__(*args, name=name, params=params, **aux)
        PanelLayout.__init__(self, layout, ncols)

        # create the series
        series = self._create_series(*args, **kwargs)
        is_3D = all([s.is_3D for s in series])
        # create the plot
        Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
        self._backend = Backend(*series, **kwargs)

    def _create_series(self, *args, **kwargs):
        # read the parameters to generate the initial numerical data for
        # the interactive series
        kwargs["params"] = self.read_parameters()
        kwargs = _set_discretization_points(kwargs, InteractiveSeries)
        _slice = kwargs.get("slice", None)
        is_complex = kwargs.get("is_complex", False)
        is_vector = kwargs.get("is_vector", False)
        series = []
        if is_complex:
            new_args = []
            for a in args:
                exprs, ranges, label = _unpack_args(
                    *a, matrices=False, fill_ranges=False
                )
                new_args.append(Tuple(exprs[0], *ranges, label, sympify=False))
                # new_args.append(Tuple(exprs[0], ranges[0], label, sympify=False))
            series = _build_complex_series(*new_args, interactive=True, **kwargs)
        elif is_vector:
            args = _preprocess(*args, matrices=False, fill_ranges=False)
            series = _build_vector_series(*args, interactive=True, **kwargs)
        else:
            for a in args:
                # with interactive-parametric plots, vectors could have more free
                # symbols than the number of dimensions. We set fill_ranges=False
                # in order to not fill ranges, otherwise ranges will be created also
                # for parameters. This means the user must provided all the necessary
                # ranges.
                exprs, ranges, label = _unpack_args(
                    *a, matrices=True, fill_ranges=False
                )
                if isinstance(_slice, (tuple, list)):
                    # Sliced 3D vector field: each slice creates a unique series
                    kwargs2 = kwargs.copy()
                    kwargs2.pop("slice")
                    for s in _slice:
                        kwargs2["slice"] = s
                        series.append(
                            InteractiveSeries(exprs, ranges, label, **kwargs2)
                        )
                else:
                    series.append(InteractiveSeries(exprs, ranges, label, **kwargs))
        return series

    @property
    def fig(self):
        """Return the plot object"""
        return self._backend.fig


def iplot(*args, show=True, **kwargs):
    """Create interactive plots of symbolic expressions.
    NOTE: this function currently only works within Jupyter Notebook!

    Parameters
    ==========

        args : tuples
            Each tuple represents an expression. Depending on the type of
            expression we are plotting, the tuple should have the following
            forms:
            1. line: (expr, range, label)
            2. parametric line: (expr1, expr2, expr3 [optional], range, label)
            3. surface (expr, range1, range2, label)
            4. parametric surface (expr1, expr2, expr3, range1, range2, label)

            The label is always optional, whereas the ranges must always be
            specified. The ranges will create the discretized domain.

    Keyword Arguments
    =================

        params : dict
            A dictionary mapping the parameter-symbols to a parameter.
            The parameter can be:
            1. an instance of param.parameterized.Parameter (at the moment,
                param.Number is supported, which will result in a slider).
            2. a tuple of the form:
                (default, (min, max), N [optional], label [optional], spacing [optional])
                where:
                    N is the number of steps of the slider.
                    (min, max) must be finite numbers.
                    label: the custom text associated to the slider.
                    spacing : str
                        Specify the discretization spacing. Default to "linear",
                        can be changed to "log".

            Note that (at the moment) the parameters cannot be linked together
            (ie, one parameter can't depend on another one).

        layout : str
            The layout for the controls/plot. Possible values:
                'tb': controls in the top bar.
                'bb': controls in the bottom bar.
                'sbl': controls in the left side bar.
                'sbr': controls in the right side bar.
            Default layout to 'tb'. Keep in mind that side bar layouts may not
            work well with some backends.

        ncols : int
            Number of columns to lay out the widgets. Default to 2.

        is_complex : boolean
            Default to False. If True, it directs the internal algorithm to
            create all the necessary series (for example, one for the real part,
            one for the imaginary part).

        is_vector : boolean
            Default to False. If True, it directs the internal algorithm to
            create all the necessary series (for example, plotting the magnitude
            of the vector field as a contour plot).

        show : bool
            Default to True.
            If True, it will return an object that will be rendered on the
            output cell of a Jupyter Notebook. If False, it returns an instance
            of `InteractivePlot`.

        use_latex : bool
            Default to True.
            If True, the latex representation of the symbols will be used in the
            labels of the parameter-controls. If False, the string
            representation will be used instead.

        All the usual keyword arguments to customize the plot, such as title,
        xlabel, n (number of discretization points), ...


    Examples
    ========

    Surface plot between -10 <= x, y <= 10 with a damping parameter varying from
    0 to 1 with a default value of 0.15:

    .. code-block:: python
        x, y, d = symbols("x, y, d")
        r = sqrt(x**2 + y**2)
        expr = 10 * cos(r) * exp(-r * d)

        iplot(
            (expr, (x, -10, 10), (y, -10, 10)),
            params = { d: (0.15, (0, 1)) },
            title = "My Title",
            xlabel = "x axis",
            ylabel = "y axis",
            zlabel = "z axis",
            n = 100
        )

    A line plot illustrating the use of multiple expressions and:
    1. some expression may not use all the parameters
    2. custom labeling of the expressions
    3. custom number of steps in the slider
    4. custom labeling of the parameter-sliders

    .. code-block:: python
        x, A1, A2, k = symbols("x, A1, A2, k")
        iplot(
            (log(x) + A1 * sin(k * x), (x, 0, 20), "f1"),
            (exp(-(x - 2)) + A2 * cos(x), (x, 0, 20), "f2"),
            (A1 + A1 * cos(x), A2 * sin(x), (x, 0, pi)),
            params = {
                k: (1, (0, 5)),
                A1: (2, (0, 10), 20, "Ampl 1"),
                A2: (2, (0, 10), 40, "Ampl 2"),
            }
        )

    A 3D slice-vector plot. Note: whenever we want to create parametric vector
    plots, we should set `is_vector=True`:

    .. code-block:: python
        var("a, b, x:z")
        iplot(
            (Matrix([z * a, y * b, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5)),
            params = {
                a: (1, (0, 5)),
                b: (1, (0, 5))
            },
            n = 50,
            is_vector = True,
            scalar = x * y,
            slice = Plane((-2, 0, 0), (1, 0, 0))
        )

    A parametric complex domain coloring plot. Note: whenever we want to create
    parametric complex plots, we must set `is_complex=True`:

    .. code-block:: python
        var("x:z")
        iplot(
            ((z**2 + 1) / (x * (z**2 - 1)), (z, -4 - 2 * I, 4 + 2 * I)),
            params = {
                x: (1, (-2, 2))
            },
            backend = MB,
            is_complex = True,
            coloring = "b"
        )


    A parametric plot of a symbolic polygon. Note the use of `param` to create
    an integer slider.

    .. code-block:: python
        import param
        iplot(
            (Polygon((a, b), c, n=d), ),
            params = {
                a: (0, (-2, 2)),
                b: (0, (-2, 2)),
                c: (1, (0, 5)),
                d: param.Integer(3, softbounds=(3, 10), label="n"),
            },
            fill = False,
            aspect = "equal",
            use_latex = False
        )
    """
    i = InteractivePlot(*args, **kwargs)
    if show:
        return i.show()
    return i
