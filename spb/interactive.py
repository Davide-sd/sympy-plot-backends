import numpy as np
import param
import panel as pn
from sympy import latex, Tuple
from spb.backends.k3d import KB
from spb.backends.base_backend import Plot
from spb.series import InteractiveSeries
from spb.utils import _plot_sympify, _unpack_args
from spb.defaults import TWO_D_B, THREE_D_B

pn.extension("plotly")


"""
TODO:
    1. Automatic axis labeling based on provided expressions
    2. Sidebar left/right layout
    3. Decouple the layout into a new class, in that way maybe it could be
        possible to use iplot with a different GUI, for example Qt, by creating
        an InteractivePlotPanel, InteractivePlotQt, ...
    4. Log slider
"""

class DynamicParam(param.Parameterized):
    """ Dynamically add parameters based on the user-provided dictionary.
    Also, generate the lambda functions to be evaluated at a later stage.
    """
    # NOTE: why DynamicParam is a child class of param.Parameterized?
    # param is a full-python library, doesn't depend on anything else.
    # In theory, by using a parameterized class it should be possible to create
    # an InteractivePlotGUI class targeting a specific GUI.
    # At this moment, InteractivePlot is built on top of 'panel', so it only
    # works inside a Jupyter Notebook.
    
    def _tuple_to_dict(self, k, v):
        """ The user can provide a variable length tuple/list containing:
            (default, softbounds, N, label, type (lin, log))
        where:
            default : float
                Default value of the slider
            softbounds : tuple
                Tuple of two float (or integer) numbers: (start, end).
            N : int
                Number of increments in the slider. (start - end) / N represents
                the step increment. Default to 40. Set N=-1 to have unit step
                increments.
            label : str
                Label of the slider. Default to None. If None, the string or
                latex representation will be used. See use_latex for more
                information.
            type : str
                Can be "linear" or "log". Default to "linear".
        """
        defaults_keys = ["default", "softbounds", "step", "label", "type"]
        defaults_values = [1, (0, 2), 40, "$%s$" % latex(k) if self.use_latex
                else str(k), "linear"]
        values = defaults_values.copy()
        values[:len(v)] = v
        # set the step increment for the slider
        if values[2] > 0:
            values[2] = (values[1][1] - values[1][0]) / values[2]
        else:
            values[2] = 1
        return {k: v for k, v in zip (defaults_keys, values)}
    
    def __init__(self, *args, name="", parameters=None, **kwargs):
        # use latex on control labels and legends
        self.use_latex = kwargs.pop("use_latex", True)
        
        # this must be present in order to assure correct behaviour
        super().__init__(name=name, **kwargs)
        if not parameters:
            raise ValueError("`parameters` must be provided.")
        
        # The following dictionary will be used to create the appropriate
        # lambda function arguments:
        #    key: the provided symbol
        #    val: name of the associated parameter
        self.mapping = {}
        
        # create and attach the params to the class
        for i, (k, v) in enumerate(parameters.items()):
            if not isinstance(v, param.parameterized.Parameter):
                v = self._tuple_to_dict(k, v)
                # TODO: modify this to implement log slider
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
        
    def layout_controls(self):
        # split the controls in two columns
        params = sorted(list(self.mapping.values()))
        n = int(len(params) / 2)
        left = pn.panel(self, parameters=params[:n])
        right = pn.panel(self, parameters=params[n:])
        return pn.Row(left, right)
        
class InteractivePlot(DynamicParam):

    # NOTE: why isn't Plot a parent class for InteractivePlot?
    # If that was the case, it would not be trivial to instantiate the selected
    # backend. Therefore, in the following implementation the backend (the
    # actual plot) is an instance attribute of InteractivePlot.

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, name="", parameters=None, fig_kw=dict(), **kwargs):
        args = list(map(_plot_sympify, args))
        super().__init__(*args, name=name, parameters=parameters, **kwargs)
        
        # create the series
        series = self._create_series(*args, **fig_kw)
        is_3D = all([s.is_3D for s in series])
        # create the plot
        Backend = fig_kw.pop("backend", THREE_D_B if is_3D else TWO_D_B)
        self._backend = Backend(*series, **fig_kw)
        # add the series to the plot
        self._backend._process_series(self._backend._series)
    
    def _create_series(self, *args, **kwargs):
        # read the parameters to generate the initial numerical data for
        # the interactive series
        kwargs["params"] = self.read_parameters()
        series = []
        for a in args:
            exprs, ranges, label = _unpack_args(*a)
            series.append(InteractiveSeries(exprs, ranges, label, **kwargs))
        return series
    
    def fig(self):
        """ Return the plot object
        """
        return self._backend.fig
            
    def view(self):
        params = self.read_parameters()
        self._backend._update_interactive(params)
        if isinstance(self._backend, KB):
            return pn.pane.Pane(self._backend.fig, width=800)
        else:
            return self.fig
    
    def show(self):
        return pn.Column(self.layout_controls, self.view)


def iplot(*args, show=True, **kwargs):
    """ Create interactive plots of symbolic expressions.
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

        parameters : dict
            A dictionary mapping the parameter-symbols to a parameter.
            The parameter can be:
            1. an instance of param.parameterized.Parameter (at the moment,
                param.Number is supported, which will result in a slider).
            2. a tuple of the form:
                (default, (min, max), N [optional], label [optional])
                where N is the number of steps of the slider.
            
            Note that (at the moment) the parameters cannot be linked together
            (ie, one parameter can't depend on another one).
        
        fig_kw : dict
            A dictionary with the usual keyword arguments to customize the plot,
            such as title, xlabel, n (number of discretization points), ...
            This dictionary will be passed to the backend: check its
            documentations to find more keyword arguments.
        
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
            parameters = { d: (0.15, (0, 1)) },
            fig_kw = dict(
                title = "My Title",
                xlabel = "x axis",
                ylabel = "y axis",
                zlabel = "z axis",
                n = 100
            )
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
            parameters = {
                k: (1, (0, 5)),
                A1: (2, (0, 10), 20, "Ampl 1"),
                A2: (2, (0, 10), 40, "Ampl 2"),
            },
            fig_kw = { "legend": True }
        )

    """
    i = InteractivePlot(*args, **kwargs)
    if show:
        return i.show()
    return i