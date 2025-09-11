"""
Exposes the common docstring associated to some parameters that will be
used on many plotting functions.
"""


_PARAMS = """
params : dict, optional
    A dictionary mapping symbols to parameters. If provided, this
    dictionary enables the interactive-widgets plot.

    When calling a plotting function, the parameter can be specified with:

    * a widget from the ``ipywidgets`` module.
    * a widget from the ``panel`` module.
    * a tuple of the form:
        `(default, min, max, N, tick_format, label, spacing)`,
        which will instantiate a
        :py:class:`ipywidgets.widgets.widget_float.FloatSlider` or
        a :py:class:`ipywidgets.widgets.widget_float.FloatLogSlider`,
        depending on the spacing strategy. In particular:

        - default, min, max : float
            Default value, minimum value and maximum value of the slider,
            respectively. Must be finite numbers. The order of these 3
            numbers is not important: the module will figure it out
            which is what.
        - N : int, optional
            Number of steps of the slider.
        - tick_format : str or None, optional
            Provide a formatter for the tick value of the slider.
            Default to ``".2f"``.
        - label: str, optional
            Custom text associated to the slider.
        - spacing : str, optional
            Specify the discretization spacing. Default to ``"linear"``,
            can be changed to ``"log"``.

    Notes:

    1. parameters cannot be linked together (ie, one parameter
        cannot depend on another one).
    2. If a widget returns multiple numerical values (like
        :py:class:`panel.widgets.slider.RangeSlider` or
        :py:class:`ipywidgets.widgets.widget_float.FloatRangeSlider`),
        then a corresponding number of symbols must be provided.

    Here follows a couple of examples. If ``imodule="panel"``:

    .. code-block:: python

        import panel as pn
        params = {
            a: (1, 0, 5), # slider from 0 to 5, with default value of 1
            b: pn.widgets.FloatSlider(value=1, start=0, end=5), # same slider as above
            (c, d): pn.widgets.RangeSlider(value=(-1, 1), start=-3, end=3, step=0.1)
        }

    Or with ``imodule="ipywidgets"``:

    .. code-block:: python

        import ipywidgets as w
        params = {
            a: (1, 0, 5), # slider from 0 to 5, with default value of 1
            b: w.FloatSlider(value=1, min=0, max=5), # same slider as above
            (c, d): w.FloatRangeSlider(value=(-1, 1), min=-3, max=3, step=0.1)
        }

    When instantiating a data series directly, ``params`` must be a
    dictionary mapping symbols to numerical values.

    Let ``series`` be any data series. Then ``series.params`` returns a
    dictionary mapping symbols to numerical values.
"""


_SYSTEM = """
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
"""


_CONTROL_KW_IMPULSE = """
control_kw : dict, optional
    A dictionary of keyword arguments passed to
    :py:func:`control.impulse_response`
"""


_CONTROL_KW_STEP = """
control_kw : dict, optional
    A dictionary of keyword arguments passed to
    :py:func:`control.step_response`
"""


_CONTROL_KW_RAMP = """
control_kw : dict, optional
    A dictionary of keyword arguments passed to
    :py:func:`control.forced_response`
"""

_LABEL_PF = """
label : str or list/tuple, optional
    The label to be shown in the legend or colorbar. If not provided,
    the string representation of expr will be used. If a list of strings is
    provided, the number of labels must be equal to the number of expressions
    being plotted.
"""
