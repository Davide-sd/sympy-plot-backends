4 - Parametric-Interactive Plots
--------------------------------

In this tutorial we are going to see how to create simple parametric-interactive
plots, that is, plots where we can move sliders to change parameters.
The word *simple* refer to the fact that:

* the function expects one-output-value widgets in order to work properly,
  such as sliders, spinners, etc.
* only one plot is generated.
* we don't have many options regarding the overall layout.

The function that allows to do that is `iplot`, which stands for
*interactive plot*. Keep in mind that it only works inside Jupyter Notebook,
since it is based on `holoviz's panel <https://panel.holoviz.org/>`_.
Refer to :ref:`interactive` to learn more about this function.

.. jupyter-execute::

    >>> from sympy import *
    >>> from spb.interactive import iplot
    >>> from spb.backends.plotly import PB
    >>> from spb.backends.bokeh import BB
    >>> from spb.backends.matplotlib import MB

**NOTE**:

* in the following examples, we can move the sliders but the data won't
  update. That's perfectly fine because this is a static documentation page:
  there is no Python kernel running below.
* the static examples might take a few seconds to load.


Example 1
=========

In this example we are going to create a parameterized surface plot with one
parameter, the damping coefficient. Note: the first time loading a
3D-interactive plot may takes a few seconds:

.. jupyter-execute::

    >>> x, y, d = symbols("x, y, d")
    >>> r = sqrt(x**2 + y**2)
    >>> expr = 10 * cos(r) * exp(-r * d)
    >>> iplot(
    ...     (expr, (x, -10, 10), (y, -10, 10)),
    ...     params = { d: (0.15, 0, 1) },
    ...     title = "My Title",
    ...     xlabel = "x axis",
    ...     ylabel = "y axis",
    ...     zlabel = "z axis",
    ...     n = 100,
    ...     threed = True,
    ...     use_latex = True,
    ...     backend = MB
    ... )


* By moving the slider, the update only happens when we release the click
  from it. This behaviour improves user experience: it should minimize the lags
  while moving the sliders waiting for the previous evaluation to complete.
  Remember: in the background the symbolic expression is being numerically
  evaluated over the discretized domain.
* By default, ``iplot`` uses the backends we set in configuration settings in
  Tutorial 3. Alternatively, we can use a different backend by setting the
  ``backend`` keyword argument.
* ``(expr, (x, -10, 10), (y, -10, 10))``: with ``iplot``, we must always
  provide all the necessary ranges: it's required by internal algorithm in
  order to automatically detect the kind of expression we are trying to plot.
* ``params = { d: (0.15, 0, 1) }``: we specified the dictionary of parameters.
  In this case there is only one parameter:

  * the key represents the symbol
  * the value is a tuple of the form (default, (min, max), N [optional],
    label [optional], spacing [optional]). In the above case, only the first
    two entries were provided. This tuple will be converted to a
    ``param.Number``, which represent a float number and will be rendered as
    a slider. Alternatively, we could have used
    `holoviz's param library <https://panel.holoviz.org/user_guide/Param.html>`_:

    .. code:: ipython3

        import param
        parameters = { 
            d: param.Number(0.15, softbounds=(0, 1), label="", step=0.025)
        }

    We can use any parameter that, once rendered in a GUI, returns a single
    numeric value. For example ``param.Parameter``, ``param.Integer``,
    ``param.Number``, ``param.ObjectSelector``, ``param.Boolean``.
* Next, we specify the usual keyword argument to customize the plot.
  Note that ``n = 100`` sets the number of discretization points in both
  directions (x and y). Alternatively, we could set them separately with
  ``n1`` and ``n2``. 
* Since ``iplot`` is a very general function, there is a risk of ambiguity:
  the above expression can be plotted with contours or with a 3D surface.
  With ``threed = True`` we ask for a 3D surface plot. In the above example,
  set ``threed = False`` to get a contour plot.
* ``use_latex = True``: by default, the label of the slider will use the
  Latex code of the parameter-symbol to provide a legible experience
  (especially when dealing with symbols with subscripts, superscripts, ...);
  that's the theory. In practice though, at the time writing this tutorial
  it is a *hit or miss* experience: most of the time this functionality
  doesn't work. If that is the case, then it is better to set
  ``use_latex = False``, which will display the string representation
  of the symbols.
* ``iplot`` returns two very different objects depending on the value of the
  keyword argument ``show``:

  * ``show=True`` (default) returns a ``panel`` object that will be rendered
    in the output cell. We can prevent the rendering from happening by
    capturing this object in a variable, for example ``p = iplot(...``.
    Then we can render it on a different cell by simply typing ``p``.
  * ``show=False`` returns an instance of ``InteractivePlot``, which can be
    used for debugging purposes.


Example 2
=========

Let's now plot three expressions having quite a lot of parameters, representing
the temperature distribution at the walls of an annular nuclear fuel rod, as
well as the temperature of the coolant.

The only things the Reader needs to be aware of are:

* ``z`` represents the position along the anular channel. It is the discretized
  domain;
* ``ri`` represents the inner radius of the channel;
* ``ro`` represents the outer radius of the channel;
* ``ri < ro``, which is a physical condition.

.. jupyter-execute::

    >>> r, ro, ri = symbols("r, r_o, r_i")
    >>> mdot, cp, hc = symbols(r"\dot{m}, c_p, h_c")
    >>> alpha, k, L, z = symbols("alpha, k, L, z")
    >>> Tin, Pave = symbols(r"T_{in}, P_{ave}")
    >>> # Fuel temperature distribution along the channel
    >>> # here, the only variable is z, everything else are parameters
    >>> Tf = (
    ...     Tin
    ...     + (Pave * L * pi * (ro ** 2 - ri ** 2) / (2 * mdot * cp))
    ...     * (1 - sin(alpha * (L / 2 - z)) / sin(alpha * L / 2))
    ...     + (alpha * Pave * L / 2)
    ...     * (cos(alpha * (L / 2 - z)) / sin(alpha * L / 2))
    ...     * (
    ...         (ro ** 2 - ri ** 2) / (2 * hc * ri)
    ...         - (1 / (2 * k)) * ((r ** 2 - ri ** 2) / 2 + ro ** 2 * log(ri / r))
    ...     )
    ... )
    >>> # Fuel temperature distribution at the inner and outer walls
    >>> Twi = Tf.subs(r, ri)
    >>> Two = Tf.subs(r, ro)
    >>> # Cooling fluid temperature
    >>> Tp = Tin + (Pave * L / 2) * pi * (ro ** 2 - ri ** 2) / (mdot * cp) * (
    ...     1 - sin(alpha * (L / 2 - z)) / sin(alpha * L / 2)
    ... )

Note that ``Twi, Two, Tp`` use a different number of parameters:

.. jupyter-execute::

    >>> Twi.free_symbols, Two.free_symbols, Tp.free_symbols


Let's try to use ``MatplotlibBackend``:

.. jupyter-execute::

    >>> # %matplotlib widget
    >>> iplot(
    ...     (Twi, (z, 0, 100), "Twi"),
    ...     (Two, (z, 0, 100), "Two"),
    ...     (Tp, (z, 0, 100), "Tp"),
    ...     params = {
    ...         ri: (0.2, 0.04, 0.5),
    ...         ro: (0.4, 0.2, 1.6),
    ...         L: (100, 25, 250),
    ...         Pave: (1000, 400, 4000),
    ...         Tin: (300, 100, 500),
    ...         hc: (1, 0.4, 15),
    ...         alpha: (0.031, 0.016, 0.031),
    ...         mdot: (1, 0.5, 5),
    ...         k: (0.2, 0.1, 2),
    ...         cp: (15, 5, 25)
    ...     },
    ...     title = "Temperature distribution",
    ...     xlabel = "Position [cm]",
    ...     ylabel = "T [K]",
    ...     ylim = (0, 3000),
    ...     xlim = (0, 100),
    ...     backend = MB
    ... )


* Whenever we use it with ``iplot`` with ``MatplotlibBackend``, we must use
  the magic line ``%matplotlib widget``, otherwise the figure won't update.
  In the above code block, it has been commented out only to be able to include
  the picture in this documentation page.
  Note that with ``MatplotlibBackend`` the widgets will always be rendered
  below the figure.
* Independently of the number of parameters, ``iplot`` arranges the sliders
  in two columns. We can change the number of columns by setting ``ncols``
  to some integer. We can also chose where to place the controls with the
  ``layout`` keyword argument, but this won't work with ``MatplotlibBackend``.
  Read the documentation to find out the available options.
* Note that we set the sliders:

  * ``0.04 <= ri <= 0.5``
  * ``0.4 <= ro <= 1.6``
  
  Therefore, it is very well possible to break the physical condition
  ``ri < ro`` (for example, ``ri = 0.5`` and
  ``ro = 0.4``), which would produce unphysical results.
  The selection of the bounds and the values of the sliders is critical,
  and we are responsible for it. Currently it is impossible to set
  relationships between parameters!

As we can see, there are quite a few widgets in this plot. Maybe we are working
with a small screen device, or maybe the width of Jupyter Notebook's main
content area is limiting us. In such cases we can launch the plot on a
different browser window to use all its available width. This only works with
``BokehBackend`` and ``PlotlyBackend``. Let's give it a try:

.. code-block:: python

   t = iplot(
       (Twi, (z, 0, 100), "Twi"),
       (Two, (z, 0, 100), "Two"),
       (Tp, (z, 0, 100), "Tp"),
       params = {
           ri: (0.2, 0.04, 0.5),
           ro: (0.4, 0.2, 1.6),
           L: (100, 25, 250),
           Pave: (1000, 400, 4000),
           Tin: (300, 100, 500),
           hc: (1, 0.4, 15),
           alpha: (0.031, 0.016, 0.031),
           mdot: (1, 0.5, 5),
           k: (0.2, 0.1, 2),
           cp: (15, 5, 25)
       },
       title = "Temperature distribution",
       xlabel = "Position [cm]",
       ylabel = "T [K]",
       ylim = (0, 3000),
       xlim = (0, 100),
       backend = BB,
       layout = "sbl",
       ncols = 1,
       size = (800, 600),
       show = True
   )
   t.show()

Here, the ``panel`` object created by ``iplot`` has been "captured" into the
variable ``t``. With ``t.show()`` we are launching a new server process that
will server the interactive plot on a new browser window. Note that we layed
out the widgets differently and we also increased the size of the plot.


Example 3
=========

In this example we are going to illustrate the use of
`holoviz's param library <https://panel.holoviz.org/user_guide/Param.html>`_.

Let's say we would like to visualize the Fourier Series approximation of a
`sawtooth wave <https://mathworld.wolfram.com/SawtoothWave.html>`_, defined as:

.. figure:: ../_static/tut-4/equation-1.png

where ``T`` is the period. Its
`Fourier Series <https://mathworld.wolfram.com/FourierSeriesSawtoothWave.html>`_
is:

.. figure:: ../_static/tut-4/equation-2.png

.. jupyter-execute::

    >>> x, T, n, m = symbols("x, T, n, m")
    >>> sawtooth = frac(x / T)
    >>> # Fourier Series of a sawtooth wave
    >>> # https://mathworld.wolfram.com/FourierSeriesSawtoothWave.html
    >>> fs = S(1) / 2 - (1 / pi) * Sum(sin(2 * n * pi * x / T) / n, (n, 1, m))


Note that we stopped the Fourier series at ``m`` rathen than ``infinity``,
because ``m`` represents the upper bound of the approximation.

In the above expressions:

* ``T`` is a float number, therefore we can use the tuple-sintax used before.
* ``n`` is an integer number, therefore we must specify an integer parameter
  with ``param.Integer``.


.. jupyter-execute::

    >>> import param
    >>> iplot(
    ...     (sawtooth, (x, 0, 10), "f"),
    ...     (fs, (x, 0, 10), "approx"),
    ...     params = {
    ...         T: (2, 0, 10),
    ...         m: param.Integer(3, bounds=(1, None), label="Sum up to n ")
    ...     },
    ...     xlabel = "x",
    ...     ylabel = "y",
    ...     backend = MB
    ... )
