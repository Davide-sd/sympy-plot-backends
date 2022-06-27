==========
 Changelog
==========

v1.1.4
======

* ``color_func`` is back-compatible with ``sympy.plotting``'s
  ``line_color`` and ``surface_color``.


v1.1.3
======

* Added ``color_func`` support to parametric line series.
* Improved docstring.


v1.1.2
======

* `iplot`:

  * Added ``servable`` keyword argument: ``servable=True`` will serves the
    application to a new browser windows,
  * Added ``name`` keyword argument: if used with ``servable=True`` it will
    add a title to the interactive application.

* Default settings:

  * Added ``servable`` and ``theme`` to ``interactive`` section.

* Fixed a bug when plotting lines with ``BokehBackend``.
* Improved the way of setting the number of discretization points: ``n``
  can now be a two (or three) elements tuple, which will override ``n1`` and
  ``n2``.
* It is now possible to pass a float number of discretization points, for
  example ``n=1e04``.
* added ``label`` keyword argument to plot functions.
  


v1.1.1
======

* Added ``color_func`` keyword argument to:

  * `plot` to apply custom coloring to lines.
  * `plot3d` and `plot3d_parametric_surface` to apply custom coloring to 3D
     surfaces.
  * to accomodate ``color_func``, ``ParametricSurfaceSeries.get_data()`` now
    returns 5 elements instead of 3.

* Added plot range to default settings.
* Implemented a custom printer for interval math to be used inside
  ``ImplicitSeries``.
* Added ``plot3d_implicit`` to visualize implicit surfaces.
* ``MatplotlibBackend`` now uses default colorloop from ``plt.rcParams['axes.prop_cycle']``.


v1.1.0
======

* ``polar_plot``:

  * a polar chart will be generated if a backend support such feature,
    otherwise the backend will apply a polar transformation and plot a
    cartesian chart.
  * ``iplot`` changes the keyword argument to request a 2D polar chart. Use
    ``is_polar=True`` instead of ``polar=True``.

* ``plot3d``:

  * Setting ``is_polar=True`` enables polar discretization.

* 3d vector plots:

  * Keyword argument ``slice`` can now acccept instances of surface-related
    series (as well as surface interactive series).
  * Improved ``PlotlyBackend`` and ``K3DBackend`` support for 3D vector-quiver
    interactive series.

* Default setting:

  * Added adaptive ``"goal"``.
  * Added ``use_cm`` for 3D plots.

* Added ``tx, ty, tz`` keyword arguments. Now it is possible to apply
  transformation functions to the numerical data, for example converting the
  domain of a function from radians to degrees.

* Added Latex support and a the `use_latex` keyword argument to toggle on/off
  the use of latex labels. Plot functions will use latex labels on the axis by
  default, if the backend supports such feature. The behaviour can be changed
  on the default settings.

* Fixed bug within ``iplot`` and ``K3DBackend`` when setting ``use_cm=False``.

* ``iplot`` parameters can accept symbolic numerical values (of type
  ``Integer``, ``Float``, ``Rational``).

* Removed ``plot_data`` module.


v1.0.4
======

* Bug fix for plotting real/imag of complex functions.


v1.0.3
======

* Deprecated ``get_plot_data`` function.
* Exposed ``create_series`` function from the ``spb.interactive`` module.
* Removed dependency on `sympy.plotting.experimental_lambdify`. Now this
  plotting module relies only on lambdify.
* Improved testing of ``plot_implicit``.
* Added quickstart tutorials to ReadTheDocs.


v1.0.2
======

* Added backend's aliases into ``__init__.py``.
* Added example to the ``plot`` function.
* Improved docstring and examples of ``plot_implicit``.
* Fixed bug with ``PlotlyBackend`` in which axis labels were not visible.
* Added ``throttled`` to default settings of interactive.
* Added ``grid`` to defaults settings of all backends.


v1.0.1
======

* Exiting development status Beta
* Updated ``K3DBackend`` documentation.
* Updated tutorial


v1.0.0
======


* Data series:

  * Integrated `adaptive module <https://github.com/python-adaptive/adaptive/>`_
    with SymPy Plotting Backends.

    * Implemented adaptive algorithm for 3D parametric lines and 3D surfaces.
    * added ``adaptive_goal`` and ``loss_fn`` keyword arguments to control the
      behaviour of adaptive algorithm.

  * Improved support for integer discretization.

  * Integrated ``lambdify`` into data series to generate numerical data.

    * partially removed dependency ``sympy.plotting.experimental_lambdify``.
      Only ``ImplicitSeries`` still uses it for its adaptive implementation
      with interval arithmetic.
    * Added ``modules`` keyword argument to data series in order to choose the
      ``lambdify`` module (except ``ImplicitSeries``).

  * Line series now implements the ``_detect_poles`` algorithm.

  * Added ``rendering_kw`` attribute to all data series.

  * Refactoring of ``InteractiveSeries``:

    * ``InteractiveSeries`` is now a base class.
    * Implemented several child classes to deal with specific tasks.
    * Removed ``update_data`` method.
    * Added ``params`` attribute as a property.
    * Fixed the instantiation of subclasses in ``__new__``.


* Functions:

  * removed aliases of plotting functions.

  * Added complex-related plotting functions:

    * ``plot_complex`` now plots the absolute value of a function colored by
      its argument.
    * ``plot_real_imag``: plot the real and imaginary parts.
    * ``plot_complex_list``: plot list of complex points.
    * ``plot_complex_vector``: plot the vector field `[re(f(z)), im(f(z))]` of
      a complex function `f`.

  * ``plotgrid`` is now fully functioning.

  * added ``plot_list`` to visualize lists of numerical data.

  * added ``sum_bound`` keyword argument to ``plot``: now it is possible to
    plot summations.

  * removed ``process_piecewise`` keyword argument from ``plot``. Now, ``plot``
    is unable to correctly display ``Piecewise`` expressions and their
    discontinuities.

  * added ``plot_piecewise`` to correctly visualize ``Piecewise`` expressions
    and their discontinuities.

  * added ``is_point`` and ``is_filled`` keyword arguments to ``plot`` and
    ``plot_list`` in order to visualize filled/empty points.

  * replaced ``fill`` keyword argument with ``is_filled`` inside
    ``plot_geometry``.

  * ``iplot``:

    * implemented addition between instances of ``InteractivePlot`` and
      ``Plot``.
    * fixed bug with ``MatplotlibBackend`` in which the figure would show up
      twice.

  * Deprecation of ``smart_plot``.

  * ``plot_parametric`` and ``plot3d_parametric_line``: the colorbar now shows
    the name of the parameter, not the name of the expression.


* Backends:

  * ``Plot``:

    * improved support for addition between instances of ``Plot``.
    * improved instantiation of child classes in ``__new__`` method.
    * removed ``_kwargs`` instance attribute.

  * ``MatplotlibBackend``:

    * ``fig`` attribute now returns only the figure. The axes can be
      retrieved from its figure.
    * Dropped support for ``jupyterthemes``.
    * Fix bug in which the figure would show up twice on Jupyter Notebook.
    * Added colorbar when plotting only 2D streamlines.

  * ``PlotlyBackend``:

    * removed the ``wireframe`` keyword argument and dropped support
      for 3D wireframes.
    * dropped support for ``plot_implicit``.

  * `BokehBackend`:

    * add `update_event` keyword argument to enable/disable auto-update on
      panning for line plots.
    * dropped support for ``plot_implicit``.

  * `K3DBackend`:

    * fixed bug with ``zlim``.

  * All backends:

    * Generates numerical data and add it to the figure only when ``show()`` or
      ``fig`` are called.
    * ``colorloop``, ``colormaps`` class attributes are now empty lists.
      User can set them to use custom coloring. Default coloring is
      implemented inside ``__init__`` method of each backend.


* Performance:

  * Improved module's load time by replacing `from sympy import somethig` with
    `from sympy.module import somethig`.
  * Improved module's load time by loading backend's dependencies not at the
    beginning of the module, but only when they are required.


* Default settings:

  * Change backend's themes to light themes.
  * Added options to show grid and minor grid on bokeh, plotly and matplotlib.
  * Added `interactive` section and the `use_latex` option.
  * Added ``update_event`` to bokeh.


* Documentation:

  * Improved examples in docstring of plotting functions.
  * Removed tutorials from the `Tutorials` section as they slowed down the
    pages.
  * Improved organization.
