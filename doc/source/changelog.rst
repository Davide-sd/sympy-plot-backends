==========
 Changelog
==========

v1.0.4
======

* Bug fix for plotting real/imag of complex functions.


v1.0.3
======

* Deprecated `get_plot_data` function.
* Exposed `create_series` function from the `spb.interactive` module.
* Removed dependency on `sympy.plotting.experimental_lambdify`. Now this
  plotting module relies only on lambdify.
* Improved testing of `plot_implicit`.
* Added quickstart tutorials to ReadTheDocs.


v1.0.2
======

* Added backend's aliases into `__init__.py`.
* Added example to the `plot` function.
* Improved docstring and examples of `plot_implicit`.
* Fixed bug with `PlotlyBackend` in which axis labels were not visible.
* Added `throttled` to default settings of interactive.
* Added `grid` to defaults settings of all backends.


v1.0.1
======

* Exiting development status Beta
* Updated `K3DBackend` documentation.
* Updated tutorial


v1.0.0
======


* Data series:

  * Integrated `adaptive module <https://github.com/python-adaptive/adaptive/>`_
    with SymPy Plotting Backends.

    * Implemented adaptive algorithm for 3D parametric lines and 3D surfaces.
    * added `adaptive_goal` and `loss_fn` keyword arguments to control the
      behaviour of adaptive algorithm.

  * Improved support for integer discretization.

  * Integrated `lambdify` into data series to generate numerical data.

    * partially removed dependency `sympy.plotting.experimental_lambdify`.
      Only `ImplicitSeries` still uses it for its adaptive implementation with
      interval arithmetic.
    * Added `modules` keyword argument to data series in order to choose the
      `lambdify` module (except `ImplicitSeries`).

  * Line series now implements the `_detect_poles` algorithm.

  * Added `rendering_kw` attribute to all data series.

  * Refactoring of `InteractiveSeries`:

    * `InteractiveSeries` is now a base class.
    * Implemented several child classes to deal with specific tasks.
    * Removed `update_data` method.
    * Added `params` attribute as a property.
    * Fixed the instantiation of subclasses in `__new__`.


* Functions:

  * removed aliases of plotting functions.

  * Added complex-related plotting functions:

    * `plot_complex` now plots the absolute value of a function colored by its
      argument.
    * `plot_real_imag`: plot the real and imaginary parts.
    * `plot_complex_list`: plot list of complex points.
    * `plot_complex_vector`: plot the vector field `[re(f(z)), im(f(z))]` of a
      complex function `f`.

  * `plotgrid` is now fully functioning.

  * added `plot_list` to visualize lists of numerical data.

  * added `sum_bound` keyword argument to `plot`: now it is possible to plot
    summations.

  * removed `process_piecewise` keyword argument from `plot`. Now, `plot` is
    unable to correctly display `Piecewise` expressions and their
    discontinuities.

  * added `plot_piecewise` to correctly visualize `Piecewise` expressions and
    their discontinuities.

  * added `is_point` and `is_filled` keyword arguments to `plot` and
    `plot_list` in order to visualize filled/empty points.

  * replaced `fill` keyword argument with `is_filled` inside `plot_geometry`.

  * `iplot`:

    * implemented addition between instances of `InteractivePlot` and `Plot`.
    * fixed bug with `MatplotlibBackend` in which the figure would show up
      twice.

  * Deprecation of `smart_plot`.

  * `plot_parametric` and `plot3d_parametric_line`: the colorbar now shows the
    name of the parameter, not the name of the expression.


* Backends:

  * `Plot`:

    * improved support for addition between instances of `Plot`.
    * improved instantiation of child classes in `__new__` method.
    * removed `_kwargs` instance attribute.

  * `MatplotlibBackend`:

    * `fig` attribute now returns only the figure. The axes can be
      retrieved from its figure.
    * Dropped support for `jupyterthemes`.
    * Fix bug in which the figure would show up twice on Jupyter Notebook.
    * Added colorbar when plotting only 2D streamlines.

  * `PlotlyBackend`:

    * removed the `wireframe` keyword argument and dropped support
      for 3D wireframes.
    * dropped support for `plot_implicit`.

  * `BokehBackend`:

    * add `update_event` keyword argument to enable/disable auto-update on
      panning for line plots.
    * dropped support for `plot_implicit`.

  * `K3DBackend`:

    * fixed bug with `zlim`.

  * All backends:

    * Generates numerical data and add it to the figure only when `show()` or
      `fig` are called.
    * `colorloop`, `colormaps` class attributes are now empty lists. User can
      set them to use custom coloring. Default coloring is implemented inside
      `__init__` method of each backend.


* Performance:

  * Improved module's load time by replacing `from sympy import somethig` with
    `from sympy.module import somethig`.
  * Improved module's load time by loading backend's dependencies not at the
    beginning of the module, but only when they are required.


* Default settings:

  * Change backend's themes to light themes.
  * Added options to show grid and minor grid on bokeh, plotly and matplotlib.
  * Added `interactive` section and the `use_latex` option.
  * Added `update_event` to bokeh.


* Documentation:

  * Improved examples in docstring of plotting functions.
  * Removed tutorials from the `Tutorials` section as they slowed down the
    pages.
  * Improved organization.
