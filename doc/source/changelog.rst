==========
 Changelog
==========

v4.0.0
======

* Refactoring of the plotting module in order to use the
  `param module <https://param.holoviz.org>`_. While this add a new
  mandatory dependency (previously, it was installed and used only
  when using the interactive module), it brings many benefits:

  * Clearer code base, in particular in the `spb.series` modules.
  * Automatic and complete generation of the documentation when the
    module is imported, on all classes and plotting function.
    The documentaion is shown when the user executes
    `help(plot_function)` or `plot_function?`. All available
    parameters will be shown.

* Changed the init signature of ``NicholsLineSeries``.

* Added capability to set axis ticks to multiple of some quantity
  (for example `pi/2`) thanks to the keyword arguments
  ``x_ticks_formatter`` and ``x_ticks_formatter`` of the function
  ``graphics()``. A few preconfigured formatter are available as well, like
  ``multiples_of_pi_over_2``, etc.

* Added ``hooks`` keyword argument to ``graphics``: it accepts a list of user
  defined functions that are going to further customize the appearance of the
  plot. For example, users can change the tick labels on the colorbars, etc.

* Added support for plotting summations with infinite bounds on plotting
  function.

* Improved evaluation of symbolic expressions producing very large numbers.

* Improved the handling of grid lines thanks to the keyword arguments ``grid``
  and ``minor_grid`` of the function ``graphics()``. This parameters can be:

  * boolean: toggle the visibility of major and minor grid lines.
  * dict: keyword arguments used to customize the grid lines.

* Improved logic dealing with lambdification in order minimize the time
  spent in this stage.

* Removed the adaptive algorithm from
  ``line, line_parametric_2d, line_parametric_3d, surface`` (and their
  respective counterpars, ``plot, plot_parametric, plot3d_parametric, plot3d``).
  Main motivations were:

  1. easier to use plotting module, as there are 3/4 keyword arguments less
     to be worried about.
  2. cleaner and simpler code base.
  3. easier to implement new features (interactive ones), that only works with
     the uniform evaluation strategy.

* Split the ``GeometrySeries`` into ``Geometry2DSeries`` and
  ``Geometry3DSeries`` for better code separation.

* Added attribute ``Geometry2DSeries.range_x`` which allows to plot a
  ``Line2D`` in the specified range along the x-axis.

* remove ``tp`` keyword argument from ``step_response``, ``ramp_response``,
  ``impulse_response``.

* fixed bugs:

  * with the numerical algorithm about singularities detection of
    2D parametric lines.
  * with the algorithm used to insert exclusion points through the
    ``exclude`` keyword argument.

* ``LineOver1DRangeSeries``: added a new attribute, ``poles_rendering_kw``,
  which is a dictionary of keyword arguments passed to the specific plotting
  library renderer in order to customize the appearance of vertical lines
  representing essential discontinuities.


v3.4.3
======

* Fixed bug with retrieval of free symbols from symbolic expressions.


v3.4.2
======

* small update to ``PlotlyBackend`` and related renderers in order to keep it
  up-to-date with Plotly.
  Thanks to `zarstensen  <https://github.com/zarstensen>`_ for the fix.

* small update to ``Implicid2DRenderer`` in order to keep it
  up-to-date with Matplotlib.

* Improved robustness of colormap conversion.


v3.4.1
======

* Adjusted code to run with SymPy>=1.13, in which numbers finally follows
the structural equality rule (for example, ``2.0 == 2``
returns ``False``).

* Added support for Numpy>=2.0.0.


v3.4.0
======

* Implemented animations.

* ``BokehBackend`` is now able to create contour plots.


v3.3.0
======

* Control system plotting

  * Fixed bug with missing title of ``plot_root_locus``.

  * Implemented ``sgrid`` based on Matplotlib projection.

  * Improved ``sgrid`` support on Bokeh interactive plots.

  * Improved labeling of MIMO systems.

  * ``step_response, impulse_response, ramp_response`` are now able to
    compute an appropriate upper time limit for the simulation.

  * Updated code to use ``control 0.10.0``. As a consequence, Python 3.9 is
    no longer supported by this module.

* Improvements of interactive plots:

  * Added support for multiple-values widgets, like ``RangeSlider``.

  * Improvements of ``spb.interactive.panel``:

    * Simplified underlying architecture. Previously, ``InteractivePlot``
      inherited from ``param.Parameterized``: widgets were class
      attributes. Instantiating a new interactive plot would make the previous
      instance completely unusable. This inheritance has been removed.
      Now, widgets are instance attributes. Multiple instances work perfectly
      fine.

    * Added support for widgets of Holoviz Panel.

    * Updated plotgrid to work with the new architecture.

  * Improvements to documentation.

* Fixed bug with legend of ``plot_vector``.


v3.2.0
======

* add ``update_event`` keyword argument to enable/disable auto-update on
  panning for plots created with ``MatplotlibBackend``, ``PlotlyBackend`` and
  ``BokehBackend``. By default, this functionality is turned off,
  ``update_event=False``.

* Improved the logic handling the creation of sliders on interactive-widget
  plots. Consider this code: ``params = {k: (1, 0, 5, formatter, label)``.
  It now works both with `ipywidgets` as well as `panel`. Previously,
  ``formatter`` was not supported by ``ipywidgets``.

* Added ``arrows=`` keyword argument to ``nichols`` and ``plot_nichols``.

* Added ``show_minus_one=`` keyword argument to ``mcircles``.

* Implemented renderers for Bokeh in order to deal with control
  system plotting.

* Improved tooltips in ``BokehBackend``.

* Breaking: refactoring of ``NicholsLineSeries``. Previously, it returned data
  about the open-loop transfer function. Now, it also returns data about the
  closed-loop transfer function, which can be used on tooltips.


v3.1.1
======

* Fix incorrect behavior of "arrow_3d".


v3.1.0
======

* User can now specify an existing figure over which symbolic expressions
  will be plotted. Just use the ``ax=`` or ``fig=`` keyword argument.

* Added ``arrow_3d`` to ``spb.graphics.vectors`` in order to plot a single
  arrow in a three-dimensional space.

* Enhanced capabilities of line plots with the ``steps`` keyword argument.
  Possible values are ``"pre", "post", "mid"``, mimicking Matplotlib's
  ``step`` function.

* New features on the ``spb.graphics.control`` sub-module:

  * It now depends on the
    `python-control module <https://python-control.readthedocs.io/en/0.9.4/>`_.
    This dependency allows the implementation of new plotting functions and to
    seamlessly deal with transfer functions from ``sympy``, ``control`` and
    ``scipy.signal``, supporting both continous-time and discrete-time systems.

  * Added support for MIMO systems.

  * Created ``sgrid, zgrid, ngrid, mcircles`` functions to easily create grids
    for control system plots. Appropriate keyword arguments have been created
    on all major plot functions in order to activate these grids.

  * Added ``plot_root_locus`` to the control submodule.

  * ``plot_bode`` now auto-computes an appropriate frequency range.

  * Removed the transfer function's Latex representation from the title of
    plots related to control systems. This decision is motivated from practical
    experience, where most of the transfer functions have floating point
    coefficients, which makes their Latex representation too big to fit into
    the small width of a plot.

* Refactoring of the ``series.py`` sub-module:

  * code has been re-organized to make extensibility easier and
    slightly improve performance. In particular, the mixin class
    ``CommonUniformEvaluation`` has been introduced, which handles all the
    machinery necessary to evaluate symbolic expressions. Series classes
    may or may not inherit from it. ``CommonUniformEvaluation`` allows for a
    better separation of scopes: data series that don't need that code are
    faster to instantiate.

  * Breaking: refactoring of ``NyquistLineSeries`` in order to use the
    ``control`` module. In particular, the ``get_data`` method now returns
    many more arrays.

  * Breaking: removed attribute ``use_quiver_solid_color`` from
    ``Vector2DSeries``.

* Fixed bug with labels of 2D vector fields.


v3.0.1
======

* Added new coloring option to ``domain_coloring``.
  Setting ``coloring="k+log"`` will apply a logarithm to the magnitude of the
  complex function. This improves the visibility of zeros in complex functions
  that have very steep poles.

* Added the ``hyper`` function to the list of functions to be evaluated with
  real numbers. This avoids unexpected errors.

* Set ``unwrap=True`` as defaul option for ``plot_bode``: this helps to get
  a continous phase plot.

* Enabled ``plot_bode`` to deal with system containing time delays.

* Enabled panel's interactive applications to render Latex labels on widgets
  when served on a new window.

* Fixed bug with evaluation of user-defined python's function.

* Fixed bug with labels of ``plot_implicit``.

* Fixed bug with labels of ``plot_piecewise``.

* Fixed bug with difficult to render labels on Matplotlib. If Matplotlib
  detects an error while parsing legend's entries, the plot won't show
  the legend.

* Fixed bug with ``plot_bode_phase`` when ``phase_units="deg"`` and
  ``unwrap=True``.

* Added settings for bode plot's ``phase_unit`` and ``freq_unit`` to the
  ``defaults`` submodule.

* Fixed bug with title of Bode plots.

* Fixed title of ``plot_step_response``.

* Implemented workaround for holoviz's Panel interactive applications
  to be able to work with a currently open bug.


v3.0.0
======

* Introducing the **graphics module**, which aims to solve the following
  problems about ordinary plotting function (whose name's start
  with ``plot_``):

  1. Some functions perform too many tasks, making them difficult and
     confusing to use.
  2. The documentation is difficult to maintain because many keywords arguments
     are repeated on all plotting functions.
  3. The procedures to combine multiple plots together is far from ideal.

  The *graphics module* implements new functions into appropriate submodules.
  Each function solves a very specific task and is able to plot only one
  symbolic expression. Each function returns a list containing one or
  more data series, depending on the required visualization.
  In order to render the data series on the screen, they must be passed into
  the ``graphics`` function. Plenty of examples about its usage are available
  on the documentation.

* Added ``arrow_2d`` to ``spb.graphics.vectors`` in order to plot a single
  arrow in a two-dimensional space.

* Reorganized old plotting functions (whose name's start with ``plot_``)
  into a new submodule: ``spb.plot_functions``. In particular:

  * Deprecated ``spb.vectors``.  Its content is now into
    ``spb.plot_functions.vectors``.
  * Deprecated ``spb.functions``. Its content is now into
    ``spb.plot_functions.functions_2d`` and
    ``spb.plot_functions.functions_3d``.
  * Deprecated ``spb.control``. Its content is now into
    ``spb.plot_functions.control``.
  * Deprecated ``spb.ccomplex.complex``. Its content is now into
    ``spb.plot_functions.complex_analysis``.
  * Deprecated ``spb.ccomplex.wegert``. Its content is now into ``spb.wegert``.

  Under the hood, many of these plotting functions now uses the
  *graphics module*.

* Bug fix on ``MatplotlibBackend`` about updating y-axis limits.
  Thanks to `Chrillebon  <https://github.com/Chrillebon>`_ for the fix.

* Improved performance of the evaluation with ``adaptive=False`` (the default
  one). Removed ``np.vectorize`` when the evaluation module is NumPy/Scipy in
  order to take full advantage of Numpy's vectorized operations.

* Keyword argument ``is_point`` now has an alias: ``scatter``. Setting
  ``scatter=True`` will render a sequence of points as a scatter rather than
  a line.

* Improved warning messages to provide more useful information.

* Fixed import-related bug with older versions of SymPy.


v2.4.3
======

* Bug fix: set axis scales only if the appropriate keyword arguments are
  provided. This allows to create symbolic plots with categorical axis.

* Fixed deprecation warning of one example using Holoviz panel and Bokeh
  formatters.

* Added new tutorial to documentation.

* Added the ``unwrap`` keyword argument to ``plot_bode`` in order to get a
  continous phase plot.


v2.4.2
======

* Fixed bug with renderers and the ``extend`` and ``append`` methods of
  plot objects.


v2.4.1
======

* Fixed bug with conda package.


v2.4.0
======

* Enabled interactive-widgets ``plotgrid``. In particular, this allows to
  create interactive widget plots with ``plot_bode`` and
  ``plot_riemann_sphere``.

* Enabled support for plotting applied undefined functions.

* Implemented parametric text for titles and axis labels.

* Implemented the ``exclude`` keyword argument for ``plot`` and
  ``plot_parametric``. It accepts a list of values at which a discontinuity
  will be introduced. This complementes the poles detection algorithm.

* Bug fixes

  * fixed bug with axis labels of ``plot_real_imag`` when creating contour
    plots.

  * fixed bug with colorbar label of 3d plots with lambda functions.


v2.3.0
======

* Improvements to the ``plot`` function:

  * Implemented reversed x-axis. Usually, a plot range is given with the
    form ``(symbol, min_val, max_val)``, with ``min_val`` on the left of
    the plot. If a range is given with ``(symbol, max_val, min_val)``, then
    the x-axis will be reversed.

  * The ``plot`` function is now able to show vertical lines at discontinuities
    when ``detect_poles="symbolic"``, at least for simple symbolic expressions.

* Introducing the ``Renderer`` class. Up to version `2.2.0`, all the rendering
  logic was located into each backend class, making it very difficult if not
  impossible to extend the capabilities for final users. From this
  version, each data series is going to be paired with an instance of
  ``Renderer``: users can create new data series and renderers. Then, by
  informing the backend of their existance, users can create new plot
  functions or modify the rendering of the old ones.

* Introducing the control module, which contains plotting functions for some
  of the common plots used in control system. This is an improved version of
  what is currently present on SymPy (version 1.12), because:

    * it allows to plot multiple systems simultaneously, making it easier to
      compare different transfer functions.
    * it works both on Matplotlib, Plotly and Bokeh.
    * it allows to create interactive-widgets plots, allowing the study of
      parametric systems.

  Thanks to all SymPy developers that worked on the
  ``sympy.physics.control.control_plots`` module.

  Further, it includes ``plot_nyquist`` and ``plot_nichols``, which currently
  only works with Matplotlib. Their underlying rendering logic comes from the
  `python-control package <https://github.com/python-control/python-control>`_.
  Huge thanks to all the ``python-control`` developers that worked on those
  functions.

* Upgrading dependency of Holoviz's Panel to version greater or equal
  than 1.0.0.

* Bug fixes:

  * complex surfaces can now be plotted with ``plot_contour``.

  * custom rendering keyword arguments can be passed to ``plot_geometry``.


v2.2.0
======

* Improved complex domain coloring and added ``plot_riemann_sphere``.

* Added ``imagegrid`` keyword argument to ``plotgrid``.

* Enabled support for plotting indexed objects.

* Implemented ``colorbar`` keyword argument to show/hide colorbar.

* Implemented ``show_in_legend`` keyword argument to show/hide a specific
  series on the legend of a plot.

* Improved logic about legend.

* Fixed bug with ``PlotlyBackend`` when creating 3D analytic landscapes.


v2.1.0
======

* Improved ``plot_implicit``:

  * implemented the ``color`` keyword argument, to set the color of line or
    region being plotted.

  * implemented the ``border_color`` keyword argument: this will add a new
    data series to represent a limiting border when plotting inequalities
    (``>, >=, <, <=``).

  * reduced the number of discretization points from 1000 to 100. Thanks to
    improvements to the backend and data generation, same quality can be
    achieved much more efficiently.

* Improved ``plot_complex`` and domain coloring plots:

  * User can now set a different colormap.

  * Added new coloring schemes.

  * User can change the label of the colorbar.

* Bug fixes on ``MatplotlibBackend``:

  * fixed bad behavior when plotting filled geometries with interactive
    widgets.

  * fixed missing legend entries when combining different types of plots.

* Bug fixes on ``K3DBackend``:

  * it is now possible to plot 3D quivers with custom colormaps.

  * fixed color bar visibility when plotting 3D complex plots.

* ``MatplotlibBackend`` and ``PlotlyBackend`` are now able to visualize legend
  entries for 3D surface plots using solid colors.


v2.0.2
======

* Bug fix: included static files necessary for serving interactive application
  on a new browser window.
* Improved documentation.


v2.0.1
======

* Improved import statements on ``spb.interactive.ipywidgets``: now, this
  module can be used even when only matplotlib and ipywidgets are installed.


v2.0.0
======

If you are upgrading from a previous version, you should run the following
code to load the new configuration settings:

.. code-block:: python

   from spb.defaults import reset
   reset()

* Breaking changes:

  * Refactoring of ``*Series`` classes. All ``*InteractiveSeries`` classes have
    been removed. The interactive functionalities have been integrated on
    regular ``*Series``. This greatly simplifies the code base, meaning bug
    fixes should take less time to implement.

  * Refactoring of ``iplot`` to take into account the aforementioned
    changes. In particular, interactive widget plots are now tighly integrated
    into the usual plotting functions. This improves user experience and
    simplifies the code base.

  * The ``spb.interactive.create_series`` function has been removed.

* Changed the default evaluation algorithm to a uniform sampling strategy,
  instead of the adaptive algorithm. The latter is still
  available, just set ``adaptive=True`` on the plotting functions that support
  it. The motivation behind this change is that the adaptive algorithm is
  usually much slower to produce comparable results: by default, the uniform
  sampling strategy uses 1000 discretization points over the specified range
  (users can increase it or decrease it), which is usually enough to smoothly
  capture the function.

  It also simplifies the dependencies of the module: now, the adaptive
  algorithm is not required by the plotting module to successfully visualize
  symbolic expressions, hence it is not installed. If users need the adaptive
  algorithm, they'll have to follow the
  `adaptive module installation instructions <https://github.com/python-adaptive/adaptive>`_.

* Improved support for plotting summations.

* Implemented wireframe lines for 3D complex plots.

* Interactive widget plots.

  * Users can now chose the interactive module to be used:

    * ``ipywidgets``: new in this release. It is the default one.
    * ``panel``: the same, old one.

    Please, read the documentation about the interactive sub-module to learn
    more about them, and how to chose one or the other.

  * Implemented the ``template`` keyword argument for interactive widget plots
    with Holoviz's Panel and ``servable=True``: user can further customize the
    layout of the web application, or can provide their own Panel's templates.

  * The module is now fully interactive. Thanks to the ``prange`` class, it is
    possible to specify parametric ranges. Explore the examples in the module
    documentation to find out how to use it.

* ``color_func`` now support symbolic expressions.

* ``line_color`` and ``surface_color`` are now deprecated in favor of
  ``color_func``.

* ``plot_implicit``:

  * now it supports interactive-widget plots, when ``adaptive=False``.

  * not it support ``rendering_kw`` for plots created with ``adaptive=True``.

  * improved logic dealing with legends. When plotting
    multiple regions, rectangles will be visible on the legend. When plotting
    multiple lines, lines will be visible on the legend.

* Removed ``tutorials`` folder containing Jupyter notebooks. The documentation
  contains plently of examples: the notebooks were just reduntant and
  difficult to maintain.

* ``MatplotlibBackend``: implemented support for ``ipywidgets``.


* ``PlotlyBackend``:

  * fixed bug with interactive update of lines.

  * implemented support for ``ipywidgets``.

* ``BokehBackend``:

  * improved support for Bokeh 3.0.
  * removed ``update_event`` because it became a redundant feature now that
    the module is fully parametric.

* ``plot_contour``: added the ``clabels`` keyword argument to show/hide
  contour labels.

* Documentation is now able to show interactive widget plots with K3D-Jupyter.

* conda package is now built and made available through the conda-forge
  channel. This greatly simplify the workflow and should allow an easier
  installation with conda.


v1.6.7
======

* Fixed bugs related to evaluation with complex numbers and parameters.
  Thanks to `Michele Ceccacci  <https://github.com/michelececcacci>`_ for the
  fix!


v1.6.6
======

* Fixed bug with ``PlaneSeries``'s data generation. Thanks to `Crillebon <https://github.com/Chrillebon>`_ for the fix!


v1.6.5
======

* Refinements and bug correction on ``plot_polar``: now it supports both
  cartesian and polar axis. Set ``polar_axis=True`` to enable polar axis.

* Added polar axis support to ``plot_contour`` with ``MatplotlibBackend``.

* 3D complex plots uses an auto aspect ratio by default.


v1.6.4
======

* ``MatplotlibBackend``:

  * improved ``aspect`` logic. It is now able to support the new values for
    3D plots for Matplotlib>=3.6.0.

  * exposed the ``ax`` attribute to easily retrieve the plot axis.

* Added ``camera`` keyword arguments to backends in order to set the 3D view
  position. Refer to each backend documentation to get more information about
  its usage.

* improved documentation.


v1.6.3
======

* Fixed bug with ``plot_geometry`` and 3D geometric entities.

* Added tutorial about combining plots together.


v1.6.2
======

* Added ``plot3d_list`` function to plot list of coordinates on 3D space.

* Changed value to default setting:
  ``cfg["matplotlib"]["show_minor_grid"]=False``. Set it to ``True`` in order
  to visualize minor grid lines.

* Improved documentation.

* Enabled ``color_func`` keyword argument on ``plot_vector``.

* ``PlotlyBackend``:

  * if the number of points of a line is greater than some threshold, the
    backend will switch to ``go.Scattergl``. This improves performance.

  * Fixed bug with interactive widget contour plot and update of colorbar.

* ``MatplotlibBackend`` can now combine 3d plots with contour plots.

* Fixed bug with addition of interactive plots.


v1.6.1
======

* Improvements to documentation. In particular, ReadTheDocs now shows pictures
  generated with ``PlotlyBackend``, ``K3DBackend`` as well as interactive
  plots with widgets.

* Default settings:

  * Changed ``cgf["interactive"]["theme"]`` to ``"light"``: interactive plots
    served on a new browser window will use a light theme.

  * Changed ``cgf["bokeh"]["update_event"]`` to ``False``: Bokeh won't update
    the plot with new data as dragging or zooming operations are performed.

  * Added new option ``cgf["k3d"]["camera_mode"]``.


* Improvements to ``MatplotlibBackend``:

  * Added label capability to ``plot_implicit``.

  * ``show()`` method now accepts keyword arguments. This is useful to detach
    the plot from a non-interactive console.

* Added ``dots`` keyword argument to ``plot_piecewise`` to choose wheter to
  show circular markers on endpoints.

* Fixed bug with plotting 3D vectors.


v1.6.0
======

* Added new plotting functions:

  * ``plot3d_revolution`` to create surface of revolution.

  * ``plot_parametric_region``, still in development.

* ``MatplotlibBackend``:

  * Fixed bug with colormaps and normalization.

  * Improved update speed when dealing with parametric domain coloring plots.

* Improved ``zlim`` support on ``K3DBackend`` for interactive widget plots.

* Fixed bug with parametric interactive widget plots and ``PlotlyBackend``: the
  update speed is now decent.

* Series:

  * Moved ``LineOver1DRangeSeries._detect_poles`` to ``_detect_poles_helper``.

  * ``plot_complex`` and ``plot_real_imag``: the input expression is no longer
    wrapped by symbolic ``re()`` or ``im()``. Instead, the necessary processing
    is done on the series after the complex function has been evaluated. This
    improves performance.

* ``Parametric2DLineSeries`` now support ``detect_poles``.

* Implemented support for ``color_func`` keyword argument on ``plot_list``
  and ``plot_complex_list``.

* Added ``extras_require`` to ``setup.py``:

  * by default, ``pip install sympy_plot_backends`` will install only the
    necessary requirements to get non-interactive plotting to work with
    Matplotlib.
  * use ``pip install sympy_plot_backends[all]`` to install all other packages:
    panel, bokeh, plotly, k3d, vtk, ...

* Documentation:

  * Improved examples.

  * Added examples with ``PlotlyBackend``.


v1.5.0
======

* Implemented the ``plot3d_spherical`` function to plot functions in
  spherical coordinates.

* Added the ``wireframe`` option to ``plot3d``,
  ``plot3d_parametric_surface`` and ``plot3d_spherical`` to add grid lines
  over the surface.

* Fixed bug with ``plot3d`` and ``plot_contour`` when dealing with instances
  of ``BaseScalar``.

* Added ``normalize`` keyword argument to ``plot_vector`` and
  ``plot_complex_vector`` to visualize quivers with unit length.

* Improve documentation of ``plot_vector`` and ``plot_complex_vector``.

* Improved test coverage on complex and vector plotting functions.

* Improvements on ``PlotlyBackend``:

  * it is now be able to plot more than 14 2d/3d parametric lines when
    ``use_cm=False``.

  * improved logic to show colorbars on 3D surface plots.

  * added support for custom aspect ratio on 3D plots.

* Improved support for ``xlim``, ``ylim``, ``zlim`` on ``K3DBackend``.

* Series:

  * Fixed bug with uniform evaluation while plotting numerical functions.

  * Fixed bug with ``color_func``.

  * Added transformation keyword arguments ``tx, ty, tz`` to parametric series.

* Breaks:

  * Inside ``plot_parametric`` and ``plot3d_parametric_line``, the ``tz``
    keyword argument has been renamed to ``tp``.

  * Removed Mayavi from setup dependencies. Mayavi is difficult to install:
    can't afford the time it requires for proper setup and testing.
    ``MayaviBackend`` is still available to be used "as is".


v1.4.0
======

* Reintroduced ``MayaviBackend`` to plot 3D symbolic expressions with Mayavi.
  Note that interactive widgets are still not supported by this backend.

* ``plot_contour`` is now able to create filled contours or line contours on
  backends that supports such distinction. Set the ``is_filled`` keyword
  argument to choose the behaviour.

* Implemented interactive widget support for ``plot_list``.

* Implemented back-compatibility-related features with SymPy.

* Fixed bugs with ``PlaneSeries``:

  * Data generation for vertical planes is now fixed.
  * ``K3DBackend`` is now able to plot this series.
  * Similar to other 3D surfaces, planes will be plotted with a solid color.

* Fixed bug with ``Vector3DSeries``: the discretized volume is now created with
  Numpy's ``meshgrid`` with ``indexing='ij'``. This improves the generation of
  3D streamlines.

* Fixed bug with ``plot3d`` and ``plot_contour``: when ``params`` is provided
  the specified backend will be instantiated.

* Fixed bug with ``K3DBackend`` and ``plot3d_implicit``.


v1.3.0
======

* Added support for plotting numerical vectorized functions. Many of the
  plotting functions exposed by this module are now able to deal with both
  symbolic expressions as well as numerical functions. This extends the scope
  of this module, as it is possible to use it directly with numpy and lambda
  functions. For example, the following is now supported:

  .. code-block:: python

       import numpy as np
       plot(lambda t: np.cos(x) * np.exp(-x / 5), ("t", 0, 10))

* Added support for vector from the ``sympy.physics.mechanics`` module in the
  ``plot_vector`` function.

* Implemented keyword argument validator: if a user writes a misspelled keyword
  arguments, a warning message will be raised showing one possible alternative.


v1.2.1
======

* Added ``used_by_default`` inside default options for adaptive
  algorithm. This let the user decide wheter to use adaptive algorithm or
  uniform meshing by default for line plots.

* Fix the axis labels for the ``plot_complex_vector`` function.

* Improved a few examples in the docstring of ``plot_vector`` and
  ``plot_complex_vector``.

* Fixed bug with interactive update of ``plot_vector`` inside
  ``MatplotlibBackend``.

* Improvements to the code in preparation for merging this module into Sympy:

  * Small refactoring about the label generation: previously, the string and
    latex representations were generated at different times and in different
    functions. Now, they are generated simultaneously inside the ``__init__``
    method of a data series.

  * Changes in names of functions that are meant to remain private:

    * ``adaptive_eval`` -> ``_adaptive_eval``.
    * ``_uniform_eval`` -> ``_uniform_eval_helper``
    * ``uniform_eval`` -> ``_uniform_eval``
    * ``_correct_size`` -> ``_correct_shape``
    * ``get_points`` -> ``_get_points``


v1.2.0
======

* Replaced the ``line_kw``, ``surface_kw``, ``image_kw``, ``fill_kw`` keyword
  arguments with ``rendering_kw``. This simplifies the usage between different
  plotting functions.

* Plot functions now accepts a new argument: ``rendering_kw``, a dictionary
  of options that will be passed directly to the backend to customize the
  appearance. In particular:

  * Possibility to plot and customize multiple expressions with a single
    function call. For example, for line plots:

    .. code-block:: python

       plot(
         (expr1, range1 [opt], label1 [opt], rendering_kw1 [opt]),
         (expr2, range2 [opt], label2 [opt], rendering_kw2 [opt]),
         **kwargs
       )

  * Possibility to achieve the same result using the ``label`` and
    ``rendering_kw`` keyword arguments by providing lists of elements (one
    element for each expression). For example, for line plots:

    .. code-block:: python

       plot(expr1, expr2, range [opt],
           label=["label1", "label2"],
           rendering_kw=[dict(...), dict(...)],
           **kwargs
       )

* Interactive submodule:

  * Fixed bug with ``spb.interactive.create_widgets``.

  * Integration of the interactive-widget plot ``iplot`` into the most
    important plotting functions. To activate the interactive-widget plot
    users need to provide the ``params`` dictionary to the plotting function.
    For example, to create a line interactive-widget plot:

    .. code-block:: python

         plot(cos(u * x), (x, -5, 5), params={u: (1, 0, 2)})

* Series:

  * Fixed a bug with line series when plotting complex-related function
    with ``adaptive=False``.

  * Fixed bug with ``lambdify`` and ``modules="sympy"``.

  * Fixed bug with the number of discretization points of vector series.

  * Enabled support for Python's built-in ``sum()`` function, which can now
    be used to combine multiple plots.

* Backends:

  * Fixed a bug with ``MatplotlibBackend`` and string-valued color maps.

  * Fixed a bug with ``BokehBackend`` about the update of quivers color when
    using ``iplot``.

* Updated tutorials and documentation.


v1.1.7
======

* Fixed bug with ``plot_complex_list``.
* Added new tutorial about singularity-dections.


v1.1.6
======

* Fixed bug with ``label`` keyword argument.
* Added error message to ``plot3d``.
* Updated documentation.


v1.1.5
======

* Implemented ``line_color`` and ``surface_color``: this plotting module should
  now be back-compatible with the current ``sympy.plotting``.


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
