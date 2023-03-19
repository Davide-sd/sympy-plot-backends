==========
 Changelog
==========

v2.0.0
======

If you are upgrading from a previous version, you should run the following
code to load the new configuration settings:

```
from spb.defaults import reset
reset()
```

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
    * ``panel``: the old one, but probably the most feature-rich.

    Please, read the documentation about the Interactive module to learn more.

  * Implemented the ``template`` keyword argument for interactive widget plots
    with Holoviz's Panel and ``servable=True``: user can further customize the
    layout of the web application, or can provide their own Panel's templates.

* ``MatplotlibBackend``:

  * implemented support for ``ipywidgets``.


* ``PlotlyBackend``:

  * fixed bug with interactive update of lines.

  * implemented support for ``ipywidgets``.

* ``BokehBackend``: improved support for Bokeh 3.0.

* ``plot_contour``: added keyword argument to show/hide contour labels:
    ``clabels=True/False``.


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
