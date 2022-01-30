Backends
--------

This module allows the user to chose between 4 different backends.

.. toctree::
   :maxdepth: 2

   matplotlib.rst
   bokeh.rst
   plotly.rst
   k3d.rst


The following table summarize the use case for the backends.

+------------------------+-----------+-------+--------+------+
|                        | Matplolib | Bokeh | Plotly |  K3D |
+========================+===========+=======+========+======+
|           2D           |     Y     |   Y   |    Y   |   N  |
+------------------------+-----------+-------+--------+------+
|           3D           |     Y     |   N   |    Y   |   Y  |
+------------------------+-----------+-------+--------+------+
|      Latex Support     |     Y     |   N   |    Y   |   Y  |
+------------------------+-----------+-------+--------+------+
|       Jupyter NB       |     Y     |   Y   |    Y   |   Y  |
+------------------------+-----------+-------+--------+------+
|   Python Interpreter   |     Y     |   Y   |    Y   |   N  |
+------------------------+-----------+-------+--------+------+

In particular:

* Matplotlib is a good general backend supporting all kinds of plots. However,
  it lacks interactivity. Even with
  `ipympl <https://github.com/matplotlib/ipympl>`_, interactivity falls behind
  in comparison to other plotting libraries. More so, ipympl causes the
  rendering of 3D plots to be very slow.
* Plotly is another general backend supporting many kinds of plots.
  Interactivity and data exploration are great, however it does have a few
  limitations:

  * Slower rendering than all other backends because it adds HTML elements to
    the DOM of the notebook.
  * Lack of gradient lines.
  * Doesn't support ``plot_implicit``.
  * No wireframe support for 3D plots.

* Bokeh: interactivity and data exploration are great. It supports auto-update
  while panning the plot (only works with 2D lines), however:

   * Doesn't support ``plot_implicit``.
   * Lack of proper contour functionalities: the actual implementation
     approximates a contour plot.

* K3D only supports 3D plots but, compared to Matplotlib, it offers amazing 3D
  performance: we can increase significantly the number of discretization
  points obtaining smoother plots. It can only be used with Jupyter Notebook,
  whereas the other backends can also be used with IPython or a simple Python
  interpreter. This backends use an aspect ratio of 1 on all axis: it doedn't
  scale the visualization. What you see is the object as you would see it in
  reality.

It is interesting to note that when in comes to 3D plots, Plotly and K3D are
able to deal with NaN values, whereas Matplolib is not. For example:

* ``plot3d(sqrt(x * y), (x, -3, 3), (y, -3, 3), backend=MB, n=50)`` don't work
* ``plot3d(sqrt(x * y), (x, -3, 3), (y, -3, 3), backend=PB, n=50)`` works fine
* ``plot3d(sqrt(x * y), (x, -3, 3), (y, -3, 3), backend=KB, n=50)`` works fine

We can choose the appropriate backend for our use case at runtime by setting the keyword argument ``backend=`` in the function call. We can also
set the default backends for 2D and 3D plots in a configuration file by using
the :doc:`Defaults module <../defaults>` .

Please, read the documentation associated to each backend to find out more
customization options.


About the Implementation
========================

Why different backends inheriting from the ``Plot`` class? Why not using
something like `holoviews <https://holoviews.org/>`_, which allows to plot
numerical data with different plotting libraries using a common interface?
In short:

* Holoviews only support Matplotlib, Bokeh, Plotly. This would make
  impossible to add support for further libraries, such as K3D, ...
* Not all needed features might be implemented on Holoviews. Think for example
  to plotting a gradient-colored line. Matplotlib and Bokeh are able to
  visualize it correctly, Plotly doesn't support this functionality. By not
  using Holoviews, we can more easily implement some work around.


Plot
====

.. module:: spb.backends.base_backend

.. autoclass:: Plot

.. autofunction:: spb.backends.base_backend.Plot.append

.. autofunction:: spb.backends.base_backend.Plot.extend

.. autoattribute:: spb.backends.base_backend.Plot.colorloop

.. autoattribute:: spb.backends.base_backend.Plot.colormaps

.. autoattribute:: spb.backends.base_backend.Plot.cyclic_colormaps

.. autoattribute:: spb.backends.base_backend.Plot.fig
