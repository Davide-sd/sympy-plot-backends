Backends
--------


.. toctree::
   :maxdepth: 1

   plot.rst
   matplotlib.rst
   bokeh.rst
   plotly.rst
   k3d.rst

This module allows the user to chose between 4 different backends.
The use case is summarized in the following table.

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
|   Interactive Widgets  |     Y     |   Y   |    Y   |   Y  |
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
  * Contour capabilities are limited in comparison to Matplotlib.

* Bokeh: interactivity and data exploration are great. It supports auto-update
  while panning the plot (only works with 2D lines), however:

   * Doesn't support ``plot_implicit``.
   * Lack of proper contour functionalities: the actual implementation
     approximates a contour plot.

* K3D only supports 3D plots but, compared to Matplotlib, it offers amazing 3D
  performance: we can increase significantly the number of discretization
  points obtaining smoother plots. It can only be used with Jupyter Notebook,
  whereas the other backends can also be used with IPython or a simple Python
  interpreter. This backends use an aspect ratio of 1 on all axis: it doesn't
  scale the visualization. What you see is the object as you would see it in
  reality.

We can choose the appropriate backend for our use case at runtime by setting the keyword argument ``backend=`` in the function call. We can also
set the default backends for 2D and 3D plots in a configuration file by using
the :doc:`Defaults module <../defaults>` .

Please, read the documentation associated to each backend to find out more
customization options.
