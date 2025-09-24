Backends
--------

Supported plotting libraries
============================

In the context of this plotting module, a backend represents the machinery
that allows symbolic expressions to be visualized with a particular plotting
library. 4 plotting libraries are supported. The use case is summarized in
the following table.

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
  performance: the number of discretization points can be increased
  significantly, thus obtaining smoother plots. It can only be used with
  Jupyter Notebook, whereas the other backends can also be used with IPython
  or a simple Python interpreter. This backends use an aspect ratio of 1 on
  all axis: it doesn't scale the visualization. What you see is the object as
  you would see it in reality.

The appropriate backend for a particular job can be selected at runtime by
setting the keyword argument ``backend=`` in the function call. We can also
set the default backends for 2D and 3D plots in a configuration file by using
the :doc:`Defaults module <../defaults>` .

Shuld we need to retrieve the actual figure from a backend we can use the
``.fig`` attribute. Should we need to perform some other processing to the
figure before showing it to the screen, we can use the ``hooks`` attribute.

Please, read the documentation associated to each backend to find out more
customization options.

.. toctree::
   :maxdepth: 1

   matplotlib.rst
   bokeh.rst
   plotly.rst
   k3d.rst
   utils.rst


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
  using Holoviews, work-arounds can be implemented more easily.

