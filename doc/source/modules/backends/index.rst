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
* Plotly is another general backend supporting many kinds of plot.
  Interactivity and data exploration are great, however it does have a few
  limitations:

  * Slower rendering than all other backends because it adds HTML elements to
    the DOM of the notebook.
  * Lack of gradient lines.
  * Generally inferior when plotting implicit expression in comparison to
    Matplotlib. Also, it can be really slow when plotting multiple implicit
    expressions simultaneously.
  * No wireframe support for 3D plots, which could lead to difficult to
    understand plots.

* Bokeh: interactivity and data exploration are great. It supports auto-update
  while panning the plot (only works with 2D lines), however:

   * Generally inferior when plotting implicit expression in comparison to
     Matplotlib.
   * Lack of contour plot.
   
* K3D only supports 3D plots but, compared to Matplotlib, it offers amazing 3D
  performance: we can increase significantly the number of discretization
  points obtaining smoother plots. It can only be used with Jupyter Notebook,
  whereas the other backends can also be used with IPython or a simple Python
  interpreter. This backends use an aspect ratio of 1 on all axis: it doedn't
  scale the visualization. What you see is the object as you would see it in
  reality.

We can choose the appropriate backend for our use case at runtime. We can also
set the default backends for 2D and 3D plots in a configuration file. We will
explore this functionality in tutorial :ref:`tut3`.

Please, read the documentation associated to each backend to find out more
customization options.

Base Class
==========

.. module:: spb.backends.base_backend

.. autoclass:: Plot

.. autofunction:: spb.backends.base_backend.Plot.append

.. autofunction:: spb.backends.base_backend.Plot.extend

.. autoattribute:: spb.backends.base_backend.Plot.fig

