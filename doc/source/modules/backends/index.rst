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
|      Save Picture      |     Y     |   Y   |    Y   |   Y  |
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
explore this functionality in REF TO TUTORIAL.

Whenever a plot function is called, it will instantiate a backend. We can set
any of the following keyword arguments in the function call: they will be passed
to the backend for customizing the appearance. Since each plotting library is
unique, some of these options may not be supported by a specific backend
(or have not been implemented yet):

+---------------+-----------+-------+--------+-----+-----------------+
|  keyword arg  | Matplolib | Bokeh | Plotly | K3D |       Type      |
+===============+===========+=======+========+=====+=================+
|     xlim      |     Y     |   Y   |    Y   |  N  | (float, float)  |
+---------------+-----------+-------+--------+-----+-----------------+
|     ylim      |     Y     |   Y   |    Y   |  N  | (float, float)  |
+---------------+-----------+-------+--------+-----+-----------------+
|     zlim      |     Y     |   N   |    Y   |  N  | (float, float)  |
+---------------+-----------+-------+--------+-----+-----------------+
|    xscale     |     Y     |   Y   |    Y   |  N  |       str       |
+---------------+-----------+-------+--------+-----+-----------------+
|    yscale     |     Y     |   Y   |    Y   |  N  |       str       |
+---------------+-----------+-------+--------+-----+-----------------+
|    zscale     |     Y     |   N   |    Y   |  N  |       str       |
+---------------+-----------+-------+--------+-----+-----------------+
|     grid      |     Y     |   Y   |    Y   |  Y  |     boolean     |
+---------------+-----------+-------+--------+-----+-----------------+
|  axis_center  |     Y     |   N   |    N   |  N  |    str / tuple  |
+---------------+-----------+-------+--------+-----+-----------------+
|    aspect     |     Y     |   Y   |    Y   |  N  |    str / tuple  |
+---------------+-----------+-------+--------+-----+-----------------+
|     size      |     Y     |   Y   |    Y   |  Y  | (float, float)  |
+---------------+-----------+-------+--------+-----+-----------------+
|     title     |     Y     |   Y   |    Y   |  Y  |       str       |
+---------------+-----------+-------+--------+-----+-----------------+
|    xlabel     |     Y     |   Y   |    Y   |  Y  |       str       |
+---------------+-----------+-------+--------+-----+-----------------+
|    ylabel     |     Y     |   Y   |    Y   |  Y  |       str       |
+---------------+-----------+-------+--------+-----+-----------------+
|    zlabel     |     Y     |   N   |    Y   |  Y  |       str       |
+---------------+-----------+-------+--------+-----+-----------------+

For example, while SymPy's default backend (Matplotlib) is implemented to mimic
hand-plotted 2D charts, that is the horizontal and vertical axis are not
necessarely fixed to the bottom-side and left-side of the plot respectively
(we can specify their location with `axis_center`), this feature is not
implemented on Bokeh and Plotly.

Please, read the documentation associated to each backend to find out what these
options do. We will also use some of them in the tutorial section.


