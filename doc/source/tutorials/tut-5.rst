
5 - Creating custom plots
-------------------------

Sometimes, the functions exposed by Sympy's plotting module are not enough to
accomplish our visualization objectives. If that's the case, we can either:

1. ``lambdify`` the symbolic expressions and evaluate it numerically.
   However, this process is manually intensive.
2. If the expressions can be plotted by the common plotting functions
   (``plot``, ``plot3d``, ``plot_parametric``, ...), we can easily extract
   the numerical data by calling the ``get_data`` method of the interested
   series. Remember, the plot object can be indexed in order to access the
   series.

Once we have the numerical data, we can use our preferred plotting library.
If we are lucky enough, we can also:

1. use one of the plotting functions as a starting point;
2. extract the plot object associated to the plotting library;
3. use the appropriate command of the plotting library to add new data to
   the plot.

Example - Editing and Adding Data
=================================

The current backends are able to plot lines, gradient lines, contours,
quivers, streamlines. However, they are not able to plot things
like *curve fills*, bars, ...

In this example we are going to illustrate a procedure that can be used to
further customize the plot. Since we are going to use backend-specific
commands, the procedure is backend-specific. In the following, we are going
to use ``PlotlyBackend``. For other backends, the procedure might need to be
adjusted.

Let's say we would like to plot on the same figure:

* a normal distribution filled to the horizontal axis.
* a dampened oscillation.
* bars following an exponential decay at integer locations.

.. code-block:: python

   from sympy import *
   from spb import *
   x, mu, sigma = symbols("x, mu, sigma")
   expr1 = 1 / (sigma * sqrt(2 * pi)) * exp(-((x - mu) / sigma)**2 / 2)
   expr2 = cos(x) * exp(-x / 6)
   expr3 = exp(-x / 5)

We start by plotting the first two expressions, as the third one requires
a different approach:

.. code-block:: python

   p = plot(
       (expr1.subs({sigma: 0.8, mu: 5}), "normal"),
       (expr2, "oscillation"),
       (x, 0, 10), backend=PB)

.. raw:: html

   <iframe src="../_static/tut-4/plotly-4.html" height="500px" width="100%"></iframe>

Now, we'd like to fill the first curve. First, we extract the figure object;
then we set the necessary attribute to get the job done. Obviously, the
following procedure depends on the backend being used.

.. code-block:: python

   f = p.fig
   f.data[0]["fill"]="tozerox"
   f

.. raw:: html

   <iframe src="../_static/tut-4/f1.html" height="500px" width="100%"></iframe>

At this point we have to convert ``expr3`` to numerical data.
We can do it with the ``plot`` function:

.. code-block:: python

   p2 = plot(expr3, (x, 0, 10), adaptive=False, only_integers=True, show=False)
   # p2[0] is the data series representing our expression
   xx, yy = p2[0].get_data()
   print(xx)
   print(yy)

.. code-block:: text

   [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
   [1.         0.81873075 0.67032005 0.54881164 0.44932896 0.36787944
    0.30119421 0.24659696 0.20189652 0.16529889 0.13533528]

The advantage of this approach is that we can visualize the data
(if ``show=True``).

It is important to realize that the ``get_data()`` method of each series may
returns different elements. Read its documentation to find out what it returns:

.. code-block:: python

   help(p2[0].get_data)

.. code-block:: text

   Help on method get_data in module spb.series:

   get_data() method of spb.series.LineOver1DRangeSeries instance
       Return coordinates for plotting the line.

       Returns
       =======

       x: np.ndarray
           x-coordinates

       y: np.ndarray
           y-coordinates

       z: np.ndarray (optional)
           z-coordinates in case of Parametric3DLineSeries,
           Parametric3DLineInteractiveSeries

       param : np.ndarray (optional)
           The parameter in case of Parametric2DLineSeries,
           Parametric3DLineSeries or AbsArgLineSeries (and their
           corresponding interactive series).


Now that we have generated the numerical values at integer locations, we can
add the bars with the appropriate command:

.. code-block:: python

   import plotly.graph_objects as go
   import numpy as np
   f.add_trace(go.Bar(x=xx, y=yy, width=np.ones_like(xx) / 2, name="bars"))
   f

.. raw:: html

   <iframe src="../_static/tut-4/f2.html" height="500px" width="100%"></iframe>


That's it, job done.
