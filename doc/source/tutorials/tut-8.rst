8 - Creating Custom Plots
-------------------------

Sometimes, the functions exposed by Sympy's plotting module are not enough to
accomplish our visualization objectives. If that's the case, we can either:

1. ``lambdify`` the symbolic expressions and evaluate it numerically. However,
   this process is manually intensive.
2. If the expressions can be plotted by the common plotting functions (``plot``,
   ``plot3d``, ``plot_parametric``, ...), then we can use the ``get_plot_data``
   function, which automate the _lambdifying_ process. This function accepts
   the same arguments of the aforementioned plotting functions, therefore it is
   really easy to get the numerical data we are interested in.

Once we have the numerical data, we can use our preferred plotting library.
If we are lucky enough, we can also:

1. use one of the plotting functions as a starting point;
2. extract the numerical data with the ``get_plot_data`` function;
3. extract the plot object associated to the plotting library;
4. use the appropriate command of the plotting library to add new data to the
   plot.

Let's see a few examples.

Example 1 - Editing and Adding Data
===================================


The current backends are able to plot lines, gradient lines, contours,
quivers, streamlines. However, they are not able to plot things like
*curve fills*, bars, etc.

In this example we are going to illustrate a procedure that can be used
to further customize the plot. Note that the procedure depends on the
backend we are going to use, because we are going to use
backend-specific commands. In the following, we are going to use
``PlotlyBackend``. For other backends, the procedure might need to be
adjusted.

.. code:: ipython3

    from sympy import *
    from spb import *
    from spb.backends.matplotlib import MB
    from spb.backends.plotly import PB

Let's say we would like to plot on the same figure:

* a normal distribution filled to the horizontal axis.
* a dampened oscillation.
* bars following an exponential decay at integer locations.

.. code:: ipython3

    x, mu, sigma = symbols("x, mu, sigma")
    expr1 = 1 / (sigma * sqrt(2 * pi)) * exp(-((x - mu) / sigma)**2 / 2)
    expr2 = cos(x) * exp(-x / 6)
    expr3 = exp(-x / 5)
    display(expr1, expr2, expr3)

:math:`\frac{\sqrt{2} e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}}}{2 \sqrt{\pi} \sigma}`

:math:`e^{- \frac{x}{6}} \cos{\left(x \right)}`

:math:`e^{- \frac{x}{5}}`

We start by plotting the first two expressions, as the third one
requires a different approach:

.. code:: ipython3

    p = plot(
        (expr1.subs({sigma: 0.8, mu: 5}), "normal"), 
        (expr2, "oscillation"),
        (x, 0, 10), backend=PB)

.. raw:: html
	:file: figs/tut-8/fig-01.html

Now, we'd like to fill the first curve. First, we extract the figure
object; then we set the necessary attribute to get the job done:

.. code:: ipython3

    f = p.fig
    f.data[0]["fill"]="tozerox"
    f

.. raw:: html
	:file: figs/tut-8/fig-02.html

At this point we have to convert ``expr3`` to numerical data. We can do
it with ``get_plot_data``, which requires the same arguments as the
``plot`` function, namely ``(expr, range, label [optional], **kwargs)``:

.. code:: ipython3

    xx, yy = get_plot_data(exp(-x / 5), (x, 0, 10), only_integers=True)
    print(xx)
    print(yy)

.. parsed-literal::

    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
    [1.         0.81873075 0.67032005 0.54881164 0.44932896 0.36787944
    0.30119421 0.24659696 0.20189652 0.16529889 0.13533528]

Now that we have generated the numerical values at integer locations, we
can add the bars with the appropriate command:

.. code:: ipython3

    import plotly.graph_objects as go
    import numpy as np
    
    f.add_trace(go.Bar(x=xx, y=yy, width=np.ones_like(xx) / 2, name="bars"))
    f

.. raw:: html
	:file: figs/tut-8/fig-03.html

Thatâs it, job done.


Example 2
=========

The backends are unable to mix 2D and 3D data series. But what if we
would like to plot a contour into a 3D plot?

Letâs say weâd like to explore the following vector field,
:math:`\vec{F}(x, y, z) = (\cos{(z)}, y, x)`, in the rectangular volume
limited by :math:`-5 \le x \le 5, \, -5 \le y \le 5, \, -5 \le z \le 5`.
We are going to plot the contours of the magnitude of the vector field
over 3 orthogonal planes, as well as quivers over a plane normal to the
y-direction.

.. code:: ipython3

    x, y, z = symbols("x:z")
    v = Matrix([cos(z), y, x])
    
    # magnitudes of the vector field over 3 orthogonal planes
    mag_func = lambda vec: sqrt(sum(t**2 for t in vec))
    mag = mag_func(v)
    m1 = mag.subs(x, 5)
    m2 = mag.subs(y, 5)
    m3 = mag.subs(z, 5)
    display(mag, m1, m2, m3)

:math:`\sqrt{x^{2} + y^{2} + \cos^{2}{\left(z \right)}}`

:math:`\sqrt{y^{2} + \cos^{2}{\left(z \right)} + 25}`

:math:`\sqrt{x^{2} + \cos^{2}{\left(z \right)} + 25}`

:math:`\sqrt{x^{2} + y^{2} + \cos^{2}{\left(5 \right)}}`

Letâs extract the data of the magnitudes:

.. code:: ipython3

    # ranges
    rx = (x, -5, 5)
    ry = (y, -5, 5)
    rz = (z, -5, 5)
    # contour data: similarly to plot3d/plot_contour the parameters 
    # to get_plot_data follows (expr, range_x, range_y)
    xx1, yy1, zz1 = get_plot_data(m1, ry, rz)
    xx2, yy2, zz2 = get_plot_data(m2, rx, rz)
    xx3, yy3, zz3 = get_plot_data(m3, rx, ry)

Now, letâs extract the data of the sliced-vector field:

.. code:: ipython3

    xq1, yq1, zq1, uu, vv, ww = get_plot_data(v, rx, ry, rz, 
            slice=Plane((0, 0, 0), (0, 1, 0)))

Finally, we create the custom plot:

.. code:: ipython3

    %matplotlib widget
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    ax = plt.figure().add_subplot(projection='3d')
    
    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph
    ax.contourf(xx1, zz1, yy1, zdir='y', offset=5, cmap=cm.coolwarm)
    ax.contourf(zz2, xx2, yy2, zdir='x', offset=-5, cmap=cm.coolwarm)
    ax.contourf(xx3, yy3, zz3, zdir='z', offset=-5, cmap=cm.coolwarm)
    ax.quiver(xq1, yq1, zq1, uu, vv, ww, color="g", length=0.5, normalize=True)
              
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

.. figure:: figs/tut-8/fig-04.png
