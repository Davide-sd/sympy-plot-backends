6 - Plotting Vector Fields
--------------------------

In this tutorial we are going to plot 2D and 3D vector fields with the
``vector_plot`` function.

.. code:: ipython3

    from spb.backends.plotly import PB
    from spb.backends.bokeh import BB
    from spb.backends.k3d import KB
    from spb import vector_plot
    from sympy import *
    init_printing(use_latex=True)

The plotting interface is basically the same of any other plotting function.
We need to specify:

* ``vector, range1, range2`` if we are plotting a single vector field.
* we can use tuples of the form:
  
  ``(vector1, range1, range2, label1), (vector2, range1, range2, label2), ...``
  to plot multiple vector fields simultaneously.

.. code:: ipython3

    help(vector_plot)


Plotting Vectors From sympy.vector
==================================

Let's create simple vector:

.. code:: ipython3

    from sympy.vector import CoordSys3D
    N = CoordSys3D("N")
    i, j, k = N.base_vectors()
    x, y, z = N.base_scalars()
    v1 = -sin(y) * i + cos(x) * j
    v1

.. figure:: ../_static/tut-6/equation-1.png

.. code:: ipython3

    vector_plot(v1, (x, -5, 5), (y, -5, 5), backend=PB, xlabel="x", ylabel="y",
        quiver_kw=dict(scale=0.3))

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-01.html" height="500px" width="100%"></iframe>

Here, we used Plotly. A few things to note:

* we need to specify the x-y labels.
* by default, the x and y axis will use an equal aspect ratio. We can disable
  it by setting the keyword argument ``aspect=None``.
* by default, a contour plot of the magnitude of the vector field is shown
  (more on this later).
* solid color is used for the arrows (or quivers), whose lengths are
  proportional to the local magnitude value. Note that Plotly doesn't support
  gradient coloring for quivers.
* We use the ``quiver_kw`` dictionary to control the appearance of the quivers,
  where we write the keyword arguments targeting the specific backend's quiver
  function. In this case, the quiver function is
  `Plotly's create_quiver <https://plotly.com/python/quiver-plots/>`_.
  Here, we used ``scale=0.3`` to set a decent size for the quivers.

Let's say we are not interested in showing the contour plot representing the
magnitude. We can disable it by setting the keyword argument ``scalar=None``:

.. code:: ipython3

    vector_plot(v1, (x, -5, 5), (y, -5, 5), backend=PB, xlabel="x", ylabel="y",
        quiver_kw=dict(scale=0.3), scalar=None)

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-02.html" height="500px" width="100%"></iframe>

Alternatively, we can set ``scalar`` to any scalar field, for example:

.. code:: ipython3

    x2, y2 = symbols("x, y")
    vector_plot(v1, (x, -5, 5), (y, -5, 5), backend=PB, xlabel="x", ylabel="y",
        quiver_kw=dict(scale=0.3), scalar=x2*y2)

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-03.html" height="500px" width="100%"></iframe>

Instead of visualizing quivers, we can plot streamlines by setting
``streamlines=True``:

.. code:: ipython3

    vector_plot(v1, (x, -5, 5), (y, -5, 5), backend=PB, xlabel="x", ylabel="y",
        streamlines=True, stream_kw=dict(density=2, arrow_scale=0.2))

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-04.html" height="500px" width="100%"></iframe>

A few things to note:

* computing and visualizing streamlines is usually computationally more
  expensive than plotting quivers, so the function may takes longer to produce
  the plot.
* We use the ``stream_kw`` dictionary to control the appearance of the
  streamlines, where we write the keyword arguments targeting the specific
  backend's quiver function. In this case, the quiver function is
  `Plotly's create_streamline <https://plotly.com/python/streamline-plots/>`_.
  Here, we increased the density and set an appropriate arrow size.


Quick Way to Plot Vectors
=========================

In the previous section we used ``sympy.vector`` module to define vectors.
However, if we are in a hurry we can avoid using that module, passing in to
the function a list containing the components of the vector. For example:

.. code:: ipython3

    x, y = symbols("x, y")
    vector_plot([-sin(y), cos(x)], (x, -5, 5), (y, -5, 5),
        backend=BB, xlabel="x", ylabel="y", quiver_kw=dict(scale=0.5))

.. raw:: html
	:file: figs/tut-6/fig-05.html


Here, we used Bokeh. A few things to note:

* by switching backend, the user experience will be overall quite similar.
  Unfortunately, it is hardly possible to have one-one-one correspondance
  between colors and color maps.
* Bokeh doesn't automatically support contour plots. If we zoom in, we will
  see that the scalar field is using square "pixels" to be rendered,
  leading to an unpleasant result. We can "fix" this problem by bumping up
  the number of discretization points for the contour plot by setting
  the keyword argument ``nc=250`` (or some other number).

Let's try to increase the number of discretization points for the contour plot
and decrease the number of discretization points for the quivers:

.. code:: ipython3

    vector_plot([-sin(y), cos(x)], (x, -5, 5), (y, -5, 5),
        backend=BB, xlabel="x", ylabel="y",
        quiver_kw=dict(scale=0.5), nc=250, n=20)

.. raw:: html
	:file: figs/tut-6/fig-06.html

Note that by increasing ``nc``, the plot is slower to render.
Having discovered that Bokeh doesn't handle that well a contour plot,
let's disable the scalar field:

.. code:: ipython3

    vector_plot([-sin(y), cos(x)], (x, -5, 5), (y, -5, 5),
        backend=BB, xlabel="x", ylabel="y",
        quiver_kw=dict(scale=0.5), scalar=None)

.. raw:: html
	:file: figs/tut-6/fig-07.html

By default, a color map will be applied to the quivers based on the local
magnitude value. We can further customize the color of the quivers by using
the ``quiver_kw``:

.. code:: ipython3

    vector_plot([-sin(y), cos(x)], (x, -5, 5), (y, -5, 5),
        backend=BB, xlabel="x", ylabel="y",
        quiver_kw=dict(scale=0.5, line_color="red", line_width=2),
        scalar=None)

.. raw:: html
	:file: figs/tut-6/fig-08.html

Finally, Bokeh also "supports" streamlines:

.. code:: ipython3

    vector_plot([-sin(y), cos(x)], (x, -5, 5), (y, -5, 5),
        backend=BB, xlabel="x", ylabel="y", streamlines=True)

.. raw:: html
	:file: figs/tut-6/fig-09.html


3D Vector Fields
================

As always, Bokeh doesn't support 3D plots, so we are left with Plotly and K3D.
The principle of operation is the same as 2D vector fields.

.. code:: ipython3

    x, y, z = symbols("x:z")
    vector_plot(Matrix([z, y, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
        n=7, quiver_kw=dict(sizeref=10), backend=PB,
        xlabel="x", ylabel="y", zlabel="z")

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-10.html" height="500px" width="100%"></iframe>

A few things to note:

* we used a matrix, ``Matrix([z, y, x])``, to represent a vector. When dealing
  with 3D vectors, some components may be numbers: in that case the internal
  algorithm might get confused, thinking of the vector as a range. In order to
  avoid this ambiguity, we wrap the 3D vector into a matrix of three elements
  and away we go.
* plotting 3D vector fields is computationally more expensive, hence we
  reduced the number of discretization points to ``n=7`` in each direction.
* 3D quivers are colored by the local value of the magnitude of the vector
  field.
* With the usual ``quiver_kw`` dictionary, we can provide backend-specific
  keyword arguments to control the appearance of the quivers. Here, we
  choose an appropriate size. Refer to
  `Plotly's Cone function <https://plotly.com/python/cone-plot/>`_ for more
  information.

It is usually difficult to understand a 3D vector field by using quivers.
Therefore, we might get a better idea by using streamlines:

.. code:: ipython3

    import numpy as np
    n = 200
    vector_plot(Matrix([z, y, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
        n=20, streamlines=True, backend=PB,
        xlabel="x", ylabel="y", zlabel="z",
        stream_kw=dict(
            starts = dict(
                    x = np.random.rand(n) * 10 - 5,
                    y = np.random.rand(n) * 10 - 5,
                    z = np.random.rand(n) * 10 - 5
            ),
            sizeref = 2800,
        )
    )

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-11.html" height="500px" width="100%"></iframe>

With the usual ``stream_kw`` dictionary we customize the appearance of the
streamlines. In order to generate them, we need to provide starting points,
which are going to be used in the integration process. In this case, we set
a random clouds of points in our domain. The tricky part is chosing the number
of points and the size of the streamlines. This is an iterative process.
Note that the streamlines are coloured according to the local magnitude value.

Now, let's change a little bit the vector for illustrative purposes:

.. code:: ipython3

    p1 = vector_plot(Matrix([y, z, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
        n=5, backend=PB, xlabel="x", ylabel="y", zlabel="z", show=False,
        quiver_kw=dict(sizeref=10))
    p2 = vector_plot(Matrix([y, z, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
        n=10, streamlines=True, backend=PB,
        xlabel="x", ylabel="y", zlabel="z", show=False)
    p1.extend(p2)
    p1.show()

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-12.html" height="500px" width="100%"></iframe>

A few things to note here:

* We created two separates plots of the same vector field and later merged
  the second (the streamlines) into the first (the quivers).
* Note that we didn't need to specify the starting points of the streamlines:
  the backend computed them based on the direction of the vectors relatively
  to the boundaries of the discretized volume.
* Also, at this moment if ``stream_kw`` was provided in the second plot,
  it would have been lost during the merging operation. Hopefully,
  this *bug* will be fixed in the future.

Now just for fun, let's visualize the original vector field with K3D:

.. code:: ipython3

    x, y, z = symbols("x:z")
    vector_plot(Matrix([z, y, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
        n=10, quiver_kw=dict(scale=0.2), backend=KB,
        xlabel="x", ylabel="y", zlabel="z")

.. raw:: html
	
    <iframe src="../_static/tut-6/fig-13.html" height="500px" width="100%"></iframe>

Note that we used different keyword argument to customize the size of
the quivers.

Let's now try to plot streamlines with K3DBackend. We can set the keyword
argument ``starts`` in the ``stream_kw`` dictionary to one of the following
values:

* ``starts=None`` (or do not provide it at all): the algorithm will
  automatically chose the seeds points of the streamlines on the surfaces of
  the discretized volume based on the direction of the vectors.
* ``starts=seeds_points``: similar to what we have seen with ``PlotlyBackend``,
  but here ``seeds_points`` is a ``n x 3`` matrix of coordinates.
* ``starts="random"``: the algorithm will randomly chose the seeds points
  of the streamlines inside the discretized volume. In this case we can also
  specify the number of points to be generated by setting ``npoints``: usually,
  the number of computed streamlines will be much lower than ``npoints``.

.. code:: ipython3

    vector_plot(Matrix([z, y, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
        n=20, streamlines=True,
        stream_kw=dict(width=0.1, starts="random", npoints=1000),
        backend=KB, xlabel="x", ylabel="y", zlabel="z")


Parametric-Interactive Vector Plots
===================================

We can also use ``iplot`` to play with parametric vector fields, all we have
to remember is to set ``is_vector = True``:

.. code:: ipython3

    from spb.interactive import iplot
    a, b, x, y, z = symbols("a, b, x:z")
    iplot(
        ([-a * sin(y), b * cos(x)], (x, -5, 5), (y, -3, 3)),
        params = {
            a: (1, (0, 2)),
            b: (1, (0, 2)),
        },
        xlabel = "x",
        ylabel = "y",
        backend = PB,
        n = 10,
        quiver_kw = dict(
            scale = 0.4
        ),
        is_vector = True
    )

In contrast to ``vector_plot``, the ``iplot`` function:

* We need to specify the number of discretization point, ``n=10``.
  Alternatively, we can set ``n1, n2, n3`` to specify the number of
  discretization points in the three directions.
  **Remembert to set ``n`` to a sufficiently low number**. Since ``n`` will be
  used on every direction, the internal algorithm will create 4 ``n x n``
  matrices for 2D vector fields, and 6 ``n x n x n`` matrices for 3D vector
  fields, hence a lot more memory will be used as we increase ``n``!!!
* A few other keyword arguments have been set to customize the appearance.

Let's try plotting streamlines with ``BokehBackend``. Remember: streamlines
are always more computationally expensive to compute, so expect a delay of a
few seconds from when you interact with the slider to the moment you will
see the updated plot:

.. code:: ipython3

    iplot(
        ([-a * sin(y), b * cos(x)], (x, -5, 5), (y, -3, 3)),
        params = {
            a: (1, (0, 2)),
            b: (1, (0, 2)),
        },
        xlabel = "x",
        ylabel = "y",
        backend = BB,
        n = 20,
        streamlines = True,
        stream_kw = dict(
            line_color = "red"
        )
    )

Let's now try to plot 3D vector fields. We are going to use Plotly and K3D:

.. code:: ipython3

    iplot(
        ([a * z, b * y, x], (x, -5, 5), (y, -3, 3), (z, -4, 4)),
        params = {
            a: (1, (0, 2)),
            b: (1, (0, 2)),
        },
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        backend = PB,
        n = 8,
        quiver_kw = dict(
            sizeref = 4
        )
    )

.. code:: ipython3

    iplot(
        ([-a * sin(y), b * cos(x)], (x, -5, 5), (y, -3, 3)),
        params = {
            a: (1, (0, 2)),
            b: (1, (0, 2)),
        },
        xlabel = "x",
        ylabel = "y",
        backend = BB,
        n = 20,
        quiver_kw = dict(
            scale = 0.25
        )
    )

At the time of writing this tutorial, Plotly and K3D do not support ``iplot``
for streamlines.


