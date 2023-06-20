=========
 Overview
=========

The following overview briefly introduce the functionalities exposed by this
module.

Plotting functions
==================

On top of the usual and limited SymPy plotting functions, many new functions
are implemented to deal with 2D or 3D lines, contours, surfaces, vectors,
complex functions and control theory. The output of all of them can be viewed
by exploring the :doc:`Modules </modules/index>` section.


Backends
========

This module allows the user to chose between 5 different backends (plotting
libraries):
`Matplotlib <https://matplotlib.org/>`_,
`Plotly <https://plotly.com/>`_,
`Bokeh <https://github.com/bokeh/bokeh>`_,
`K3D-Jupyter <https://github.com/K3D-tools/K3D-jupyter>`_,
`Mayavi <https://docs.enthought.com/mayavi/mayavi/>`_ (support for this backend
is limited).

The 3 most important reasons for supporting multiple backends are:

#. **In the Python ecosystem there is no perfect plotting library**. Each one
   is great at something and terrible at something else. Supporting multiple
   backends allows the plotting module to have a greater capability of
   visualizing different kinds of symbolic expressions.

#. **Better interactive** experience (explored in the tutorial section), which
   translates to better data exploration and visualization (especially when
   working with Jupyter Notebook).

#. To use the **plotting library we are most comfortable with**. The backend
   can be used as a starting point to plot symbolic expressions; then, we could
   use the figure object to add numerical (or experimental) results using the
   commands associated to the specific plotting library.

More information about the backends can be found at:
:doc:`Backends </modules/backends/index>` .


Examples
========

The following code blocks shows a few examples about the capabilities of
this module. Please, try them on a Jupyter Notebook to explore the interactive
figures. Alternatively, consider loading the ``html`` output when available:
note that changing the state of widgets won't update the plot as there is no
active Python kernel running on this web page.


Interactive-Parametric 2D plot of the magnitude of a second order transfer
function:

.. panel-screenshot::

   from sympy import symbols, log, sqrt, re, im, I
   from spb import plot, BB
   from bokeh.models.formatters import PrintfTickFormatter
   formatter = PrintfTickFormatter(format="%.3f")
   kp, t, z, o = symbols("k_P, tau, zeta, omega")
   G = kp / (I**2 * t**2 * o**2 + 2 * z * t * o * I + 1)
   mod = lambda x: 20 * log(sqrt(re(x)**2 + im(x)**2), 10)
   plot(
       (mod(G.subs(z, 0)), (o, 0.1, 100), "G(z=0)", {"line_dash": "dotted"}),
       (mod(G.subs(z, 1)), (o, 0.1, 100), "G(z=1)", {"line_dash": "dotted"}),
       (mod(G), (o, 0.1, 100), "G"),
       params = {
           kp: (1, 0, 3),
           t: (1, 0, 3),
           z: (0.2, 0, 1, 200, formatter, "z")
       },
       backend = BB,
       n = 2000,
       xscale = "log",
       xlabel = "Frequency, omega, [rad/s]",
       ylabel = "Magnitude [dB]",
       use_latex = False
   )


Polar plot with Matplotlib:

.. plot::
   :context: reset
   :include-source: True

   from sympy import symbols, sin, cos, pi, latex
   from spb import plot_polar
   x = symbols("x")
   expr = sin(2 * x) * cos(5 * x) + pi / 2
   plot_polar(expr, (x, 0, 2 * pi),
       polar_axis=True, ylim=(0, 3), title="$%s$" % latex(expr))


2D parametric plot with Matplotlib, using Numpy and lambda functions:

.. plot::
   :context: reset
   :include-source: True

   import numpy as np
   from spb import plot_parametric
   plot_parametric(
      lambda t: np.sin(3 * t + np.pi / 4), lambda t: np.sin(4 * t),
      ("t", 0, 2 * np.pi), "t [rad]", xlabel="x", ylabel="y", aspect="equal")


Interactive-Parametric domain coloring plot of a complex function:

.. panel-screenshot::
   :small-size: 800, 625

   from sympy import symbols, latex
   from spb import plot_complex
   import colorcet
   u, v, w, z = symbols("u, v, w, z")
   expr = (z - 1) / (u * z**2 + v * z + w * 1)
   plot_complex(
     expr, (z, -2-2j, 2+2j),
     params={
         u: (1, 1e-5, 2),
         v: (1, 0, 2),
         w: (1, 0, 2),
     },
     coloring="b", cmap=colorcet.CET_C7, n=500,
     use_latex=False, title="$%s$" % latex(expr), grid=False)


3D plot with K3D-Jupyter and polar discretization. Two identical expressions
are going to be plotted, one will display the mesh with a solid color, the
other will display the connectivity of the mesh (wireframe).
Customization on the colors, surface/wireframe can easily be done after the
plot is created:

.. k3d-screenshot::
   :camera: 1.092, -3.01, 1.458, 0.159, -0.107, -0.359, -0.185, 0.427, 0.885

   from sympy import symbols, cos, sin, pi, latex
   from spb import plot3d, KB
   r, theta = symbols("r, theta")
   expr = cos(r) * cos(sin(4 * theta))
   plot3d(
       (expr, {"color": 0x1f77b4}),
       (expr, {"color": 0x1a5fb4, "opacity": 0.15, "wireframe": True}),
       (r, 0, 2), (theta, 0, 2 * pi),
       n1=50, n2=200, is_polar=True, grid=False,
       title=r"f\left(r, \theta\right) = " + latex(expr), backend=KB)


3D plot with Plotly of a parametric surface, colored according to the
radius, with wireframe lines (also known as grid lines) highlighting the
parameterization:

.. plotly::
   :camera: 1.75, 0, 0, 0, 0, 0, 0, 0, 1

   from sympy import symbols, cos, sin, pi
   from spb import plot3d_parametric_surface, PB
   import numpy as np
   u, v = symbols("u, v")
   def trefoil(u, v, r):
       x = r * sin(3 * u) / (2 + cos(v))
       y = r * (sin(u) + 2 * sin(2 * u)) / (2 + cos(v + pi * 2 / 3))
       z = r / 2 * (cos(u) - 2 * cos(2 * u)) * (2 + cos(v)) * (2 + cos(v + pi * 2 / 3)) / 4
       return x, y, z
   plot3d_parametric_surface(
      trefoil(u, v, 3), (u, -pi, 3*pi), (v, -pi, 3*pi), "radius",
      grid=False, title="Trefoil Knot", backend=PB, use_cm=True,
      color_func=lambda x, y, z: np.sqrt(x**2 + y**2 + z**2),
      wireframe=True, wf_n1=100, wf_n2=30, n1=250, show=False)


Visualizing a 2D vector field:

.. plotly::

   from sympy import *
   from spb import *
   x, y = symbols("x, y")
   expr = Tuple(1, sin(x**2 + y**2))
   l = 2
   plot_vector(
      expr, (x, -l, l), (y, -l, l),
      backend=PB, streamlines=True, scalar=False,
      stream_kw={"line_color": "black", "density": 1.5},
      xlim=(-l, l), ylim=(-l, l),
      title=r"$\vec{F} = " + latex(expr) + "$")


Visualizing a 3D vector field with a random number of streamtubes:

.. k3d-screenshot::
   :camera: 40.138, -37.134, 35.253, 4.387, -4.432, 25.837, 0.338, 0.513, 0.789

   from sympy import *
   from spb import *
   var("x:z")

   l = 30
   u = 10 * (y - x)
   v = 28 * x - y - x * z
   w = -8 * z / 3 + x * y

   plot_vector(
      [u, v, w], (x, -l, l), (y, -l, l), (z, 0, 50),
      backend=KB, n=50, grid=False, use_cm=False, streamlines=True,
      stream_kw={"starts": True, "npoints": 15},
      title="Lorentz \, attractor"
   )



Differences with sympy.plotting
===============================

* While the backends implemented in this module might resemble the ones from
  the `sympy.plotting` module, they are not interchangeable.

* `sympy.plotting` also provides a ``Plotgrid`` class to combine multiple plots
  into a grid-like layout. This module replaces that class with the
  ``plotgrid`` function. Again, they are not interchangeable.

* The ``plot_implicit`` function uses a mesh grid algorithm and contour plots
  by default (in contrast to the adaptive algorithm used by `sympy.plotting`).
  It is going to automatically switch to an adaptive algorithm if
  Boolean expressions are found. This ensures a better visualization for
  non-Boolean implicit expressions.

* ``experimental_lambdify``, used by `sympy.plotting`, has been completely
  removed.

* `sympy.plotting` is unable to visualize summations containing infinity in
  their lower/upper bounds. The new module introduces the ``sum_bound`` keyword
  argument into the ``plot`` function: it substitutes infinity with a large
  integer number. As such, it is possible to visualize summations.

* The adaptive algorithm is also different: this module relies on
  `adaptive <https://github.com/python-adaptive/adaptive/>`_, which allows more
  flexibility.

  * The ``depth`` keyword argument has been removed, while ``adaptive_goal``
    and ``loss_fn`` have been introduced to control the new module.
  * It has also been implemented to 3D lines and surfaces.
  * It allows to generate smoother line plots, at the cost of performance.

* `sympy.plotting` exposed the ``nb_of_points_*`` keyword arguments. These have
  been replaced with ``n`` or ``n1, n2``.

* `sympy.plotting` exposed the ``TextBackend`` class to create very basic
  plots on a terminal window. This module removed it.

  The following example compares how to customize a plot created with
  `sympy.plotting` and one created with this module.

  This is pretty much all we can do with `sympy.plotting`:

  .. code-block:: python

     from sympy.plotting import plot
     from sympy import symbols, sin, cos
     x = symbols("x")
     p = plot(sin(x), cos(x), show=False)
     p[0].label = "a"
     p[0].line_color = "red"
     p[1].label = "b"
     p.show()

  The above command works perfectly fine also with this new module. However,
  we can customize the plot even further. In particular:

  * it is possible to set a custom label directly from any plot function.
  * the full potential of each backend can be accessed by providing
    dictionaries containing backend-specific keyword arguments.

  .. code-block:: python

     from spb import plot
     from sympy import symbols, sin, cos
     x = symbols("x")
     # pass customization options directly to matplotlib (or other backends)
     plot(
         (sin(x), "a", dict(color="k", linestyle=":")),
         (cos(x), "b"),
         backend=MB)
     # alternatively, set the label and rendering_kw keyword arguments
     # to lists: each element target an expression
     # plot(sin(x), cos(x), label=["a", "b"], rendering_kw=[dict(color="k", linestyle=":"), None])

  Read the documentation to learn how to further customize the appearance of
  figures.

Take a look at :doc:`Modules </modules/index>` for more examples about the output of this module.
