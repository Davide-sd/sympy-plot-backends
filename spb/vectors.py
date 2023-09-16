from spb.plot_functions.functions_2d import _set_labels
from spb.graphics import vector_field_2d, vector_field_3d, graphics
from spb.graphics.vectors import _split_vector
from spb.utils import _plot_sympify, _is_range, _unpack_args
from sympy import Tuple


def _preprocess(*args):
    """Loops over the arguments and build a list of arguments having the
    following form: [expr, *ranges, label].
    `expr` can be a vector, a matrix or a list/tuple/Tuple.
    """

    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        # In this case we received arguments in one of the following forms.
        # Here we wrapped them into a list, so that they can be further
        # processed:
        #   v               -> [v]
        #   v, range        -> [v, range]
        #   v1, v2, ..., range   -> [v1, v2, range]
        args = [args]

    if any([_is_range(a) for a in args]):
        args = [args]

    new_args = []
    for a in args:
        exprs, ranges, label, rendering_kw = _unpack_args(*a)
        if len(exprs) == 1:
            new_args.append([*exprs, *ranges, label])
        else:
            # this is the case where the user provided: v1, v2, ..., range
            # we use the same ranges for each expression
            for e in exprs:
                new_args.append([e, *ranges, None])
    return new_args


def plot_vector(*args, **kwargs):
    """
    Plot a 2D or 3D vector field. By default, the aspect ratio of the plot
    is set to `aspect="equal"`.

    Typical usage examples are in the followings:

    - Plotting a 2D vector field:

      .. code-block::

         plot_vector(vec, range_x, range_y, **kwargs)

    - Plotting multiple 2D vector fields with different ranges and
      custom labels:

      .. code-block::

         plot_vector(
            (vec1, range1_x, range1_y, label1 [opt]),
            (vec2, range2_x, range2_y, label2 [opt]),
            **kwargs)

    - Plotting a 3D vector field:

      .. code-block::

         plot_vector(vec, range_x, range_y, range_z, **kwargs)

    - Plotting multiple 3D vector fields with different ranges and
      custom labels:

      .. code-block::

         plot_vector(
            (vec1, range1_x, range1_y, range1_z, label1 [opt]),
            (vec2, range2_x, range2_y, range2_z, label2 [opt]),
            **kwargs)

    Refer to :func:`~spb.graphics.vectors.vector_field_2d` for a full
    list of keyword arguments to customize the appearances of quivers,
    streamlines and contours for a 2D vector field.

    Refer to :func:`~spb.graphics.vectors.vector_field_3d` for a full
    list of keyword arguments to customize the appearances of quivers and
    streamlines for a 3D vector field.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, cos, Plane, Matrix, sqrt, latex
       >>> from spb import plot_vector
       >>> x, y, z = symbols('x, y, z')

    Quivers plot of a 2D vector field with a contour plot in background
    representing the vector's magnitude (a scalar field).

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> v = [sin(x - y), cos(x + y)]
       >>> plot_vector(v, (x, -3, 3), (y, -3, 3),
       ...     quiver_kw=dict(color="black", scale=30, headwidth=5),
       ...     contour_kw={"cmap": "Blues_r", "levels": 15},
       ...     grid=False, xlabel="x", ylabel="y")
       Plot object containing:
       [0]: contour: sqrt(sin(x - y)**2 + cos(x + y)**2) for x over (-3.0, 3.0) and y over (-3.0, 3.0)
       [1]: 2D vector series: [sin(x - y), cos(x + y)] over (x, -3.0, 3.0), (y, -3.0, 3.0)

    Quivers plot of a 2D vector field with no background scalar field,
    a custom label and normalized quiver lengths:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector(
       ...     v, (x, -3, 3), (y, -3, 3),
       ...     label="Magnitude of $%s$" % latex([-sin(y), cos(x)]),
       ...     scalar=False, normalize=True,
       ...     quiver_kw={
       ...         "scale": 35, "headwidth": 4, "cmap": "gray",
       ...         "clim": [0, 1.6]},
       ...     grid=False, xlabel="x", ylabel="y")
       Plot object containing:
       [0]: 2D vector series: [sin(x - y), cos(x + y)] over (x, -3.0, 3.0), (y, -3.0, 3.0)

    Streamlines plot of a 2D vector field with no background scalar field, and
    a custom label:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_vector(v, (x, -3, 3), (y, -3, 3),
       ...     streamlines=True, scalar=None,
       ...     stream_kw={"density": 1.5},
       ...     label="Magnitude of %s" % str(v), xlabel="x", ylabel="y")
       Plot object containing:
       [0]: 2D vector series: [sin(x - y), cos(x + y)] over (x, -3.0, 3.0), (y, -3.0, 3.0)


    Plot multiple 2D vectors fields, setting a background scalar field to be
    the magnitude of the first vector. Also, apply custom rendering options
    to all data series.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> scalar_expr = sqrt((-sin(y))**2 + cos(x)**2)
       >>> plot_vector([-sin(y), cos(x)], [2 * y, x], (x, -5, 5), (y, -3, 3),
       ...     n=20, legend=True, grid=False, xlabel="x", ylabel="y",
       ...     scalar=[scalar_expr, "$%s$" % latex(scalar_expr)],
       ...     rendering_kw=[
       ...         {"cmap": "summer"}, # to the contour
       ...         {"color": "k"},     # to the first quiver
       ...         {"color": "w"}      # to the second quiver
       ... ])
       Plot object containing:
       [0]: contour: sqrt(sin(y)**2 + cos(x)**2) for x over (-5.0, 5.0) and y over (-3.0, 3.0)
       [1]: 2D vector series: [-sin(y), cos(x)] over (x, -5.0, 5.0), (y, -3.0, 3.0)
       [2]: 2D vector series: [2*y, x] over (x, -5.0, 5.0), (y, -3.0, 3.0)

    Plotting a the streamlines of a 2D vector field defined with numerical
    functions instead of symbolic expressions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> f = lambda x, y: np.sin(2 * x + 2 * y)
       >>> fx = lambda x, y: np.cos(f(x, y))
       >>> fy = lambda x, y: np.sin(f(x, y))
       >>> plot_vector([fx, fy], ("x", -1, 1), ("y", -1, 1),
       ...     streamlines=True, scalar=False, use_cm=False)  # doctest: +SKIP

    Interactive-widget 2D vector plot. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * customizing the appearance of quivers and countour.
    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 610

       from sympy import *
       from spb import *
       x, y, a, b, c, d = symbols("x, y, a, b, c, d")
       v = [-sin(a * y), cos(b * x)]
       plot_vector(
           v, prange(x, -3*c, 3*c), prange(y, -3*d, 3*d),
           params={
               a: (1, -2, 2), b: (1, -2, 2),
               c: (1, 0, 2), d: (1, 0, 2),
           },
           quiver_kw=dict(color="black", scale=30, headwidth=5),
           contour_kw={"cmap": "Blues_r", "levels": 15},
           grid=False, xlabel="x", ylabel="y", use_latex=False)

    3D vector field.

    .. k3d-screenshot::

       from sympy import *
       from spb import *
       var("x:z")
       plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
           backend=KB, n=8, xlabel="x", ylabel="y", zlabel="z",
           quiver_kw={"scale": 0.5, "line_width": 0.1, "head_size": 10})

    3D vector field with 3 orthogonal slice planes.

    .. k3d-screenshot::
       :camera: 18.45, -25.63, 14.10, 0.45, -1.02, -2.32, -0.25, 0.35, 0.9

       from sympy import *
       from spb import *
       var("x:z")
       plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
           backend=KB, n=8, use_cm=False, grid=False,
           xlabel="x", ylabel="y", zlabel="z",
           quiver_kw={"scale": 0.25, "line_width": 0.1, "head_size": 10},
           slice=[
               Plane((-10, 0, 0), (1, 0, 0)),
               Plane((0, 10, 0), (0, 2, 0)),
               Plane((0, 0, -10), (0, 0, 1))])

    3D vector streamlines starting at a 300 random points:

    .. k3d-screenshot::
       :camera: 3.7, -8.16, 2.8, -0.75, -0.51, -0.63, -0.16, 0.27, 0.96

       from sympy import *
       from spb import *
       import k3d
       var("x:z")
       plot_vector(Matrix([z, -x, y]), (x, -3, 3), (y, -3, 3), (z, -3, 3),
           backend=KB, n=40, streamlines=True,
           stream_kw=dict(
               starts=True,
               npoints=400,
               width=0.025,
               color_map=k3d.colormaps.matplotlib_color_maps.viridis
           ),
           xlabel="x", ylabel="y", zlabel="z")

    3D vector streamlines starting at the XY plane. Note that the number of
    discretization points of the plane controls the numbers of streamlines.

    .. k3d-screenshot::
       :camera: -2.64, -22.6, 8.8, 0.03, -0.6, -1.13, 0.1, 0.35, 0.93

       from sympy import *
       from spb import *
       import k3d
       var("x:z")
       u = -y - z
       v = x + y / 5
       w = S(1) / 5 + (x - S(5) / 2) * z
       s = 10 # length of the cubic discretization volume
       # create an XY plane with n discretization points along each direction
       n = 8
       p1 = plot_geometry(
           Plane((0, 0, 0), (0, 0, 1)), (x, -s, s), (y, -s, s), (z, -s, s),
           n1=n, n2=n, show=False)
       # extract the coordinates of the starting points for the streamlines
       xx, yy, zz = p1[0].get_data()
       # streamlines plot
       plot_vector(Matrix([u, v, w]), (x, -s, s), (y, -s, s), (z, -s, s),
           backend=KB, n=40, streamlines=True, grid=False,
           stream_kw=dict(
               starts=dict(x=xx, y=yy, z=zz),
               width=0.025,
               color_map=k3d.colormaps.matplotlib_color_maps.plasma
           ),
           title=r"Rössler \, attractor", xlabel="x", ylabel="y", zlabel="z")

    Visually verify the normal vector to a circular cone surface.
    The following steps are executed:

    1. compute the normal vector to a circular cone surface. This will be the
       vector field to be plotted.
    2. plot the cone surface for visualization purposes (use high number of
       discretization points).
    3. plot the cone surface that will be used to slice the vector field (use
       a low number of discretization points). The data series associated to
       this plot will be used in the ``slice`` keyword argument in the next
       step.
    4. plot the sliced vector field.
    5. combine the plots of step 4 and 2 to get a nice visualization.

    .. k3d-screenshot::
       :camera: 4.5, -3.9, 2, 1.3, 0.04, -0.36, -0.25, 0.27, 0.93

       from sympy import tan, cos, sin, pi, symbols
       from spb import plot3d_parametric_surface, plot_vector, KB
       from sympy.vector import CoordSys3D, gradient

       u, v = symbols("u, v")
       N = CoordSys3D("N")
       i, j, k = N.base_vectors()
       xn, yn, zn = N.base_scalars()

       t = 0.35    # half-cone angle in radians
       expr = -xn**2 * tan(t)**2 + yn**2 + zn**2    # cone surface equation
       g = gradient(expr)
       n = g / g.magnitude()    # unit normal vector
       n1, n2 = 10, 20 # number of discretization points for the vector field

       # cone surface for visualization (high number of discretization points)
       p1 = plot3d_parametric_surface(
           u / tan(t), u * cos(v), u * sin(v), (u, 0, 1), (v, 0 , 2*pi),
           {"opacity": 1}, backend=KB, show=False, wireframe=True,
           wf_n1=n1, wf_n2=n2, wf_rendering_kw={"width": 0.004})
       # cone surface to discretize vector field (low numb of discret points)
       p2 = plot3d_parametric_surface(
           u / tan(t), u * cos(v), u * sin(v), (u, 0, 1), (v, 0 , 2*pi),
           n1=n1, n2=n2, show=False)
       # plot vector field on over the surface of the cone
       p3 = plot_vector(
           n, (xn, -5, 5), (yn, -5, 5), (zn, -5, 5), slice=p2[0],
           backend=KB, use_cm=False, show=False,
           quiver_kw={"scale": 0.5, "pivot": "tail"})
       (p1 + p3).show()

    """
    args = _plot_sympify(args)
    args = _preprocess(*args)
    kwargs.setdefault("aspect", "equal")
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    scalar = kwargs.pop("scalar", -1)
    if (scalar == -1) and len(args) == 1:
        scalar = True

    series = []
    for i, a in enumerate(args):
        vec = _split_vector(a[0])
        label = None if not isinstance(a[-1], str) else a[-1]
        ranges = a[1:-1]

        kw = kwargs.copy()
        kw.update(dict(zip(["range1", "range2", "range3"], ranges)))
        kw["label"] = label
        kw["scalar"] = scalar if (i == 0) and (scalar != -1) else None

        if (vec[-1] is None) or (vec[-1] == 0):
            series.extend(vector_field_2d(*vec[:2], **kw))
        else:
            series.extend(vector_field_3d(*vec, **kw))
    _set_labels(series, global_labels, global_rendering_kw)

    return graphics(*series, **kwargs)
