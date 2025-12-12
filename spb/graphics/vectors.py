from sympy import (
    sympify, sqrt, Tuple, Expr, Plane
)
from sympy.matrices.dense import DenseMatrix
from sympy.external import import_module
from spb.doc_utils.docstrings import _PARAMS
from spb.doc_utils.ipython import modify_graphics_series_doc
from spb.series import (
    Vector2DSeries, Vector3DSeries, SliceVector3DSeries, BaseSeries,
    ContourSeries, Arrow2DSeries, Arrow3DSeries
)
from spb.utils import _preprocess_multiple_ranges
import warnings


def _split_vector(expr):
    """Extract the components of the given vector or matrix.

    Parameters
    ----------
    expr : Vector, DenseMatrix or list/tuple

    Returns
    -------
    split_expr : tuple
        Tuple of the form (x_expr, y_expr, z_expr) for a 3D vector field.
        If a 2D vector field is provided, returns (x_expr, y_expr).
    """
    from sympy.vector import Vector
    from sympy.vector.operators import _get_coord_systems
    from sympy.physics.vector import Vector as MechVector

    if isinstance(expr, Vector):
        N = list(_get_coord_systems(expr))[0]
        expr = expr.to_matrix(N)
    elif isinstance(expr, MechVector):
        expr = expr.args[0][0]
    elif not isinstance(expr, (DenseMatrix, list, tuple, Tuple)):
        raise TypeError(
            "The provided expression must be a symbolic vector, or a "
            "symbolic matrix, or a tuple/list with 2 or 3 symbolic "
            + "elements.\nReceived type = {}".format(type(expr))
        )
    elif (len(expr) < 2) or (len(expr) > 3):
        raise ValueError(
            "This function only plots 2D or 3D vectors.\n"
            + "Received: {}. Number of elements: {}".format(expr, len(expr))
        )

    if len(expr) == 3:
        xexpr, yexpr, zexpr = expr
    else:
        xexpr, yexpr = expr
        zexpr = None
    return xexpr, yexpr, zexpr


@modify_graphics_series_doc(Vector2DSeries, replace={"params": _PARAMS})
def vector_field_2d(
    u, v=None, range_x=None, range_y=None, label=None, is_streamlines=False,
    quiver_kw=None, stream_kw=None, contour_kw=None, **kwargs
):
    """
    Plot a 2D vector field.

    Parameters
    ==========

    u : Vector, Expr or callable
        The first component of the vector field. It can be:

        * A symbolic expression.
        * A numerical functions of 2 variables supporting vectorization.
        * A vector from the `sympy.vector` module or from the
          `sympy.physics.mechanics` module. In this case, ``v`` can be None
          and the algorithm will extract the components.
    v : Expr or callable
        The second component of the vector field. It can be:

        * A symbolic expression.
        * A numerical functions of 2 variables supporting vectorization.
    contour_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend
        contour function to customize the appearance. Refer to the plotting
        library (backend) manual for more informations.
    nc : int, optional
        Number of discretization points for the scalar contour plot.
        Default to 100.
    quiver_kw : dict, optional
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the quivers.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/quiver_api.html#module-matplotlib.quiver
        * Plotly:
          https://plotly.com/python/quiver-plots/
    scalar : boolean, Expr, None or list or tuple of 2 elements, optional
        Represents the scalar field to be plotted in the background of a 2D
        vector field plot. It can be:

        - `True`: plot the magnitude of the vector field. Only works when a
          single vector field is plotted.
        - `False`/`None`: do not plot any scalar field.
        - `Expr`: a symbolic expression representing the scalar field.
        - a numerical function of 2 variables supporting vectorization.
        - `list`/`tuple`: [scalar_expr, label], where the label will be
          shown on the colorbar. scalar_expr can be a symbolic expression
          or a numerical function of 2 variables supporting vectorization.

        Default to True.
    stream_kw : dict, optional
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the streamlines.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.streamplot.html#matplotlib.axes.Axes.streamplot
        * Plotly:
          https://plotly.com/python/streamline-plots/

    Returns
    =======

    series : list
        A list containing one instance of ``ContourSeries`` (if ``scalar`` is
        set) and one instance of ``Vector2DSeries``.

     Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, cos, Plane, Matrix, sqrt, latex
       >>> from spb import *
       >>> x, y, z = symbols('x, y, z')

    Quivers plot of a 2D vector field with a contour plot in background
    representing the vector's magnitude (a scalar field).

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     vector_field_2d(sin(x - y), cos(x + y), (x, -3, 3), (y, -3, 3),
       ...         quiver_kw=dict(color="black", scale=30, headwidth=5),
       ...         contour_kw={"cmap": "Blues_r", "levels": 15}),
       ...     grid=False, xlabel="x", ylabel="y")
       Plot object containing:
       [0]: contour: sqrt(sin(x - y)**2 + cos(x + y)**2) for x over (-3, 3) and y over (-3, 3)
       [1]: 2D vector series: [sin(x - y), cos(x + y)] over (x, -3.0, 3.0), (y, -3.0, 3.0)

    Quivers plot of a 2D vector field with no background scalar field,
    a custom label and normalized quiver lengths:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     vector_field_2d(sin(x - y), cos(x + y), (x, -3, 3), (y, -3, 3),
       ...         label="Magnitude of $%s$" % latex([-sin(y), cos(x)]),
       ...         scalar=False, normalize=True,
       ...         quiver_kw={
       ...             "scale": 35, "headwidth": 4, "cmap": "gray",
       ...             "clim": [0, 1.6]}),
       ...     grid=False, xlabel="x", ylabel="y")
       Plot object containing:
       [0]: 2D vector series: [sin(x - y), cos(x + y)] over (x, -3.0, 3.0), (y, -3.0, 3.0)

    Streamlines plot of a 2D vector field with no background scalar field, and
    a custom label:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     vector_field_2d(sin(x - y), cos(x + y), (x, -3, 3), (y, -3, 3),
       ...         streamlines=True, scalar=None,
       ...         stream_kw={"density": 1.5},
       ...         label="Magnitude of %s" % str([sin(x - y), cos(x + y)])),
       ...     xlabel="x", ylabel="y", grid=False)
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
       >>> graphics(
       ...     vector_field_2d(-sin(y), cos(x), (x, -5, 5), (y, -3, 3), n=20,
       ...         scalar=[scalar_expr, "$%s$" % latex(scalar_expr)],
       ...         contour_kw={"cmap": "summer"},
       ...         quiver_kw={"color": "k"}),
       ...     vector_field_2d(2 * y, x, (x, -5, 5), (y, -3, 3), n=20,
       ...         scalar=False, quiver_kw={"color": "r"}, use_cm=False),
       ...     aspect="equal", grid=False, xlabel="x", ylabel="y")
       Plot object containing:
       [0]: contour: sqrt(sin(y)**2 + cos(x)**2) for x over (-5, 5) and y over (-3, 3)
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
       >>> graphics(
       ...     vector_field_2d(fx, fy, ("x", -1, 1), ("y", -1, 1),
       ...         streamlines=True, scalar=False, use_cm=False),
       ...     aspect="equal", xlabel="x", ylabel="y", grid=False
       ... )  # doctest: +SKIP

    Interactive-widget 2D vector plot. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * customizing the appearance of quivers and countour.
    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).
    * the use of :py:class:`panel.widgets.slider.RangeSlider`, which is a
      2-values widget.

    .. panel-screenshot::
       :small-size: 800, 610

       from sympy import *
       from spb import *
       import panel as pn
       x, y, a, b, c, d, e = symbols("x, y, a, b, c, d, e")
       v = [-sin(a * y), cos(b * x)]
       graphics(
           vector_field_2d(-sin(a * y), cos(b * x),
               prange(x, -3*c, 3*c), prange(y, d, e),
               params={
                   a: (1, -2, 2), b: (1, -2, 2),
                   c: (1, 0, 2),
                   (d, e): pn.widgets.RangeSlider(
                       value=(-3, 3), start=-9, end=9, step=0.1)
               },
               quiver_kw=dict(color="black", scale=30, headwidth=5),
               contour_kw={"cmap": "Blues_r", "levels": 15}
           ),
           grid=False, xlabel="x", ylabel="y")

    See Also
    ========

    vector_field_3d

    """
    # back-compatibility
    range_x = kwargs.pop("range1", range_x)
    range_y = kwargs.pop("range2", range_y)

    if v is None:
        u, v, _ = _split_vector(u)
    is_vec_lambda_function = any(callable(e) for e in [u, v])
    if not is_vec_lambda_function:
        u, v = map(sympify, [u, v])

    if not (range_x or range_y):
        warnings.warn(
            "No ranges were provided. This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )

    nc = kwargs.pop("nc", 100) # number of discr points for contours
    scalar = kwargs.get("scalar", True)
    kwargs.setdefault("use_cm", False if scalar else True)
    is_streamlines = kwargs.pop("streamlines", False)
    params = kwargs.get("params", {})
    ranges = _preprocess_multiple_ranges(
        [u, v], [range_x, range_y], 2, params)
    s = Vector2DSeries(
        u, v, *ranges, label,
        is_streamlines=is_streamlines,
        rendering_kw=quiver_kw if not is_streamlines else stream_kw,
        **kwargs
    )

    if scalar is True:
        if not is_vec_lambda_function:
            scalar_field = sqrt(u**2 + v**2)
        else:
            np = import_module("numpy")
            scalar_field = lambda x, y: (
                np.sqrt(u(x, y)**2 + v(x, y)**2))
        scalar_label = "Magnitude"
    elif isinstance(scalar, Expr):
        scalar_field = scalar
        scalar_label = str(scalar)
    elif isinstance(scalar, (list, tuple)):
        scalar_field = scalar[0]
        scalar_label = scalar[1]
    elif callable(scalar):
        scalar_field = scalar
        scalar_label = "Magnitude"
    elif not scalar:
        scalar_field = None
    else:
        raise ValueError(
            "`scalar` must be either:\n"
            "1. True, in which case the magnitude of the vector field "
            "will be plotted.\n"
            "2. a symbolic expression representing a scalar field.\n"
            "3. None/False: do not plot any scalar field.\n"
            "4. list/tuple of two elements, [scalar_expr, label].\n"
            "5. a numerical function of 2 variables supporting "
            "vectorization."
        )

    series = [s]
    if scalar_field:
        cs_kwargs = kwargs.copy()
        for kw in ["n", "n1", "n2"]:
            if kw in cs_kwargs.keys():
                cs_kwargs.pop(kw)
        cs_kwargs["n1"] = nc
        cs_kwargs["n2"] = nc
        cs = ContourSeries(
            scalar_field, *ranges, scalar_label,
            rendering_kw=contour_kw, **cs_kwargs)
        series = [cs] + series
    return series


@modify_graphics_series_doc(Vector3DSeries, replace={"params": _PARAMS})
def vector_field_3d(
    u, v=None, w=None, range_x=None, range_y=None,
    range_z=None, label=None, is_streamlines=False, quiver_kw=None,
    stream_kw=None, **kwargs
):
    """
    Plot a 3D vector field in a cartesian coordinate system.

    Note that while it is possible to define vectors in a curvilinear
    coordinate system (spherical, cylindrical, etc.), the vector must first
    be expressed in a Cartesian system. More about this in the Examples
    section.

    Parameters
    ==========

    u : Vector, Expr or callable
        The first component of the vector field. It can be:

        * A symbolic expression.
        * A numerical functions of 2 variables supporting vectorization.
        * A vector from the `sympy.vector` module or from the
          `sympy.physics.mechanics` module. In this case, ``v, w`` can be None
          and the algorithm will extract the components.
    v : Expr or callable
        The second component of the vector field. It can be:

        * A symbolic expression.
        * A numerical functions of 2 variables supporting vectorization.
    w : Expr or callable
        The second component of the vector field. It can be:

        * A symbolic expression.
        * A numerical functions of 2 variables supporting vectorization.
    quiver_kw : dict, optional
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the quivers.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/quiver_api.html#module-matplotlib.quiver
        * Plotly:
          https://plotly.com/python/cone-plot/
    slice : Plane, list, Expr, optional
        Plot the 3D vector field over the provided slice. It can be:

        - a Plane object from sympy.geometry module.
        - a list of planes.
        - an instance of ``SurfaceOver2DRangeSeries`` or
          ``ParametricSurfaceSeries``.
        - a symbolic expression representing a surface of two variables.

        The number of discretization points will be `n1`, `n2`, `n3`.
        Note that:

        - only quivers plots are supported with slices. Streamlines plots
          are unaffected.
        - `n3` will only be used with planes parallel to xz or yz.
        - `n1`, `n2`, `n3` doesn't affect the slice if it is an instance of
          ``SurfaceOver2DRangeSeries`` or ``ParametricSurfaceSeries``.
    stream_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend
        streamlines-plotting function to customize the appearance.

        By default, the streamlines will start at the boundaries of the
        domain where the vectors are pointed inward.
        Depending on the vector field, this may results in too tight
        streamlines. Use the ``starts`` keyword argument to control the
        generation of streamlines:

        - ``starts=None``: the default aforementioned behaviour.
        - ``starts=dict(x=x_list, y=y_list, z=z_list)``: specify the starting
          points of the streamlines.
        - ``starts=True``: randomly create starting points inside the domain.
          In this setup we can set the number of starting point with
          ``npoints`` (default value to 200).

        If 3D streamlines appears to be cut short inside the specified domain,
        try to increase ``max_prop`` (default value to 5000).

        To further customize the appearance, here are some useful links for
        the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.streamplot.html#matplotlib.axes.Axes.streamplot
        * Plotly:
          https://plotly.com/python/streamtube-plot/
        * K3D-Jupyter: refers to k3d.line documentation.

    Returns
    =======

    series : list
        If ``slice`` is not set, the function returns a list containing one
        instance of ``Vector3DSeries``. Conversely, it returns a list
        containing instances of ``SliceVector3DSeries``.

    Examples
    ========

    Plot a 3D vector field defined in a Cartesian coordinate system:

    .. k3d-screenshot::

       from sympy import *
       from spb import *
       var("x:z")
       graphics(
           vector_field_3d(z, y, x, (x, -10, 10), (y, -10, 10), (z, -10, 10),
               n=8, quiver_kw={"scale": 0.5, "line_width": 0.1, "head_size": 10}),
           backend=KB, xlabel="x", ylabel="y", zlabel="z")

    3D vector field over 3 orthogonal slice planes.

    .. k3d-screenshot::
       :camera: 18.45, -25.63, 14.10, 0.45, -1.02, -2.32, -0.25, 0.35, 0.9

       from sympy import *
       from spb import *
       var("x:z")
       graphics(
           vector_field_3d(z, y, x, (x, -10, 10), (y, -10, 10), (z, -10, 10),
               n=8, use_cm=False,
               quiver_kw={"scale": 0.25, "line_width": 0.1, "head_size": 10},
               slice=[
                   Plane((-10, 0, 0), (1, 0, 0)),
                   Plane((0, 10, 0), (0, 2, 0)),
                   Plane((0, 0, -10), (0, 0, 1))]
           ),
           backend=KB, grid=False, xlabel="x", ylabel="y", zlabel="z",)

    3D streamlines starting at a 400 random points:

    .. k3d-screenshot::
       :camera: 3.7, -8.16, 2.8, -0.75, -0.51, -0.63, -0.16, 0.27, 0.96

       from sympy import *
       from spb import *
       import k3d
       var("x:z")
       graphics(
           vector_field_3d(z, -x, y, (x, -3, 3), (y, -3, 3), (z, -3, 3),
               n=40, streamlines=True,
               stream_kw=dict(
                   starts=True,
                   npoints=400,
                   width=0.025,
                   color_map=k3d.colormaps.matplotlib_color_maps.viridis
               )
           ),
           backend=KB, xlabel="x", ylabel="y", zlabel="z")

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
       p = plane(
           Plane((0, 0, 0), (0, 0, 1)), (x, -s, s), (y, -s, s), (z, -s, s),
           n1=n, n2=n)[0]
       xx, yy, zz = p.get_data()
       graphics(
           vector_field_3d(
               u, v, w, (x, -s, s), (y, -s, s), (z, -s, s),
               n=40, streamlines=True,
               stream_kw=dict(
                   starts=dict(x=xx, y=yy, z=zz),
                   width=0.025,
                   color_map=k3d.colormaps.matplotlib_color_maps.plasma
               )),
           title="Rössler \\, attractor", xlabel="x", ylabel="y", zlabel="z",
           backend=KB, grid=False)

    Visually verify the normal vector to a circular cone surface.
    The following steps are executed:

    1. compute the normal vector to a circular cone surface. This will be the
       vector field to be plotted.
    2. create a data series representing the cone surface for visualization
       purposes (use high number of discretization points).
    3. create a data series representing the cone surface that will be used
       to slice the vector field (use a low number of discretization points).
    4. create a data series for the normal vector, and assign step 3 to
       the ``slice`` keyword.
    5. Assemble the ``graphics`` command and get a nice visualization.

    .. k3d-screenshot::
       :camera: 4.5, -3.9, 2, 1.3, 0.04, -0.36, -0.25, 0.27, 0.93

       from sympy import tan, cos, sin, pi, symbols
       from spb import *
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

       # cone surface to discretize vector field (low numb of discret points)
       cone_discr = surface_parametric(
           u / tan(t), u * cos(v), u * sin(v), (u, 0, 1), (v, 0 , 2*pi),
           n1=n1, n2=n2)[0]
       graphics(
           surface_parametric(
               u / tan(t), u * cos(v), u * sin(v), (u, 0, 1), (v, 0 , 2*pi),
               rendering_kw={"opacity": 1}, wireframe=True,
               wf_n1=n1, wf_n2=n2, wf_rendering_kw={"width": 0.004}),
           vector_field_3d(
               n, range_x=(xn, -5, 5), range_y=(yn, -5, 5), range_z=(zn, -5, 5),
               use_cm=False, slice=cone_discr,
               quiver_kw={"scale": 0.5, "pivot": "tail"}
           ),
           backend=KB)

    Compute the normal vector and tangent vectors to a sphere using a
    spherical coordinate system. Then, convert them to cartesian coordinates
    for plotting. The conversion is performed with the ``express`` function
    exposed by this module, which supports curvilinear to cartesian (and
    viceversa) transformations:

    .. k3d-screenshot::

       from sympy import *
       from spb import *
       from spb.graphics.vector_transforms import express
       from sympy.vector import CoordSys3D
       C = CoordSys3D("C")
       S = C.create_new("S", transformation="spherical")
       x, y, z = C.base_scalars()
       r, theta, phi = S.base_scalars()

       # position vector for a point on the surface of a sphere of radius r
       sphere = r * S.i
       # get the parametric equation for a sphere
       parameterization = express(sphere, C).to_matrix(C).subs(r, 7)

       # normal and tangents to the sphere expressed in the cartesian frame,
       # using spherical variables
       n = express(sphere, C).normalize().simplify()
       t_theta = n.diff(theta).normalize().simplify()
       t_phi = n.diff(phi).normalize().simplify()

       # normal and tangents to the sphere expressed in the cartesian frame,
       # using cartesian variables
       d = {k: v for k, v in zip([r, theta, phi], S.transformation_from_parent())}
       n = n.subs(d)
       t_theta = t_theta.subs(d)
       t_phi = t_phi.subs(d)

       phi_max = 3 * pi / 2
       theta_max = pi / 2
       sphere_series = surface_parametric(
           *parameterization, (phi, 0, phi_max), (theta, 0, theta_max))
       locations_for_vectors = surface_parametric(
           *parameterization, (phi, 0, phi_max), (theta, 0, theta_max),
           n1=15, n2=8)[0]
       quiver_kw=dict(pivot="tail", head_size=2, line_width=0.03)
       common_kws = dict(
           # NOTE: dummy values for ranges. The important thing is
           # to order them appropriately. For example, variable x must go
           # on range_x, etc.
           range_x=(x, -1, 1), range_y=(y, -1, 1), range_z=(z, -1, 1),
           slice=locations_for_vectors, 
           use_cm=False, 
           quiver_kw=quiver_kw
       )
       graphics(
           sphere_series[0],
           vector_field_3d(t_theta, **common_kws),
           vector_field_3d(t_phi, **common_kws),
           vector_field_3d(n, **common_kws),
           grid=False, backend=KB
       )

    See Also
    ========

    vector_field_2d

    """
    # back-compatibility
    range_x = kwargs.pop("range1", range_x)
    range_y = kwargs.pop("range2", range_y)
    range_z = kwargs.pop("range3", range_z)

    if ((v is None) and w) or ((w is None) and v):
        raise ValueError(
            "`v` or `w` is None. This is not supported. "
            "Please, provide all components of the vector field.")
    if (v is None) and (w is None):
        u, v, w = _split_vector(u)
    is_vec_lambda_function = any(callable(e) for e in [u, v, w])
    if not is_vec_lambda_function:
        u, v, w = map(sympify, [u, v, w])
    if any(not isinstance(e, Expr) for e in [u, v, w]):
        raise ValueError("`u` and `v` must be symbolic expressions.")

    check = [range_x is None, range_y is None, range_z is None]
    if sum(check) >= 2:
        pre = "Not enough ranges were provided. "
        if sum(check) == 3:
            pre = "No ranges were provided. "
        warnings.warn(
            pre + "This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )

    params = kwargs.get("params", {})
    is_streamlines = kwargs.get("streamlines", False)
    _slice = kwargs.pop("slice", None)
    ranges = _preprocess_multiple_ranges(
        [u, v, w], [range_x, range_y, range_z], 3, params)
    series = [
        Vector3DSeries(
            u, v, w, *ranges, label,
            rendering_kw=quiver_kw if not is_streamlines else stream_kw,
            **kwargs
        )
    ]

    if _slice is None:
        return series

    # verify that the slices are of the correct type, , because symbolic
    # manipulation are applied to them.
    def _check_slice(s):
        if not isinstance(s, (Expr, Plane, BaseSeries)):
            raise ValueError(
                "A slice must be of type Plane or Expr or BaseSeries.\n"
                + "Received: {}, {}".format(type(s), s)
            )

    if isinstance(_slice, (list, tuple, Tuple)):
        for s in _slice:
            _check_slice(s)
    else:
        _check_slice(_slice)
        _slice = [_slice]

    slice_series = []
    for s in _slice:
        slice_series.append(
            SliceVector3DSeries(
                s, u, v, w, *ranges, label,
                rendering_kw=quiver_kw, **kwargs))
    return slice_series


@modify_graphics_series_doc(Arrow2DSeries, replace={"params": _PARAMS})
def arrow_2d(
    start, direction, label=None, rendering_kw=None, show_in_legend=True,
    **kwargs
):
    """
    Draw an arrow in a 2D space.

    Returns
    =======

    A list containing one instance of ``Arrow2DSeries``.

    See Also
    ========

    vector_field_2d

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from spb import *
       >>> graphics(
       ...     arrow_2d((0, 0), (1, 1)),
       ...     arrow_2d((0, 0), (-1, 1)),
       ...     grid=False, aspect="equal"
       ... )
       Plot object containing:
       [0]: 2D arrow from (0.0, 0.0) to (1.0, 1.0)
       [1]: 2D arrow from (0.0, 0.0) to (-1.0, 1.0)

    Interactive-widget plot of arrows. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.

    .. panel-screenshot::
       :small-size: 800, 610

       from sympy import *
       from spb import *
       r, theta = symbols("r, theta")
       params = {
           r: (4, 0, 5),
           theta: (pi/3, 0, 2*pi),
       }
       graphics(
           arrow_2d(
               (0, 0), (5, 0), show_in_legend=False,
               rendering_kw={"color": "k"}),
           arrow_2d(
               (0, 0), (0, 5), show_in_legend=False,
               rendering_kw={"color": "k"}),
           arrow_2d(
               (0, 0), (r * cos(theta), r * sin(theta)),
               params=params),
           arrow_2d(
               (0, 0), (r * cos(theta + pi/2), r * sin(theta + pi/2)),
               params=params),
           xlim=(-6, 6), ylim=(-6, 6), aspect="equal",
           grid=False
       )

    """
    return [
        Arrow2DSeries(
            start, direction, label, rendering_kw=rendering_kw,
            show_in_legend=show_in_legend, **kwargs)
    ]


@modify_graphics_series_doc(Arrow3DSeries, replace={"params": _PARAMS})
def arrow_3d(
    start, direction, label=None, rendering_kw=None, show_in_legend=True,
    **kwargs
):
    """
    Draw an arrow in a 2D space.

    Returns
    =======

    A list containing one instance of ``Arrow3DSeries``.

    See Also
    ========

    arrow_2d, vector_field_3d

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from spb import *
       >>> graphics(
       ...     arrow_3d((0, 0, 0), (1, 0, 0)),
       ...     arrow_3d((0, 0, 0), (0, 1, 0)),
       ...     arrow_3d((0, 0, 0), (0, 0, 1), show_in_legend=False,
       ...              rendering_kw={
       ...                  "mutation_scale": 20,
       ...                  "arrowstyle": "-|>",
       ...                  "linestyle": 'dashed',
       ...              }),
       ...     xlabel="x", ylabel="y", zlabel="z")
       Plot object containing:
       [0]: 3D arrow from (0.0, 0.0, 0.0) to (1.0, 0.0, 0.0)
       [1]: 3D arrow from (0.0, 0.0, 0.0) to (0.0, 1.0, 0.0)
       [2]: 3D arrow from (0.0, 0.0, 0.0) to (0.0, 0.0, 1.0)

    Interactive-widget plot of arrows. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.

    .. panel-screenshot::
       :small-size: 800, 610

       from sympy import *
       from spb import *
       phi, theta = symbols("phi, theta")
       r = 0.75
       params = {
           phi: (-pi/2, -pi, pi),
           theta: (2*pi/3, -pi, pi),
       }
       graphics(
           arrow_3d((0, 0, 0), (1, 0, 0), rendering_kw={"color": "k"},
               show_in_legend=False),
           arrow_3d((0, 0, 0), (0, 1, 0), rendering_kw={"color": "k"},
               show_in_legend=False),
           arrow_3d((0, 0, 0), (0, 0, 1), rendering_kw={"color": "k"},
               show_in_legend=False),
           arrow_3d(
               (0, 0, 0),
               (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)),
               params=params),
           xlabel="x", ylabel="y", zlabel="z",
           xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5), aspect="equal"
       )

    """
    return [
        Arrow3DSeries(
            start, direction, label, rendering_kw=rendering_kw,
            show_in_legend=show_in_legend, **kwargs)
    ]
