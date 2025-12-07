from PIL import ImageColor
from sympy.external import import_module
import param
import inspect
from fractions import Fraction
from numbers import Number as PythonNumber
from sympy import (
    Number as SympyNumber,
    NumberSymbol
)
from spb.doc_utils.ipython import modify_parameterized_doc


def convert_colormap(cm, to, n=256):
    """Convert the provided colormap to a format usable by the specified
    plotting library. The following plotting libraries are supported:
    matplotlib, plotly, bokeh, k3d.

    Parameters
    ==========
        cm : Colormap, list, tuple, ndarray
            The provided colormap. It can be:
            * an instance of matplotlib.colors.Colormap
            * a string with the name of a Plotly color scale
            * a list of string HEX colors (colorcet colormaps)
            * a list of float numbers between 0 and 1 (k3d colormaps)
        to : str
            Specify the plotting library.
        n : int
            Number of discretization points in the range [0, 1].
            Default to 256.
            This is only used if `cm` is an instance of Colormap or if `cm` is
            a string with the name of a Plotly color scale.

    Returns
    =======
        A new colormap. Note that the conversion is not guardanteed.
        The function returns the provided colormap if it cannot be converted.
    """
    np = import_module('numpy')
    matplotlib = import_module(
        'matplotlib',
        import_kwargs={'fromlist': ['colors']},
        min_module_version='1.1.0',
        catch=(RuntimeError,))
    Colormap = matplotlib.colors.Colormap

    assert isinstance(to, str)
    to = to.lower()
    assert to in ["matplotlib", "plotly", "k3d", "bokeh", "mayavi"]
    if not isinstance(cm, (str, list, tuple, np.ndarray, Colormap)):
        raise ValueError(
            "`cm` must be either:\n"
            + "1. a string with the name of a Plotly color scale.\n"
            + "2. a list of string HEX colors (colorcet colormaps).\n"
            + "2. a list of float numbers between 0 and 1 (k3d colormaps).\n"
            + "3. an instance of matplotlib.colors.Colormap.\n"
            + "4. an array of colors extracted from a matplotlib.colors.Colormap."
        )

    if to == "mayavi":
        # NOTE: Mayavi colormaps are based on look up tables.
        # It is possible to modify a colormap after an object (mesh) has been
        # created (see this example):
        # https://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html
        # However, it is not possible to pass a look up table to the colormap
        # keyword argument of a Mayavi function. Hence, we cannot implement
        # intercompatibility with other plotting libraries.
        return cm

    r = []
    if to == "k3d":
        # K3D color maps are lists of the form:
        # [loc1, r1, g1, b1, loc2, r2, b2, g2, ...]
        if isinstance(cm, Colormap):
            # matplotlib color map
            discr = np.linspace(0, 1, n)
            for loc, color in zip(discr, cm(discr)):
                r.append(loc)
                r += list(color[:-1])
        elif isinstance(cm, str):
            # Plotly color scale
            discr = np.linspace(0, 1, n)
            colors = get_plotly_colors(cm, discr)
            for loc, color in zip(discr, colors):
                r.append(loc)
                r += color
        elif isinstance(cm, np.ndarray) or all(
            [isinstance(c, (list, tuple)) for c in cm]
        ):
            if isinstance(cm, (list, tuple)):
                cm = np.array(cm)

            if cm.shape[1] == 4:
                # matplotlib color map already extracted
                for loc, color in zip(np.linspace(0, 1, len(cm)), cm):
                    r.append(loc)
                    r += list(color[:-1])
            else:
                # colorcet color map
                for loc, color in zip(np.linspace(0, 1, len(cm)), cm):
                    r.append(loc)
                    r += list(color)
        elif all([isinstance(c, str) for c in cm]):
            # colorcet colormap
            for loc, color in zip(np.linspace(0, 1, len(cm)), cm):
                r.append(loc)
                c = ImageColor.getcolor(color, "RGB")
                r += [float(e) / 255 for e in c]
        else:
            r = cm
    elif to == "plotly":
        if isinstance(cm, str):
            # plotly color scale name
            r = cm
        elif isinstance(cm, Colormap):
            # matplotlib color map
            discr = np.linspace(0, 1, n)
            colors = (cm(discr) * 255).astype(int)
            r = [
                [loc, "rgb(%d, %d, %d)" % tuple(c[:-1])]
                for loc, c in zip(discr, colors)
            ]
        elif isinstance(cm, np.ndarray) or all(
            [isinstance(c, (list, tuple)) for c in cm]
        ):
            if isinstance(cm, (list, tuple)):
                cm = np.array(cm)

            cm = (cm * 255).astype(int)
            if cm.shape[1] == 4:
                # matplotlib color map already extracted
                for loc, color in zip(np.linspace(0, 1, len(cm)), cm):
                    r.append([loc, "rgb" + str(tuple(color[:-1]))])
            else:
                # colorcet colormap
                for loc, color in zip(np.linspace(0, 1, len(cm)), cm):
                    r.append([loc, "rgb" + str(tuple(color))])
        elif all([isinstance(c, str) for c in cm]):
            # colorcet colormap
            for loc, color in zip(np.linspace(0, 1, len(cm)), cm):
                c = ImageColor.getcolor(color, "RGB")
                r.append([loc, "rgb" + str(tuple(c))])
        elif all([isinstance(t, (float, int)) for t in cm]):
            # k3d color map
            cm = np.array(cm).reshape(-1, 4)
            colors = (cm[:, 1:] * 255).astype(int)
            for loc, color in zip(cm[:, 0], colors):
                r.append([loc, "rgb" + str(tuple(color))])
        else:
            r = cm
    elif to == "matplotlib":  # to matplotlib
        if isinstance(cm, Colormap):
            r = cm
        elif isinstance(cm, str):
            # Plotly color scale
            discr = np.linspace(0, 1, n)
            colors = np.array(get_plotly_colors(cm, discr))
            r = np.c_[colors, np.ones(len(colors))]
        elif all([isinstance(t, (float, int, np.float64)) for t in cm]):
            # k3d color map
            cm = np.array(cm).reshape(-1, 4)
            r = np.c_[cm[:, 1:], np.ones(len(cm))]
        elif isinstance(cm, np.ndarray) or all(
            [isinstance(c, (list, tuple)) for c in cm]
        ):
            if isinstance(cm, (list, tuple)):
                cm = np.array(cm)

            if cm.shape[1] == 4:
                # matplotlib color map already extracted
                r = cm
            else:
                # colorcet color map
                r = np.c_[cm, np.ones(len(cm))]
        elif all([isinstance(c, str) for c in cm]):
            # colorcet colormap
            colors = [ImageColor.getcolor(color, "RGB") for color in cm]
            colors = np.array(colors) / 255
            r = np.c_[colors, np.ones(len(colors))]
        else:
            r = cm
    else:  # to bokeh
        if isinstance(cm, Colormap):
            # matplotlib color map
            discr = np.linspace(0, 1, n)
            colors = (cm(discr) * 255).astype(int)
            r = ["#%02x%02x%02x" % tuple(c[:-1]) for c in colors]
        elif isinstance(cm, str):
            # Plotly color scale
            discr = np.linspace(0, 1, n)
            colors = np.array(get_plotly_colors(cm, discr))
            colors = (colors * 255).astype(np.uint8)
            r = ["#%02x%02x%02x" % tuple(c) for c in colors]
        elif isinstance(cm, np.ndarray) or all(
            [isinstance(c, (list, tuple)) for c in cm]
        ):
            if isinstance(cm, (list, tuple)):
                cm = np.array(cm)
            colors = (cm * 255).astype(int)

            if cm.shape[1] == 4:
                # matplotlib color map already extracted
                r = ["#%02x%02x%02x" % tuple(c[:-1]) for c in colors]
            else:
                # colorcet color map
                r = ["#%02x%02x%02x" % tuple(c) for c in colors]
        elif all([isinstance(t, (float, int)) for t in cm]):
            # k3d color map
            cm = np.array(cm).reshape(-1, 4)
            colors = (cm[:, 1:] * 255).astype(int)
            r = ["#%02x%02x%02x" % tuple(c) for c in colors]
        else:
            r = cm
    return r


def _get_continuous_color(colorscale, intermed):
    """Computes the intermediate color for any value in the [0, 1] range of a
    Plotly color scale.

    From: https://stackoverflow.com/a/64655638/2329968

    Parameters
    ==========

        colorscale : list
            A plotly colorscale in the form:
            [[loc1, "rgb1"], [loc2, "rgb2"], ...] where loc is the location
            in the range [0, 1] and "rgb1" is a string representing and RGB
            color.

        intermed : float
            Value in the range [0, 1]

    Returns
    =======
        color : str
            An RGB color string in which the components are float numbers in
            the range [0, 255].
    """
    plotly = import_module(
        'plotly',
        import_kwargs={'fromlist': ['colors']},
        min_module_version='5.0.0')

    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


def get_plotly_colors(colorscale_name, loc):
    """Extract the color at the specified location from the specified Plotly's
    color scale.

    Parameters
    ==========

        colorscale_name : str
            Name of Plotly's color scale.

        loc : float or iterable
            Location in the range [0, 1]

    Returns
    =======
        An RGB list with components in the range [0, 1] or a list of RGB lists.
    """
    _plotly_utils = import_module(
        '_plotly_utils',
        import_kwargs={'fromlist': ['basevalidators']})

    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter for our use cae
    cv = _plotly_utils.basevalidators.ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        str_colors = [_get_continuous_color(colorscale, x) for x in loc]
        return [
            [float(t) / 255 for t in s[4:-1].split(",")] for s in str_colors
        ]

    str_color = _get_continuous_color(colorscale, loc)
    return [float(t) / 255 for t in str_color[4:-1].split(",")]


def get_seeds_points_entry_vector(xx, yy, zz, uu, vv, ww):
    """Returns an optimal list of seeds points to be used to generate 3D
    streamlines.

    Parameters
    ==========
        xx, yy, zz: np.ndarray
            3D discretization of the space from meshgrid.

        uu, vv, ww: np.ndarray
            Vector components calculated at the discretized points in space.

    Returns
    =======
        points : np.ndarray
            [n x 3] matrix of seed-points coordinates.
    """
    np = import_module('numpy')

    coords = np.stack([xx, yy, zz])
    # vector field
    vf = np.stack([uu, vv, ww])

    # extract the coordinate of the points at planes:
    # x_min, x_max, y_min, y_max, z_min, z_max
    c_xmin = coords[:, :, 0, :]
    c_xmax = coords[:, :, -1, :]
    c_ymin = coords[:, 0, :, :]
    c_ymax = coords[:, -1, :, :]
    c_zmin = coords[:, :, :, 0]
    c_zmax = coords[:, :, :, -1]

    # extract the vector field points at planes:
    # x_min, x_max, y_min, y_max, z_min, z_max
    vf_xmin = vf[:, :, 0, :]
    vf_xmax = vf[:, :, -1, :]
    vf_ymin = vf[:, 0, :, :]
    vf_ymax = vf[:, -1, :, :]
    vf_zmin = vf[:, :, :, 0]
    vf_zmax = vf[:, :, :, -1]

    def find_points_at_input_vectors(vf_plane, coords, i, sign="g"):
        check = {
            "g": lambda x: x > 0,
            "l": lambda x: x < 0,
        }
        # extract coordinates where the vectors are entering the plane
        tmp = np.where(
            check[sign](vf_plane[i, :, :]),
            coords,
            np.nan * np.zeros_like(coords)
        )
        # reshape the matrix to obtain an [n x 3] array of coordinates
        tmp = np.array(
            [
                tmp[0, :, :].flatten(),
                tmp[1, :, :].flatten(),
                tmp[2, :, :].flatten()]
        ).T
        # remove NaN entries
        tmp = [a for a in tmp if not np.all([np.isnan(t) for t in a])]
        return tmp

    p_xmin = find_points_at_input_vectors(vf_xmin, c_xmin, 0, "g")
    p_xmax = find_points_at_input_vectors(vf_xmax, c_xmax, 0, "l")
    p_ymin = find_points_at_input_vectors(vf_ymin, c_ymin, 1, "g")
    p_ymax = find_points_at_input_vectors(vf_ymax, c_ymax, 1, "l")
    p_zmin = find_points_at_input_vectors(vf_zmin, c_zmin, 2, "g")
    p_zmax = find_points_at_input_vectors(vf_zmax, c_zmax, 2, "l")
    # TODO: there could be duplicates
    points = np.array(p_xmin + p_xmax + p_ymin + p_ymax + p_zmin + p_zmax)

    return points


def get_seeds_points(xx, yy, zz, uu, vv, ww, to_numpy=True, **kw):
    """
    Parameters
    ==========

    xx, yy, zz, uu, vv, ww : np.ndarray [n x m x r]
        Discretized volume and vector components

    to_numpy : boolean (Default to True)
        If True, return a [N x 3] numpy array of coordinates. If False, return
        a vtk object representing seeds points.

    kw : dict
        Keyword arguments controlling the generation of streamlines.


    Returns
    =======

    seeds : np.ndarray [N x 3] or a vtk object
        Depending on the value of ``to_numpy``:

        - ``True``: numpy matrix [N x 3] of x-y-z coordinates of the
          streamtubes, which also contains NaN values. Think of the streamtubes
          as a single long tube: NaN values separate the different sections.
        - ``False``: a vtk object representing the seeds points.

    """
    np = import_module('numpy')
    import vtk
    from vtk.util import numpy_support
    starts = kw.get("starts", None)

    if starts is None:
        points = get_seeds_points_entry_vector(xx, yy, zz, uu, vv, ww)

        if to_numpy:
            return points

        seeds = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        for p in points:
            vtk_points.InsertNextPoint(p)
        seeds.SetPoints(vtk_points)
        return seeds

    elif isinstance(starts, dict):
        if not all([t in starts.keys() for t in ["x", "y", "z"]]):
            raise KeyError(
                "``starts`` must contains the following keys: "
                + "'x', 'y', 'z', whose values are going to be "
                + "lists of coordinates."
            )
        x, y, z = starts["x"], starts["y"], starts["z"]
        x, y, z = [
            t if not isinstance(t, np.ndarray) else t.flatten()
            for t in [x, y, z]
        ]
        points = np.array([x, y, z]).T

        if to_numpy:
            return points

        seeds = vtk.vtkPolyData()
        seeds_points = vtk.vtkPoints()
        for p in points:
            seeds_points.InsertNextPoint(p)
        seeds.SetPoints(seeds_points)
        return seeds

    else:
        # generate a random cloud of points
        npoints = kw.get("npoints", 200)
        radius = kw.get("radius", None)
        center = 0, 0, 0

        if not radius:
            xmin, xmax = min(xx[0, :, 0]), max(xx[0, :, 0])
            ymin, ymax = min(yy[:, 0, 0]), max(yy[:, 0, 0])
            zmin, zmax = min(zz[0, 0, :]), max(zz[0, 0, :])
            radius = max(
                [abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)]
            )
            center = (xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2

        seeds = vtk.vtkPointSource()
        seeds.SetRadius(radius)
        seeds.SetCenter(*center)
        seeds.SetNumberOfPoints(npoints)

        if to_numpy:
            seeds.Update()
            source = seeds.GetOutput()
            # Extract the points from the point cloud.
            points = numpy_support.vtk_to_numpy(source.GetPoints().GetData())
            return points

        return seeds


def compute_streamtubes(xx, yy, zz, uu, vv, ww, kwargs, color_func=None, ):
    """ Leverage vtk to compute streamlines in a 3D vector field.

    Parameters
    ==========

    xx, yy, zz, uu, vv, ww : np.ndarray [n x m x r]
        Discretized volume and vector components

    kwargs : dict
        Keyword arguments passed to the backend.

    color_func : callable
        The color function to apply to streamlines. Default to None, which
        will apply the magnitude of the vector field.


    Returns
    =======

    vertices : np.ndarray [Nx3]
        A matrix of x-y-z coordinates of the streamtubes, which also
        contains NaN values. Think of the streamtubes as a single long
        tube: NaN values separate the different sections.

    attributes : np.ndarray [N]
        An array containing the magnitudes of the streamtubes. It also
        contains NaN values.


    Notes
    =====

    To compute streamlines in a 3D vector field there are multiple options:

    * custom built integrator. Requires times to be implemented and properly
      tested.
    * vtk, which is an "heavy" dependency (around 60MB) and (within this
      module) is only used to compute streamlines in 3D vector fields.
    * yt, which is also an "heavy" dependency (around 60MB).
    * one may erroneously think of Plotly as an alternative. Turns out that
      Plotly uses a
      `JS based shader <https://github.com/gl-vis/gl-streamtube3d>`_ from which
      it is very difficult or impossible to extract the necessary mesh data
      directly from Python.

    Hence, vtk is used. The interface provided by this function deliberately
    extend the one provided by Plotly's Streamtube class. Read ``plot_vector``
    docstring for more information.
    """
    np = import_module('numpy')
    import vtk
    from vtk.util import numpy_support
    n2, n1, n3 = xx.shape

    vector_field = np.array([uu.flatten(), vv.flatten(), ww.flatten()]).T
    vtk_vector_field = numpy_support.numpy_to_vtk(
        num_array=vector_field, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_vector_field.SetName("vector_field")

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(n2 * n1 * n3)
    for i, (x, y, z) in enumerate(
        zip(xx.flatten(), yy.flatten(), zz.flatten())
    ):
        points.SetPoint(i, [x, y, z])

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions([n2, n1, n3])
    grid.SetPoints(points)
    grid.GetPointData().SetVectors(vtk_vector_field)

    # copy the dictionary: if multiple vector fields are being plotted
    # simultaneously, we need the original again.
    kwargs = kwargs.copy()
    starts = kwargs.get("starts", None)
    max_prop = kwargs.pop("max_prop", 5000)

    streamer = vtk.vtkStreamTracer()
    streamer.SetInputData(grid)
    streamer.SetMaximumPropagation(max_prop)

    seeds = get_seeds_points(
        xx, yy, zz, uu, vv, ww, to_numpy=False, **kwargs)

    if starts is None:
        streamer.SetSourceData(seeds)
        streamer.SetIntegrationDirectionToForward()
    elif isinstance(starts, dict):
        streamer.SetSourceData(seeds)
        streamer.SetIntegrationDirectionToBoth()
    else:
        streamer.SetSourceConnection(seeds.GetOutputPort())
        streamer.SetIntegrationDirectionToBoth()

    streamer.SetComputeVorticity(0)
    streamer.SetIntegrator(vtk.vtkRungeKutta4())
    streamer.Update()

    streamline = streamer.GetOutput()
    streamlines_points = numpy_support.vtk_to_numpy(
        streamline.GetPoints().GetData())
    streamlines_velocity = numpy_support.vtk_to_numpy(
        streamline.GetPointData().GetArray("vector_field"))

    if color_func is None:
        streamlines_speed = np.linalg.norm(streamlines_velocity, axis=1)
    else:
        x, y, z = streamlines_points.T
        u, v, w = streamlines_velocity.T
        streamlines_speed = color_func(x, y, z, u, v, w)

    vtkLines = streamline.GetLines()
    vtkLines.InitTraversal()
    point_list = vtk.vtkIdList()

    # extract vtk data to lists
    lines = []
    lines_attributes = []
    while vtkLines.GetNextCell(point_list):
        start_id = point_list.GetId(0)
        end_id = point_list.GetId(point_list.GetNumberOfIds() - 1)
        l = []
        v = []

        for i in range(start_id, end_id):
            l.append(streamlines_points[i])
            v.append(streamlines_speed[i])

        lines.append(np.array(l))
        lines_attributes.append(np.array(v))

    # create a matrix of coordinates from all the lines previously extracted.
    # NaN values will be used by the backends to separate the different
    # streamtubes
    count = sum([len(l) for l in lines])
    vertices = np.nan * np.zeros((count + (len(lines) - 1), 3))
    attributes = np.zeros(count + (len(lines) - 1))
    c = 0
    for k, (l, a) in enumerate(zip(lines, lines_attributes)):
        vertices[c : c + len(l), :] = l
        attributes[c : c + len(l)] = a
        if k < len(lines) - 1:
            c = c + len(l) + 1

    return vertices, attributes


@modify_parameterized_doc()
class tick_formatter_multiples_of(param.Parameterized):
    """
    Create a tick formatter where each tick is a multiple of a `quantity / n`.
    This formatter is meant to be used directly by the backend classes
    (MB, BB, PB).

    Examples
    --------

    Consider a quantity, for example `pi`. Let's suppose our region is limited
    to [-2*pi, 2*pi].

    To get a major tick at multiples of `pi`, then n=1:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import *
        >>> from spb import tick_formatter_multiples_of, graphics, line
        >>> tf = tick_formatter_multiples_of(quantity=pi, label="\\pi", n=1)
        >>> x = symbols("x")
        >>> graphics(
        ...     line(cos(x), (x, -2*pi, 2*pi)),
        ...     x_ticks_formatter=tf
        ... )

    To get a major tick at multiples of `2*pi`, than n=0.5:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> tf = tick_formatter_multiples_of(quantity=pi, label="\\pi", n=0.5)
        >>> graphics(
        ...     line(cos(x), (x, -2*pi, 2*pi)),
        ...     x_ticks_formatter=tf
        ... )

    To get a major tick at multiples of `pi / 2`, then, n=2:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> tf = tick_formatter_multiples_of(quantity=pi, label="\\pi", n=2)
        >>> graphics(
        ...     line(cos(x), (x, -2*pi, 2*pi)),
        ...     x_ticks_formatter=tf
        ... )

    To get a major tick at multiples of `e` (Euler number):

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> tf = tick_formatter_multiples_of(quantity=E, label="e", n=1)
        >>> graphics(
        ...     line(sin(pi*x/E) * ln(x), (x, 0, 5*E)),
        ...     x_ticks_formatter=tf
        ... )

    Notes
    -----
    This implementation is really basic because it doesn't consider the font
    size of the tick labels, nor the width of the tick labels, nor the spacing
    between them, nor the range being visualized, etc.
    It is up to the user to select an appropriate value of the parameter `n`
    in order to achieve properly spaced tick labels thus improving readability.
    """

    quantity = param.ClassSelector(
        default=1, class_=(PythonNumber, SympyNumber, NumberSymbol), doc="""
        Numeric value of the base quantity.""")
    label = param.String(default="", doc="""
        Label associated to `quantity` to be shown on the ticks.""")
    n = param.Number(default=1, bounds=(0, None), doc="""
        Denominator of the reference quantity for placing major grid lines.""")
    n_minor = param.Integer(default=4, bounds=(0, None), doc="""
        Number of minor ticks to be shown between two consecutive major ticks.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._cast_quantity_to_float()
        self.np = import_module("numpy")
        self.bokeh = import_module(
            'bokeh',
            import_kwargs={
                'fromlist': [
                    'models'
                ]
            },
            warn_not_installed=False,
            min_module_version='3.0.0'
        )
        matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['pyplot']},
            min_module_version='3.0.0',
            catch=(RuntimeError,))
        self.plt = matplotlib.pyplot

    @param.depends("quantity", watch=True)
    def _cast_quantity_to_float(self):
        if self.quantity:
            self.quantity = float(self.quantity)

    @param.depends("label", watch=True, on_init=True)
    def _preprocess_label(self):
        # users might be passing in latex labels wrapped in $...$.
        # Remove the dollar sign.
        label = self.label.strip()
        if len(label) > 1:
            if (label[0] == "$") and (label[-1] == "$"):
                label = label[1:-1]
        label = label.strip()
        # if the label is a latex white space, remove it
        if label == r"\,":
            label = ""

        with param.discard_events(self):
            self.label = label

    def MB_func_formatter(self):
        """
        Return a function to be used by matplotlib's ``FuncFormatter``
        in order to customize the tick labels.
        """

        def formatter(value, tick_number):
            N = abs(int(self.np.round(value / (self.quantity / self.n))))
            if N == 0:
                return "$0$"

            sign = "" if value >= 0 else "-"

            if N % self.n == 0:
                # whole multiples of pi / n
                num = int(N / self.n)
                if (num == 1) and len(self.label) > 0:
                    num = ""
                # num = "" if (num == 1) else num
                return r"$%s%s%s$" % (sign, num, self.label)
            else:
                f = Fraction(N, self.n)
                num = f.numerator
                den = f.denominator
                num = "" if (num == 1) else num
                return r"$%s\frac{%s%s}{%d}$" % (sign, num, self.label, den)
        return formatter

    def MB_major_locator(self):
        """
        Returns a matplotlib ``MultipleLocator`` in order to locate major
        grid lines.
        """
        return self.plt.MultipleLocator(self.quantity / self.n)

    def MB_minor_locator(self):
        """
        Returns a matplotlib ``MultipleLocator`` in order to locate minor
        grid lines.
        """
        den = (self.n_minor + 1) * self.n
        return self.plt.MultipleLocator(self.quantity / den)

    def BB_formatter(self):
        """
        Returns a bokeh ``CustomJSTickFormatter`` in order to customize the
        tick labels.
        """
        return self.bokeh.models.CustomJSTickFormatter(code=f"""
            const N = {self.n};
            const step = {self.quantity} / N;
            const k = Math.round(tick / step);  // integer multiple of quantity/N

            if (k === 0) return "0";

            function gcd(a, b) {{ a = Math.abs(a); b = Math.abs(b);
                while (b) {{ const t = b; b = a % b; a = t; }}
                return a;
            }}

            const g = gcd(k, N);
            const num = k / g;    // numerator of k/N reduced
            const den = N / g;    // denominator reduced

            if (den === 1) {{
                if (num === 1)  return "{self.label if self.label else 1}";
                if (num === -1) return "-{self.label if self.label else 1}";
                return `${{num}}{self.label}`;
            }} else {{
                if (Math.abs(num) === 1)
                    return `${{num === -1 ? "-" : ""}}{self.label}/${{den}}`;
                return `${{num}}{self.label}/${{den}}`;
            }}
        """)

    def BB_ticker(self):
        """
        Returns a matplotlib ``MultipleLocator`` in order to locate major
        grid lines.
        """
        return self.bokeh.models.SingleIntervalTicker(
            interval=self.quantity / self.n, num_minor_ticks=(self.n_minor+1))

    def PB_ticks(self, t_min, t_max, latex=False):
        """
        Return tick values and labels for multiples of `quantity/n`
        between `t_min` and `t_max`.
        """
        step = self.quantity / self.n
        kmin = int(self.np.floor(t_min / step))
        kmax = int(self.np.ceil(t_max / step))

        tickvals = []
        ticktext = []
        for k in range(kmin, kmax+1):
            val = k * step
            tickvals.append(val)

            if k == 0:
                ticktext.append("$0$" if latex else "0")
            else:
                frac = Fraction(k, self.n).limit_denominator()
                num, den = frac.numerator, frac.denominator
                sign = '-' if num < 0 else ''
                num = abs(num)
                wrapper = "$%s$" if latex else "%s"

                if den == 1:
                    # whole multiples of pi
                    if num == 1:
                        content = f"{sign}{self.label if self.label else 1}"
                        ticktext.append(wrapper % content)
                    else:
                        content = f"{sign}{num}{self.label}"
                        ticktext.append(wrapper % content)
                else:
                    if abs(num) == 1:
                        content = (
                            rf"{sign}\frac{{{self.label}}}{{{den}}}"
                            if latex else f"{sign}{self.label}/{den}"
                        )
                        ticktext.append(wrapper % content)
                    else:
                        content = (
                            rf"{sign}\frac{{{num}{self.label}}}{{{den}}}"
                            if latex else f"{sign}{num}{self.label}/{den}"
                        )
                        ticktext.append(wrapper % content)

        return tickvals, ticktext


def multiples_of_pi(label="\\pi"):
    """
    Create a tick formatter where each tick is a multiple of pi.
    """
    # minor grid lines every pi/4
    np = import_module("numpy")
    return tick_formatter_multiples_of(quantity=np.pi, label=label, n=1, n_minor=3)


def multiples_of_2_pi(label="\\pi"):
    """
    Create a tick formatter where each tick is a multiple of 2*pi.
    """
    # minor grid lines every pi/2
    np = import_module("numpy")
    return tick_formatter_multiples_of(quantity=np.pi, label=label, n=0.5, n_minor=3)


def multiples_of_pi_over_2(label="\\pi"):
    """
    Create a tick formatter where each tick is a multiple of pi/2.
    """
    # minor grid lines every pi/8
    np = import_module("numpy")
    return tick_formatter_multiples_of(quantity=np.pi, label=label, n=2, n_minor=3)


def multiples_of_pi_over_3(label="\\pi"):
    """
    Create a tick formatter where each tick is a multiple of pi/3.
    """
    # minor grid lines every pi/12
    np = import_module("numpy")
    return tick_formatter_multiples_of(quantity=np.pi, label=label, n=3, n_minor=3)


def multiples_of_pi_over_4(label="\\pi"):
    """
    Create a tick formatter where each tick is a multiple of pi/4.
    """
    # minor grid lines every pi/16
    np = import_module("numpy")
    return tick_formatter_multiples_of(quantity=np.pi, label=label, n=4, n_minor=3)


def _get_cmin_cmax(surfacecolor, plot, series):
    """
    Given an array of values for the surface color, the plot object and
    the data series, returns the appropriate minimum and maximum color values
    to be used in the colorbar.
    """
    cmin = surfacecolor.min()
    cmax = surfacecolor.max()
    # if clipping planes are present in the z-direction, and colormap is
    # to be used, and the colorfunc is not set (hence, default colormap
    # according to z-value is used), then the maximum and minimum values
    # shown on the colorbar take into consideration the zlim values.
    returns_z_val = _returns_z_coord(series.color_func)
    if returns_z_val and plot.zlim and (plot.zlim[0] > cmin):
        cmin = plot.zlim[0]
    if returns_z_val and plot.zlim and (plot.zlim[1] < cmax):
        cmax = plot.zlim[1]
    return cmin, cmax


def _returns_z_coord(func):
    """
    Attempt to verify if the `color_func` attribute of the surface series
    only returns the z-coordinate.

    Notes
    -----
    The defualt color_func are:

    * f = lambda x, y, z: z (for instances of SurfaceOver2DRangeSeries)
    * f = lambda x, y, z, u, v: z (for instances of ParametricSurfaceSeries)

    This approach attempts to evaluate the provided `func` with dummy objects.
    Parsing the source code of func would be very difficult, because
    when defining lambda functions inside another function call, or __init__
    method, `inspect.getsource` returns a lot more than just the lambda
    function.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    params = list(sig.parameters.values())
    if len(params) < 3:
        return False

    third = params[2]

    # Only allow true positional
    if third.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return False

    # Build args list: first 3 unique sentinels, rest fillers
    sentinels = [object(), object(), object()]
    fillers = [object()] * (len(params) - 3)
    args = sentinels + fillers

    try:
        result = func(*args)
    except Exception:
        return False

    # Must return the exact third positional arg
    return result is sentinels[2]
