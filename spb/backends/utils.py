from PIL import ImageColor
from sympy.external import import_module


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
            Number of discretization points in the range [0, 1]. Default to 256.
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
    assert to in ["matplotlib", "plotly", "k3d", "bokeh"]
    if not isinstance(cm, (str, list, tuple, np.ndarray, Colormap)):
        raise ValueError(
            "`cm` must be either:\n"
            + "1. a string with the name of a Plotly color scale.\n"
            + "2. a list of string HEX colors (colorcet colormaps).\n"
            + "2. a list of float numbers between 0 and 1 (k3d colormaps).\n"
            + "3. an instance of matplotlib.colors.Colormap.\n"
            + "4. an array of colors extracted from a matplotlib.colors.Colormap."
        )

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
            r = [[loc, "rgb" + str(tuple(c[:-1]))] for loc, c in zip(discr, colors)]
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
            An RGB color string in which the components are float numbers in the
            range [0, 255].
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
        return [[float(t) / 255 for t in s[4:-1].split(",")] for s in str_colors]

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
            check[sign](vf_plane[i, :, :]), coords, np.nan * np.zeros_like(coords)
        )
        # reshape the matrix to obtain an [n x 3] array of coordinates
        tmp = np.array(
            [tmp[0, :, :].flatten(), tmp[1, :, :].flatten(), tmp[2, :, :].flatten()]
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
        x, y, z = [t if not isinstance(t, np.ndarray) else t.flatten()
            for t in [x, y, z]]
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


def compute_streamtubes(xx, yy, zz, uu, vv, ww, kwargs):
    """ Leverage vtk to compute streamlines in a 3D vector field.

    Parameters
    ==========

    xx, yy, zz, uu, vv, ww : np.ndarray [n x m x r]
        Discretized volume and vector components

    kwargs : dict
        Keyword arguments passed to the backend.


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

    seeds = get_seeds_points(xx, yy, zz, uu, vv, ww,
        to_numpy=False, **kwargs)

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
    streamlines_speed = np.linalg.norm(streamlines_velocity, axis=1)

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
