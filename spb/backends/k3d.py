from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.utils import convert_colormap
from spb.utils import get_vertices_indices, get_seeds_points
import k3d
import numpy as np
import warnings
from matplotlib.tri import Triangulation
import itertools
from mergedeep import merge
import colorcet as cc
from PIL import ImageColor
import matplotlib.cm as cm

# TODO:
# 1. load the plot with menu minimized
# 2. iplot support for streamlines?


class K3DBackend(Plot):
    """ A backend for plotting SymPy's symbolic expressions using K3D-Jupyter.

    Keyword Arguments
    =================
        
        line_kw : dict
            A dictionary of keywords/values which is passed to K3D's line
            functions to customize the appearance. Default to:
            ``line_kw = dict(width=0.1, shader="mesh")``
            Set `use_cm=False` to switch to a solid color.
        
        quiver_kw : dict
            A dictionary to customize the apppearance of quivers. Default to:
            ``quiver_kw = dict(scale = 1)``.
            Set `use_cm=False` to switch to a solid color.
        
        show_label : boolean
            Show/hide labels of the expressions. Default to False (labels not
            visible).
        
        stream_kw : dict
            A dictionary to customize the apppearance of streamlines.
            Default to:
            ``stream_kw = dict( width=0.1, shader='mesh' )``
            Refer to k3d.line for more options.
            Set `use_cm=False` to switch to a solid color.
        
        use_cm : boolean
            If True, apply a color map to the meshes/surface. If False, solid
            colors will be used instead. Default to True.
    """

    _library = "k3d"

    # TODO: better selection of colormaps
    colormaps = [
        k3d.basic_color_maps.CoolWarm, k3d.basic_color_maps.Jet,
        k3d.basic_color_maps.BlackBodyRadiation, k3d.matplotlib_color_maps.Plasma,
        k3d.matplotlib_color_maps.Autumn, k3d.matplotlib_color_maps.Winter,
        k3d.paraview_color_maps.Nic_Edge, k3d.paraview_color_maps.Haze
    ]

    cyclic_colormaps = [
        cc.colorwheel, k3d.paraview_color_maps.Erdc_iceFire_H
    ]

    quivers_colormaps = [
        k3d.basic_color_maps.CoolWarm, k3d.matplotlib_color_maps.Plasma,
        k3d.matplotlib_color_maps.Winter, k3d.matplotlib_color_maps.Viridis,
        k3d.paraview_color_maps.Haze, k3d.matplotlib_color_maps.Summer,
        k3d.paraview_color_maps.Blue_to_Yellow
    ]
    
    def __new__(cls, *args, **kwargs):
        # Since Plot has its __new__ method, this will prevent infinite
        # recursion
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._get_mode() != 0:
            raise ValueError(
                    "Sorry, K3D backend only works within Jupyter Notebook")

        self._bounds = []
        self._clipping = []
        self._handles = dict()

        self._init_cyclers()

        self._fig = k3d.plot(
            grid_visible = self.grid,
            menu_visibility = True,
            background_color = int(cfg["k3d"]["bg_color"])
        )
        if (self.xscale == "log") or (self.yscale == "log"):
            warnings.warn("K3D-Jupyter doesn't support log scales. We will " +
                         "continue with linear scales.")
        self.plot_shown = False
        self._process_series(self._series)

    @staticmethod
    def _int_to_rgb(RGBint):
        """ Convert an integer number to an RGB tuple with components from 0 to
        255.

        https://stackoverflow.com/a/2262152/2329968
        """
        B =  RGBint & 255
        G = (RGBint >> 8) & 255
        R =   (RGBint >> 16) & 255
        return R, G, B
    
    @staticmethod
    def _rgb_to_int(RGB):
        """ Convert an RGB tuple to an integer number.

        https://stackoverflow.com/q/2262100/2329968
        """
        R, G, B = RGB
        return R * 256**2 + G * 256 + B
    
    @classmethod
    def _convert_to_int(cls, color):
        """ Convert the provided RGB tuple with values from 0 to 1 to an integer
        number.
        """
        color = [int(c * 255) for c in color]
        return cls._rgb_to_int(color)

    def _init_cyclers(self):
        super()._init_cyclers()
        quivers_colormaps = [convert_colormap(cm, self._library) for cm
                in self.quivers_colormaps]
        self._qcm = itertools.cycle(quivers_colormaps)

    def _process_series(self, series):
        self._init_cyclers()
        self._fig.auto_rendering = False
        # clear data
        for o in self._fig.objects:
            self._fig.remove_class(o)

        for ii, s in enumerate(series):
            if s.is_3Dline:
                x, y, z, param = s.get_data()
                vertices = np.vstack([x, y, z]).T.astype(np.float32)
                # keyword arguments for the line object
                a = dict(
                    width = 0.1,
                    name = s.label if self._kwargs.get("show_label", False) else None,
                    color = self._convert_to_int(next(self._cl)),
                    shader = "mesh",
                )
                if self._use_cm:
                    a["attribute"] = param.astype(np.float32),
                    a["color_map"] = next(self._cm)
                    a["color_range"] = [s.start, s.end]
                line_kw = self._kwargs.get("line_kw", dict())
                line = k3d.line(vertices, **merge({}, a, line_kw))
                self._fig += line

            elif s.is_3Dsurface and (not s.is_complex):
                x, y, z = s.get_data()
                print("K3D is_surface", 
                    len(x) if not hasattr(x, "shape") else x.shape,
                    len(y) if not hasattr(y, "shape") else y.shape,
                    len(z) if not hasattr(z, "shape") else z.shape,
                )

                # TODO:
                # Can I use get_vertices_indices also for non parametric surfaces?

                if s.is_parametric:
                    print("K3D is_parametric surface")
                    vertices, indices = get_vertices_indices(x, y, z)
                    vertices = vertices.astype(np.float32)
                else:
                    print("K3D is_surface")
                    x = x.flatten()
                    y = y.flatten()
                    z = z.flatten()
                    print(np.amin(z), np.amax(z))
                    vertices = np.vstack([x, y, z]).T.astype(np.float32)
                    indices = Triangulation(x, y).triangles.astype(np.uint32)

                    # look for high aspect ratio meshes, where (dz >> dx, dy) 
                    # and eventually set the bounds around the mid point of the 
                    # mesh in order to improve visibility.
                    # Bounds will be used to set the camera position.
                    mz, Mz, meanz = z.min(), z.max(), z.mean()
                    mx, Mx = x.min(), x.max()
                    my, My = y.min(), y.max()
                    dx, dy, dz = (Mx - mx), (My - my), (Mz - mz)
                    # thresholds
                    t1, t2 = 10, 3
                    if (dz / dx >= t1) and (dz / dy >= t1):
                        if abs(Mz / meanz) > t1:
                            Mz = meanz + t2 * max(dx, dy)
                        if abs(mz / meanz) > t1:
                            mz = meanz - t2 * max(dx, dy)
                        self._bounds.append([mx, Mx, my, My, mz, Mz])

                
                a = dict(
                    name = s.label if self._kwargs.get("show_label", False) else None,
                    side = "double",
                    flat_shading = False,
                    wireframe = False,
                    color = self._convert_to_int(next(self._cl)),
                    # volume_bounds = (min(x), max(x), min(y), max(y), min(z), max(z))
                )
                if self._use_cm:
                    a["color_map"] = (next(self._cm) if not s.is_complex 
                            else next(self._cyccm))
                    a["attribute"] = z
                surface_kw = self._kwargs.get("surface_kw", dict())
                surf = k3d.mesh(vertices, indices, 
                        **merge({}, a, surface_kw))
                    
                self._fig += surf
            
            elif s.is_3Dvector and self._kwargs.get("streamlines", False):
                xx, yy, zz, uu, vv, ww = s.get_data()
                magnitude = np.sqrt(uu**2 + vv**2 + ww**2)
                min_mag = min(magnitude.flatten())
                max_mag = max(magnitude.flatten())
                
                import vtk
                from vtk.util import numpy_support

                vector_field = np.array([uu.flatten(), vv.flatten(), ww.flatten()]).T
                vtk_vector_field = numpy_support.numpy_to_vtk(num_array=vector_field, deep=True, array_type=vtk.VTK_FLOAT)
                vtk_vector_field.SetName("vector_field")

                points = vtk.vtkPoints()
                points.SetNumberOfPoints(s.n2 * s.n1 * s.n3)
                for i, (x, y, z) in enumerate(zip(xx.flatten(), yy.flatten(), zz.flatten())):
                    points.SetPoint(i, [x, y, z])
                    
                grid = vtk.vtkStructuredGrid()
                grid.SetDimensions([s.n2, s.n1, s.n3])
                grid.SetPoints(points)
                grid.GetPointData().SetVectors( vtk_vector_field )

                stream_kw = self._kwargs.get("stream_kw", dict())
                starts = stream_kw.pop("starts", None)
                max_prop = stream_kw.pop("max_prop", 500)

                streamer = vtk.vtkStreamTracer()
                streamer.SetInputData(grid)
                streamer.SetMaximumPropagation(max_prop)
                
                if starts is None:
                    seeds_points = get_seeds_points(xx, yy, zz, uu, vv, ww)
                    seeds = vtk.vtkPolyData()
                    points = vtk.vtkPoints()
                    for p in seeds_points:
                        points.InsertNextPoint(p)
                    seeds.SetPoints(points)
                    streamer.SetSourceData(seeds)
                    streamer.SetIntegrationDirectionToForward()
                elif isinstance(starts, dict):
                    if not all([t in starts.keys() for t in ["x", "y", "z"]]):
                        raise KeyError(
                            "``starts`` must contains the following keys: " +
                            "'x', 'y', 'z', whose values are going to be " +
                            "lists of coordinates.")
                    seeds_points = np.array([starts["x"], starts["y"], starts["z"]]).T
                    seeds = vtk.vtkPolyData()
                    points = vtk.vtkPoints()
                    for p in seeds_points:
                        points.InsertNextPoint(p)
                    seeds.SetPoints(points)
                    streamer.SetSourceData(seeds)
                    streamer.SetIntegrationDirectionToBoth()
                else:
                    npoints = stream_kw.get("npoints", 200)
                    radius = stream_kw.get("radius", None)
                    center = 0, 0, 0
                    if not radius:
                        xmin, xmax = min(xx[0, :, 0]), max(xx[0, :, 0])
                        ymin, ymax = min(yy[:, 0, 0]), max(yy[:, 0, 0])
                        zmin, zmax = min(zz[0, 0, :]), max(zz[0, 0, :])
                        radius = max([abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)])
                        center = (xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2
                    seeds = vtk.vtkPointSource()
                    seeds.SetRadius(radius)
                    seeds.SetCenter(*center)
                    seeds.SetNumberOfPoints(npoints)

                    streamer.SetSourceConnection(seeds.GetOutputPort())
                    streamer.SetIntegrationDirectionToBoth()
                
                streamer.SetComputeVorticity(0)
                streamer.SetIntegrator(vtk.vtkRungeKutta4())
                streamer.Update()

                streamline = streamer.GetOutput()
                streamlines_points = numpy_support.vtk_to_numpy(streamline.GetPoints().GetData())
                streamlines_velocity = numpy_support.vtk_to_numpy(streamline.GetPointData().GetArray('vector_field'))
                streamlines_speed = np.linalg.norm(streamlines_velocity, axis=1)
                
                vtkLines = streamline.GetLines()
                vtkLines.InitTraversal()
                point_list = vtk.vtkIdList()

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

                count = sum([len(l) for l in lines])
                vertices = np.nan * np.zeros((count + (len(lines) - 1), 3))
                attributes = np.zeros(count + (len(lines) - 1))
                c = 0
                for k, (l, a) in enumerate(zip(lines, lines_attributes)):
                    vertices[c : c + len(l), :] = l
                    attributes[c : c + len(l)] = a
                    if k < len(lines) - 1:
                        c = c + len(l) + 1

                skw = dict( width=0.1, shader='mesh', compression_level=9 )
                if self._use_cm and ("color" not in stream_kw.keys()):
                    skw["color_map"] = next(self._qcm)
                    skw["color_range"] = [min_mag, max_mag]
                    skw["attribute"] = attributes
                else:
                    col = stream_kw.pop("color", next(self._cl))
                    if not isinstance(col, int):
                        col = self._convert_to_int(col)
                    stream_kw["color"] = col

                self._fig += k3d.line(
                    vertices.astype(np.float32),
                    **merge({}, skw, stream_kw)
                )
            elif s.is_3Dvector:
                xx, yy, zz, uu, vv, ww = s.get_data()
                xx, yy, zz, uu, vv, ww = [t.flatten().astype(np.float32) for t
                    in [xx, yy, zz, uu, vv, ww]]
                # default values
                qkw = dict(scale = 1)
                # user provided values
                quiver_kw = self._kwargs.get("quiver_kw", dict())
                qkw = merge(qkw, quiver_kw)
                scale = qkw["scale"]
                magnitude = np.sqrt(uu**2 + vv**2 + ww**2)
                vectors = np.array((uu, vv, ww)).T * scale
                origins = np.array((xx, yy, zz)).T
                if self._use_cm and ("color" not in quiver_kw.keys()):
                    colormap = next(self._qcm)
                    colors = k3d.helpers.map_colors(magnitude, colormap, [])
                    self._handles[ii] = [qkw, colormap]
                else:
                    col = quiver_kw.get("color", next(self._cl))
                    if not isinstance(col, int):
                        col = self._convert_to_int(col)
                    colors = col * np.ones(len(magnitude))
                    self._handles[ii] = [qkw, None]
                vec_colors = np.zeros(2 * len(colors))
                for i, c in enumerate(colors):
                    vec_colors[2 * i] = c
                    vec_colors[2 * i + 1] = c
                vec_colors = vec_colors.astype(np.uint32)
                vec = k3d.vectors(
                    origins = origins - vectors / 2,
                    vectors = vectors,
                    colors = vec_colors,
                )
                self._fig += vec
            elif s.is_complex and (not s.is_domain_coloring):
                x, y, z, mag, arg = s.get_data()
                print("K3D is_surface", 
                    len(x) if not hasattr(x, "shape") else x.shape,
                    len(y) if not hasattr(y, "shape") else y.shape,
                    len(z) if not hasattr(z, "shape") else z.shape,
                )

                x = x.flatten()
                y = y.flatten()
                z = mag.flatten()
                print(np.amin(mag), np.amax(mag))
                vertices = np.vstack([x, y, z]).T.astype(np.float32)
                indices = Triangulation(x, y).triangles.astype(np.uint32)

                # look for high aspect ratio meshes, where (dz >> dx, dy) 
                # and eventually set the bounds around the mid point of the 
                # mesh in order to improve visibility.
                # Bounds will be used to set the camera position.
                mz, Mz, meanz = z.min(), z.max(), z.mean()
                mx, Mx = x.min(), x.max()
                my, My = y.min(), y.max()
                dx, dy, dz = (Mx - mx), (My - my), (Mz - mz)
                # thresholds
                t1, t2 = 10, 3
                if (dz / dx >= t1) and (dz / dy >= t1):
                    if abs(Mz / meanz) > t1:
                        Mz = meanz + t2 * max(dx, dy)
                    if abs(mz / meanz) > t1:
                        mz = meanz - t2 * max(dx, dy)
                    self._bounds.append([mx, Mx, my, My, mz, Mz])

                
                a = dict(
                    name = s.label if self._kwargs.get("show_label", False) else None,
                    side = "double",
                    flat_shading = False,
                    wireframe = False,
                    color = self._convert_to_int(next(self._cl)),
                    color_range = [-np.pi, np.pi]
                )
                if self._use_cm:
                    a["color_map"] = (next(self._cm) if not s.is_complex 
                            else next(self._cyccm))
                    a["attribute"] = arg.flatten().astype(np.float32)
                surface_kw = self._kwargs.get("surface_kw", dict())
                surf = k3d.mesh(vertices, indices, 
                        **merge({}, a, surface_kw))
                    
                self._fig += surf
            else:
                raise NotImplementedError(
                    "{} is not supported by {}\n".format(type(s), type(self).__name__) +
                    "K3D-Jupyter only supports 3D plots."
                )
        
        xl = self.xlabel if self.xlabel else "x"
        yl = self.ylabel if self.ylabel else "y"
        zl = self.zlabel if self.zlabel else "z"
        self._fig.axes = [xl, yl, zl]

        if self.title:
            self._fig += k3d.text2d(self.title, 
                 position=[0.025, 0.015], color=0, size=1, label_box=False)
        self._fig.auto_rendering = True
    
    def _update_interactive(self, params):
        # self._fig.auto_rendering = False
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].update_data(params)
                if s.is_3Dline:
                    x, y, z = self.series[i].get_data()
                    vertices = np.vstack([x, y, z]).T.astype(np.float32)
                    self._fig.objects[i].vertices = vertices

                elif s.is_3Dsurface and (not s.is_complex):
                    x, y, z = self.series[i].get_data()
                    x, y, z = [t.flatten().astype(np.float32) for t in [x, y, z]]
                    vertices = np.vstack([x, y, z]).astype(np.float32)
                    self._fig.objects[i].vertices = vertices.T
                    self._fig.objects[i].attribute = z
                    self._fig.objects[i].color_range = [z.min(), z.max()]

                elif s.is_vector and s.is_3D:
                    if self._kwargs.get("streamlines", False):
                        raise NotImplementedError

                    xx, yy, zz, uu, vv, ww = self.series[i].get_data()
                    xx, yy, zz, uu, vv, ww = [t.flatten().astype(np.float32) 
                        for t in [xx, yy, zz, uu, vv, ww]]
                    magnitude = np.sqrt(uu**2 + vv**2 + ww**2)
                    qkw, colormap = self._handles[i]
                    if colormap is not None:
                        colors = k3d.helpers.map_colors(magnitude, colormap, [])
                        vec_colors = np.zeros(2 * len(colors))
                        for j, c in enumerate(colors):
                            # print("j", j, "c", c)
                            vec_colors[2 * j] = c
                            vec_colors[2 * j + 1] = c
                        vec_colors = vec_colors.astype(np.uint32)
                        self.fig.objects[i].colors = vec_colors
                        
                    scale = qkw["scale"]
                    vectors = np.array((uu, vv, ww)).T * scale
                    self.fig.objects[i].vectors = vectors

                elif s.is_complex:
                    if s.is_3Dsurface:
                        x, y, z, mag, arg = s.get_data()
                        x, y, z = [t.flatten().astype(np.float32) for t in [x, y, mag]]
                        vertices = np.vstack([x, y, z]).astype(np.float32)
                        self._fig.objects[i].vertices= vertices.T
                        self._fig.objects[i].attribute = arg.flatten().astype(np.float32)
                    else:
                        raise NotImplementedError
        # self._fig.auto_rendering = True

    def _get_auto_camera(self, factor=1.5, yaw=40, pitch=60):
        """ This function is very similar to k3d.plot.Plot.get_auto_camera.
        However, it uses 
        """
        bounds = np.array(self._bounds)
        bounds = np.dstack([np.min(bounds[:, 0::2], axis=0), 
                np.max(bounds[:, 1::2], axis=0)]).flatten()
        center = (bounds[::2] + bounds[1::2]) / 2.0
        radius = 0.5 * np.sum(np.abs(bounds[::2] - bounds[1::2]) ** 2) ** 0.5
        cam_distance = radius * factor / np.sin(np.deg2rad(self._fig.camera_fov / 2.0))
        
        x = np.sin(np.deg2rad(pitch)) * np.cos(np.deg2rad(yaw))
        y = np.sin(np.deg2rad(pitch)) * np.sin(np.deg2rad(yaw))
        z = np.cos(np.deg2rad(pitch))

        if pitch not in [0, 180]:
            up = [0, 0, 1]
        else:
            up = [0, 1, 1]

        asd = [
            center[0] + x * cam_distance,
            center[1] + y * cam_distance,
            center[2] + z * cam_distance,
            *center,
            *up
        ]
        return asd

    def show(self):
        # self._process_series(self._series)
        self.plot_shown = True
        # self._fig.grid = [-2, 2, -2, 2, 0, 10]
        if len(self._bounds) > 0:
            self._fig.camera_auto_fit = False
            self._fig.camera = self._get_auto_camera()
        self._fig.display()
        if self.zlim:
            self._fig.clipping_planes = [
                [0, 0, 1, self.zlim[0]],
                [0, 0, -1, self.zlim[1]]
            ]
    
    def save(self, path, **kwargs):
        if not self.plot_shown:
            raise ValueError(
                "K3D-Jupyter requires the plot to be shown on the screen " + 
                "before saving it."
            )

        @self._fig.yield_screenshots
        def _func():
            self._fig.fetch_screenshot()
            screenshot = yield
            with open(path, 'wb') as f:
                f.write(screenshot)
        _func()

KB = K3DBackend