from collections.abc import Callable
from sympy import sympify, Tuple
from sympy.core.relational import (Equality, GreaterThan, LessThan,
                Relational, StrictLessThan, StrictGreaterThan)
from sympy.external import import_module
from sympy.plotting.experimental_lambdify import (
    vectorized_lambdify, lambdify, experimental_lambdify)
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.function import arity
from sympy.core.compatibility import is_sequence
from sympy.plotting.intervalmath import interval
from spb.utils import _unpack_args, get_lambda
import warnings
import numpy as np

##############################################################################
# Data Series
##############################################################################
#TODO more general way to calculate aesthetics (see get_color_array)

### The base class for all series
class BaseSeries:
    """Base class for the data objects containing stuff to be plotted.

    Explanation
    ===========

    The backend should check if it supports the data series that it's given.
    (eg TextBackend supports only LineOver1DRange).
    It's the backend responsibility to know how to use the class of
    data series that it's given.

    Some data series classes are grouped (using a class attribute like is_2Dline)
    according to the api they present (based only on convention). The backend is
    not obliged to use that api (eg. The LineOver1DRange belongs to the
    is_2Dline group and presents the get_points method, but the
    TextBackend does not use the get_points method).
    """

    # Some flags follow. The rationale for using flags instead of checking base
    # classes is that setting multiple flags is simpler than multiple
    # inheritance.

    is_2Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dsurface = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_contour = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_implicit = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x (1D array), mesh_y(1D array,
    #     mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes
    # Different from is_contour as the colormap in backend will be
    # different

    is_parametric = False
    # The calculation of aesthetics expects:
    #   - get_parameter_points returning one or two np.arrays (1D or 2D)
    # used for calculation aesthetics

    is_interactive = False
    # An interactive series can update its data.

    def __init__(self):
        super().__init__()

    @property
    def is_3D(self):
        flags3D = [
            self.is_3Dline,
            self.is_3Dsurface
        ]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [
            self.is_2Dline,
            self.is_3Dline
        ]
        return any(flagslines)
    
    def get_data(self):
        """ All child series should implement this method to return the
        numerical data which can be used by a plotting library.
        """
        raise NotImplementedError


### 2D lines
class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - adding the label, steps and only_integers options
    - making is_2Dline true
    - defining get_segments and get_color_array
    """

    is_2Dline = True

    _dim = 2

    def __init__(self):
        super().__init__()
        self.label = None
        self.steps = False
        self.only_integers = False
        self.line_color = None

    def get_data(self):
        """ Return lists of coordinates for plotting the line.

        Returns
        =======
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates

            y: list
                List of z-coordinates in case of Parametric3DLineSeries
        """
        np = import_module('numpy')
        points = self.get_points()
        if self.steps is True:
            if len(points) == 2:
                x = np.array((points[0], points[0])).T.flatten()[1:]
                y = np.array((points[1], points[1])).T.flatten()[:-1]
                points = (x, y)
            else:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                points = (x, y, z)
        return points

    def get_segments(self):
        SymPyDeprecationWarning(
                feature="get_segments",
                issue=21329,
                deprecated_since_version="1.9",
                useinstead="MatplotlibBackend.get_segments").warn()

        np = import_module('numpy')
        points = type(self).get_data(self)
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def get_color_array(self):
        np = import_module('numpy')
        c = self.line_color
        if hasattr(c, '__call__'):
            f = np.vectorize(c)
            nargs = arity(c)
            if nargs == 1 and self.is_parametric:
                x = self.get_parameter_points()
                return f(centers_of_segments(x))
            else:
                variables = list(map(centers_of_segments, self.get_points()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables[:2])
                else:  # only if the line is 3D (otherwise raises an error)
                    return f(*variables)
        else:
            return c*np.ones(self.n)


class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    def __init__(self, list_x, list_y, label=""):
        np = import_module('numpy')
        super().__init__()
        self.list_x = np.array(list_x)
        self.list_y = np.array(list_y)
        self.label = label

    def __str__(self):
        return 'list plot'

    def get_points(self):
        return (self.list_x, self.list_y)


class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__()
        self.expr = sympify(expr)
        self.label = label
        self.var = sympify(var_start_end[0])
        self.start = float(var_start_end[1])
        self.end = float(var_start_end[2])
        self.n = kwargs.get('n', 300)
        self.adaptive = kwargs.get('adaptive', True)
        self.depth = kwargs.get('depth', 12)
        self.line_color = kwargs.get('line_color', None)
        self.xscale = kwargs.get('xscale', 'linear')

    def __str__(self):
        return 'cartesian line: %s for %s over %s' % (
            str(self.expr), str(self.var), str((self.start, self.end)))

    def get_points(self):
        """ Return lists of coordinates for plotting. Depending on the
        `adaptive` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        Returns
        =======
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates


        Explanation
        ===========

        The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
               Luiz Henrique de Figueiredo.

        """
        if self.only_integers or not self.adaptive:
            return self._uniform_sampling()
        else:
            f = lambdify([self.var], self.expr)
            x_coords = []
            y_coords = []
            np = import_module('numpy')
            def sample(p, q, depth):
                """ Samples recursively if three points are almost collinear.
                For depth < 6, points are added irrespective of whether they
                satisfy the collinearity condition or not. The maximum depth
                allowed is 12.
                """
                # Randomly sample to avoid aliasing.
                random = 0.45 + np.random.rand() * 0.1
                if self.xscale == 'log':
                    xnew = 10**(np.log10(p[0]) + random * (np.log10(q[0]) -
                                                           np.log10(p[0])))
                else:
                    xnew = p[0] + random * (q[0] - p[0])
                ynew = f(xnew)
                new_point = np.array([xnew, ynew])

                # Maximum depth
                if depth > self.depth:
                    x_coords.append(q[0])
                    y_coords.append(q[1])

                # Sample irrespective of whether the line is flat till the
                # depth of 6. We are not using linspace to avoid aliasing.
                elif depth < 6:
                    sample(p, new_point, depth + 1)
                    sample(new_point, q, depth + 1)

                # Sample ten points if complex values are encountered
                # at both ends. If there is a real value in between, then
                # sample those points further.
                elif p[1] is None and q[1] is None:
                    if self.xscale == 'log':
                        xarray = np.logspace(p[0], q[0], 10)
                    else:
                        xarray = np.linspace(p[0], q[0], 10)
                    yarray = list(map(f, xarray))
                    if any(y is not None for y in yarray):
                        for i in range(len(yarray) - 1):
                            if yarray[i] is not None or yarray[i + 1] is not None:
                                sample([xarray[i], yarray[i]],
                                    [xarray[i + 1], yarray[i + 1]], depth + 1)

                # Sample further if one of the end points in None (i.e. a
                # complex value) or the three points are not almost collinear.
                elif (p[1] is None or q[1] is None or new_point[1] is None
                        or not flat(p, new_point, q)):
                    sample(p, new_point, depth + 1)
                    sample(new_point, q, depth + 1)
                else:
                    x_coords.append(q[0])
                    y_coords.append(q[1])

            f_start = f(self.start)
            f_end = f(self.end)
            x_coords.append(self.start)
            y_coords.append(f_start)
            sample(np.array([self.start, f_start]),
                   np.array([self.end, f_end]), 0)

        return (x_coords, y_coords)

    def _uniform_sampling(self):
        np = import_module('numpy')
        if self.only_integers is True:
            if self.xscale == 'log':
                list_x = np.logspace(int(self.start), int(self.end),
                        num=int(self.end) - int(self.start) + 1)
            else:
                list_x = np.linspace(int(self.start), int(self.end),
                    num=int(self.end) - int(self.start) + 1)
        else:
            if self.xscale == 'log':
                list_x = np.logspace(self.start, self.end, num=self.n)
            else:
                list_x = np.linspace(self.start, self.end, num=self.n)
        f = vectorized_lambdify([self.var], self.expr)
        list_y = f(list_x)
        return (list_x, list_y)

class Parametric2DLineSeries(Line2DBaseSeries):
    """Representation for a line consisting of two parametric sympy expressions
    over a range."""

    is_parametric = True

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__()
        self.expr_x = sympify(expr_x)
        self.expr_y = sympify(expr_y)
        self.label = label
        self.var = sympify(var_start_end[0])
        self.start = float(var_start_end[1])
        self.end = float(var_start_end[2])
        self.n = kwargs.get('n', 300)
        self.adaptive = kwargs.get('adaptive', True)
        self.depth = kwargs.get('depth', 12)
        self.line_color = kwargs.get('line_color', None)

    def __str__(self):
        return 'parametric cartesian line: (%s, %s) for %s over %s' % (
            str(self.expr_x), str(self.expr_y), str(self.var),
            str((self.start, self.end)))

    def get_parameter_points(self):
        np = import_module('numpy')
        return np.linspace(self.start, self.end, num=self.n)

    def _uniform_sampling(self):
        param = self.get_parameter_points()
        fx = vectorized_lambdify([self.var], self.expr_x)
        fy = vectorized_lambdify([self.var], self.expr_y)
        list_x = fx(param)
        list_y = fy(param)
        return (list_x, list_y)

    def get_points(self):
        """ Return lists of coordinates for plotting. Depending on the
        `adaptive` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        Returns
        =======
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates


        Explanation
        ===========

        The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
            Luiz Henrique de Figueiredo.

        """
        if not self.adaptive:
            return self._uniform_sampling()

        f_x = lambdify([self.var], self.expr_x)
        f_y = lambdify([self.var], self.expr_y)
        x_coords = []
        y_coords = []

        def sample(param_p, param_q, p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            np = import_module('numpy')
            random = 0.45 + np.random.rand() * 0.1
            param_new = param_p + random * (param_q - param_p)
            xnew = f_x(param_new)
            ynew = f_y(param_new)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])

            # Sample irrespective of whether the line is flat till the
            # depth of 6. We are not using linspace to avoid aliasing.
            elif depth < 6:
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif ((p[0] is None and q[1] is None) or
                    (p[1] is None and q[1] is None)):
                param_array = np.linspace(param_p, param_q, 10)
                x_array = list(map(f_x, param_array))
                y_array = list(map(f_y, param_array))
                if any(x is not None and y is not None
                        for x, y in zip(x_array, y_array)):
                    for i in range(len(y_array) - 1):
                        if ((x_array[i] is not None and y_array[i] is not None) or
                                (x_array[i + 1] is not None and y_array[i + 1] is not None)):
                            point_a = [x_array[i], y_array[i]]
                            point_b = [x_array[i + 1], y_array[i + 1]]
                            sample(param_array[i], param_array[i], point_a,
                                   point_b, depth + 1)

            # Sample further if one of the end points in None (i.e. a complex
            # value) or the three points are not almost collinear.
            elif (p[0] is None or p[1] is None
                    or q[1] is None or q[0] is None
                    or not flat(p, new_point, q)):
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])

        f_start_x = f_x(self.start)
        f_start_y = f_y(self.start)
        start = [f_start_x, f_start_y]
        f_end_x = f_x(self.end)
        f_end_y = f_y(self.end)
        end = [f_end_x, f_end_y]
        x_coords.append(f_start_x)
        y_coords.append(f_start_y)
        sample(self.start, self.end, start, end, 0)

        return x_coords, y_coords


### 3D lines
class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""

    is_2Dline = False
    is_3Dline = True
    _dim = 3

    def __init__(self):
        super().__init__()


class Parametric3DLineSeries(Line3DBaseSeries):
    """Representation for a 3D line consisting of three parametric sympy
    expressions and a range."""

    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs):
        super().__init__()
        self.expr_x = sympify(expr_x)
        self.expr_y = sympify(expr_y)
        self.expr_z = sympify(expr_z)
        self.label = label
        self.var = sympify(var_start_end[0])
        self.start = float(var_start_end[1])
        self.end = float(var_start_end[2])
        self.n = kwargs.get('n', 300)
        self.line_color = kwargs.get('line_color', None)

    def __str__(self):
        return '3D parametric cartesian line: (%s, %s, %s) for %s over %s' % (
            str(self.expr_x), str(self.expr_y), str(self.expr_z),
            str(self.var), str((self.start, self.end)))

    def get_parameter_points(self):
        np = import_module('numpy')
        return np.linspace(self.start, self.end, num=self.n)

    def get_points(self):
        np = import_module('numpy')
        param = self.get_parameter_points()
        fx = vectorized_lambdify([self.var], self.expr_x)
        fy = vectorized_lambdify([self.var], self.expr_y)
        fz = vectorized_lambdify([self.var], self.expr_z)

        list_x = fx(param)
        list_y = fy(param)
        list_z = fz(param)

        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        list_z = np.array(list_z, dtype=np.float64)

        list_x = np.ma.masked_invalid(list_x)
        list_y = np.ma.masked_invalid(list_y)
        list_z = np.ma.masked_invalid(list_z)

        self._xlim = (np.amin(list_x), np.amax(list_x))
        self._ylim = (np.amin(list_y), np.amax(list_y))
        self._zlim = (np.amin(list_z), np.amax(list_z))
        return list_x, list_y, list_z


### Surfaces
class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""

    is_3Dsurface = True

    def __init__(self):
        super().__init__()
        self.surface_color = None
    
    def get_data(self):
        return self.get_meshes()

    def get_color_array(self):
        np = import_module('numpy')
        c = self.surface_color
        if isinstance(c, Callable):
            f = np.vectorize(c)
            nargs = arity(c)
            if self.is_parametric:
                variables = list(map(centers_of_faces, self.get_parameter_meshes()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables)
            variables = list(map(centers_of_faces, self.get_meshes()))
            if nargs == 1:
                return f(variables[0])
            elif nargs == 2:
                return f(*variables[:2])
            else:
                return f(*variables)
        else:
            if isinstance(self, SurfaceOver2DRangeSeries):
                return c*np.ones(min(self.n1, self.n2))
            else:
                return c*np.ones(min(self.n1, self.n2))


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a sympy expression and 2D
    range."""
    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__()
        self.expr = sympify(expr)
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self.label = label
        self.n1 = kwargs.get('n1', 50)
        self.n2 = kwargs.get('n2', 50)
        self.surface_color = kwargs.get('surface_color', None)
        self._xlim = (self.start_x, self.end_x)
        self._ylim = (self.start_y, self.end_y)

    def __str__(self):
        return ('cartesian surface: %s for'
                ' %s over %s and %s over %s') % (
                    str(self.expr),
                    str(self.var_x),
                    str((self.start_x, self.end_x)),
                    str(self.var_y),
                    str((self.start_y, self.end_y)))

    def get_meshes(self):
        np = import_module('numpy')
        mesh_x, mesh_y = np.meshgrid(np.linspace(self.start_x, self.end_x,
                                                 num=self.n1),
                                     np.linspace(self.start_y, self.end_y,
                                                 num=self.n2))
        f = vectorized_lambdify((self.var_x, self.var_y), self.expr)
        mesh_z = f(mesh_x, mesh_y)
        mesh_z = np.array(mesh_z, dtype=np.float64)
        mesh_z = np.ma.masked_invalid(mesh_z)
        self._zlim = (np.amin(mesh_z), np.amax(mesh_z))
        return mesh_x, mesh_y, mesh_z


class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric sympy
    expressions and a range."""

    is_parametric = True

    def __init__(
        self, expr_x, expr_y, expr_z, var_start_end_u, var_start_end_v,
        label="", **kwargs):
        super().__init__()
        self.expr_x = sympify(expr_x)
        self.expr_y = sympify(expr_y)
        self.expr_z = sympify(expr_z)
        self.var_u = sympify(var_start_end_u[0])
        self.start_u = float(var_start_end_u[1])
        self.end_u = float(var_start_end_u[2])
        self.var_v = sympify(var_start_end_v[0])
        self.start_v = float(var_start_end_v[1])
        self.end_v = float(var_start_end_v[2])
        self.label = label
        self.n1 = kwargs.get('n1', 50)
        self.n2 = kwargs.get('n2', 50)
        self.surface_color = kwargs.get('surface_color', None)

    def __str__(self):
        return ('parametric cartesian surface: (%s, %s, %s) for'
                ' %s over %s and %s over %s') % (
                    str(self.expr_x),
                    str(self.expr_y),
                    str(self.expr_z),
                    str(self.var_u),
                    str((self.start_u, self.end_u)),
                    str(self.var_v),
                    str((self.start_v, self.end_v)))

    def get_parameter_meshes(self):
        np = import_module('numpy')
        return np.meshgrid(np.linspace(self.start_u, self.end_u,
                                       num=self.n1),
                           np.linspace(self.start_v, self.end_v,
                                       num=self.n2))

    def get_meshes(self):
        np = import_module('numpy')

        mesh_u, mesh_v = self.get_parameter_meshes()
        fx = vectorized_lambdify((self.var_u, self.var_v), self.expr_x)
        fy = vectorized_lambdify((self.var_u, self.var_v), self.expr_y)
        fz = vectorized_lambdify((self.var_u, self.var_v), self.expr_z)

        mesh_x = fx(mesh_u, mesh_v)
        mesh_y = fy(mesh_u, mesh_v)
        mesh_z = fz(mesh_u, mesh_v)

        mesh_x = np.array(mesh_x, dtype=np.float64)
        mesh_y = np.array(mesh_y, dtype=np.float64)
        mesh_z = np.array(mesh_z, dtype=np.float64)

        mesh_x = np.ma.masked_invalid(mesh_x)
        mesh_y = np.ma.masked_invalid(mesh_y)
        mesh_z = np.ma.masked_invalid(mesh_z)

        self._xlim = (np.amin(mesh_x), np.amax(mesh_x))
        self._ylim = (np.amin(mesh_y), np.amax(mesh_y))
        self._zlim = (np.amin(mesh_z), np.amax(mesh_z))

        return mesh_x, mesh_y, mesh_z


### Contours
class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False
    is_contour = True

    def __str__(self):
        return ('contour: %s for '
                '%s over %s and %s over %s') % (
                    str(self.expr),
                    str(self.var_x),
                    str((self.start_x, self.end_x)),
                    str(self.var_y),
                    str((self.start_y, self.end_y)))

class ImplicitSeries(BaseSeries):
    """ Representation for Implicit plot """
    is_implicit = True

    def __init__(self, expr, var_start_end_x, var_start_end_y,
            has_equality, use_interval_math, depth, n,
            line_color):
        super().__init__()
        self.expr = sympify(expr)
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self.get_points = self.get_raster
        self.has_equality = has_equality  # If the expression has equality, i.e.
                                         #Eq, Greaterthan, LessThan.
        self.n = n
        self.use_interval_math = use_interval_math
        self.depth = 4 + depth
        self.line_color = line_color

    def __str__(self):
        return ('Implicit equation: %s for '
                '%s over %s and %s over %s') % (
                    str(self.expr),
                    str(self.var_x),
                    str((self.start_x, self.end_x)),
                    str(self.var_y),
                    str((self.start_y, self.end_y)))

    def get_data(self):
        return self.get_raster()
        
    def get_raster(self):
        func = experimental_lambdify((self.var_x, self.var_y), self.expr,
                                    use_interval=True)
        xinterval = interval(self.start_x, self.end_x)
        yinterval = interval(self.start_y, self.end_y)
        try:
            func(xinterval, yinterval)
        except AttributeError:
            # XXX: AttributeError("'list' object has no attribute 'is_real'")
            # That needs fixing somehow - we shouldn't be catching
            # AttributeError here.
            if self.use_interval_math:
                warnings.warn("Adaptive meshing could not be applied to the"
                            " expression. Using uniform meshing.")
            self.use_interval_math = False

        if self.use_interval_math:
            return self._get_raster_interval(func)
        else:
            return self._get_meshes_grid()

    def _get_raster_interval(self, func):
        """ Uses interval math to adaptively mesh and obtain the plot"""
        k = self.depth
        interval_list = []
        #Create initial 32 divisions
        np = import_module('numpy')
        xsample = np.linspace(self.start_x, self.end_x, 33)
        ysample = np.linspace(self.start_y, self.end_y, 33)

        #Add a small jitter so that there are no false positives for equality.
        # Ex: y==x becomes True for x interval(1, 2) and y interval(1, 2)
        #which will draw a rectangle.
        jitterx = (np.random.rand(
            len(xsample)) * 2 - 1) * (self.end_x - self.start_x) / 2**20
        jittery = (np.random.rand(
            len(ysample)) * 2 - 1) * (self.end_y - self.start_y) / 2**20
        xsample += jitterx
        ysample += jittery

        xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1],
                           xsample[1:])]
        yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1],
                           ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        #recursive call refinepixels which subdivides the intervals which are
        #neither True nor False according to the expression.
        def refine_pixels(interval_list):
            """ Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:

                #Convert the array indices to x and y values
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                #The expression is valid in the interval. Change the contour
                #array values to 1.
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    #Subdivide
                    avgx = intervalx.mid
                    avgy = intervaly.mid
                    a = interval(intervalx.start, avgx)
                    b = interval(avgx, intervalx.end)
                    c = interval(intervaly.start, avgy)
                    d = interval(avgy, intervaly.end)
                    temp_interval_list.append([a, c])
                    temp_interval_list.append([a, d])
                    temp_interval_list.append([b, c])
                    temp_interval_list.append([b, d])
            return temp_interval_list, plot_list

        while k >= 0 and len(interval_list):
            interval_list, plot_list_temp = refine_pixels(interval_list)
            plot_list.extend(plot_list_temp)
            k = k - 1
        #Check whether the expression represents an equality
        #If it represents an equality, then none of the intervals
        #would have satisfied the expression due to floating point
        #differences. Add all the undecided values to the plot.
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return plot_list, 'fill'

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        equal = False
        if isinstance(self.expr, Equality):
            expr = self.expr.lhs - self.expr.rhs
            equal = True

        elif isinstance(self.expr, (GreaterThan, StrictGreaterThan)):
            expr = self.expr.lhs - self.expr.rhs

        elif isinstance(self.expr, (LessThan, StrictLessThan)):
            expr = self.expr.rhs - self.expr.lhs
        else:
            raise NotImplementedError("The expression is not supported for "
                                    "plotting in uniform meshed plot.")
        np = import_module('numpy')
        xarray = np.linspace(self.start_x, self.end_x, self.n)
        yarray = np.linspace(self.start_y, self.end_y, self.n)
        x_grid, y_grid = np.meshgrid(xarray, yarray)

        func = vectorized_lambdify((self.var_x, self.var_y), expr)
        z_grid = func(x_grid, y_grid)
        z_grid[np.ma.where(z_grid < 0)] = -1
        z_grid[np.ma.where(z_grid > 0)] = 1
        if equal:
            return xarray, yarray, z_grid, 'contour'
        else:
            return xarray, yarray, z_grid, 'contourf'


##############################################################################
# Finding the centers of line segments or mesh faces
##############################################################################

def centers_of_segments(array):
    np = import_module('numpy')
    return np.mean(np.vstack((array[:-1], array[1:])), 0)


def centers_of_faces(array):
    np = import_module('numpy')
    return np.mean(np.dstack((array[:-1, :-1],
                             array[1:, :-1],
                             array[:-1, 1:],
                             array[:-1, :-1],
                             )), 2)


def flat(x, y, z, eps=1e-3):
    """Checks whether three points are almost collinear"""
    np = import_module('numpy')
    # Workaround plotting piecewise (#8577):
    #   workaround for `lambdify` in `.experimental_lambdify` fails
    #   to return numerical values in some cases. Lower-level fix
    #   in `lambdify` is possible.
    vector_a = (x - y).astype(np.float64)
    vector_b = (z - y).astype(np.float64)
    dot_product = np.dot(vector_a, vector_b)
    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)
    cos_theta = dot_product / (vector_a_norm * vector_b_norm)
    return abs(cos_theta + 1) < eps


class InteractiveSeries(BaseSeries):
    is_interactive = True

    def __init__(self, *args, **kwargs):
        # free symbols of the parameters
        fs_params = kwargs.get("fs_params", set())
        # number of discretization points
        n = kwargs.get("n", 300)

        exprs, ranges, label = _unpack_args(*args)
        self.label = label
        nexpr, npar = len(exprs), len(ranges)

        if nexpr == 0:
            raise ValueError("At least one expression must be provided." +
                "\nReceived: {}".format(args))
        if npar > 2:
            raise ValueError(
                    "Depending on the backend, only 2D and 3D plots are " +
                    "supported (1 or 2 ranges at most). The provided " +
                    "expressions uses {} ranges.".format(npar))

        # set series attributes
        if (nexpr == 1) and (npar == 1):
            self.is_2Dline = True
        elif (nexpr == 2) and (npar == 1):
            self.is_2Dline = True
            self.is_parametric = True
            # necessary to draw a gradient line with some backends
            self.var = ranges[0][0]
            self.start = float(ranges[0][1])
            self.end = float(ranges[0][2])
        elif (nexpr == 3) and (npar == 1):
            self.is_3Dline = True
            self.is_parametric = True
            # necessary to draw a gradient line with some backends
            self.var = ranges[0][0]
            self.start = float(ranges[0][1])
            self.end = float(ranges[0][2])
        elif (nexpr == 1) and (npar == 2):
            self.is_3Dsurface = True
        elif (nexpr == 3) and (npar == 2):
            self.is_3Dsurface = True
            self.is_parametric = True

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = set().union(*[e.free_symbols for e in exprs])
        fs = fs.difference(fs_params).difference([r[0] for r in ranges])
        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n" +
                "Expression: {}\n".format(args) +
                "Specify what these symbols represent: {}\n".format(fs) +
                "Are they ranges or parameters?")
        
        # if we are dealing with parametric expressions, we pack them into a
        # Tuple so that it can be lambdified
        expr = exprs[0] if len(exprs) == 1 else Tuple(*exprs, sympify=False)
        # generate the lambda function
        signature, f = get_lambda(expr)
        self.signature = signature
        self.function = f

        # Discretize the ranges. In the following dictionary self.ranges:
        #    key: symbol associate to this particular range
        #    val: the numpy array representing the discretization
        discr_symbols = []
        discretizations = []
        for r in ranges:
            discr_symbols.append(r[0])
            discretizations.append(np.linspace(float(r[1]), float(r[2]), n))
        
        if len(ranges) == 1:
            # 2D or 3D lines
            self.ranges = {k: v for k, v in zip(discr_symbols, discretizations)}
        else:
            # surfaces, needs mesh grids
            meshes = np.meshgrid(*discretizations)
            self.ranges = {k: v for k, v in zip(discr_symbols, meshes)}
        
        # this will be used in update_data
        self._discret_shape = discretizations[0].shape
        
        self.data = []

    def update_data(self, params):
        """ Update the data based on the values of the parameters.

        Parameters
        ==========

            params : dict
                key: symbol associated to the parameter
                val: the value

        """
        args = []
        for s in self.signature:
            if s in params.keys():
                args.append(params[s])
            else:
                args.append(self.ranges[s])
        
        results = self.function(*args)

        if ((self.is_3Dsurface and (not self.is_parametric)) or
            (self.is_2Dline and (not self.is_parametric))):
            # in the case of single-expression 2D lines of 3D surfaces
            results = [*self.ranges.values(), results]
    
        if (self.is_parametric and (self.is_3Dsurface or self.is_3Dline or 
                self.is_2Dline)):
            # in this case, some components of the result might be a scalar.
            # need to convert them to the appropriate shape
            results = list(results)
            for i, r in enumerate(results):
                if not is_sequence(r):
                    results[i] = r * np.ones(self._discret_shape)
        
        self.data = results
        
    def get_data(self):
        return self.data


def _set_discretization_points(kwargs, pt):
    """ This function allows the user two use the keyword arguments n, n1 and n2
    to specify the number of discretization points in two directions.

    Parameters
    ==========

        kwargs : dict

        pt : type
            The type of the series, which indicates the kind of plot we are
            trying to create: plot, plot_parametric, ...
    """
    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries, 
                Parametric3DLineSeries, ImplicitSeries]:
        if "n1" in kwargs.keys() and ("n" not in kwargs.keys()):
            kwargs["n"] = kwargs["n1"]
    elif pt in [SurfaceOver2DRangeSeries, ContourSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            kwargs["n2"] = kwargs["n"]
    elif pt in [ParametricSurfaceSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            kwargs["n2"] = kwargs["n"]
    return kwargs
    