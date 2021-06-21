from collections.abc import Callable
from sympy import sympify, Tuple, symbols, solve, re, im
from sympy.geometry import Plane
from sympy.core.relational import (Equality, GreaterThan, LessThan,
                Relational, StrictLessThan, StrictGreaterThan)
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.experimental_lambdify import (
    vectorized_lambdify, lambdify, experimental_lambdify)
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.function import arity
from sympy.core.compatibility import is_sequence
from sympy.plotting.intervalmath import interval
from spb.utils import get_lambda
import warnings
import numpy as np

"""
TODO:
1. InteractiveSeries: allow for setting the number of discretization points
    individually on each direction.
"""

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

    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    # Represents a 2D or 3D vector

    is_complex = False
    # Represent a complex expression

    is_point = False
    # If True, the rendering will use points, not lines.

    def __init__(self):
        super().__init__()

    @property
    def is_3D(self):
        flags3D = [
            self.is_3Dline,
            self.is_3Dsurface,
            self.is_3Dvector
        ]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [
            self.is_2Dline,
            self.is_3Dline
        ]
        return any(flagslines)
    
    @staticmethod
    def _discretize(start, end, N, base=10, scale="linear"):
        if scale == "linear":
            return np.linspace(start, end, N)
        return np.logspace(start, end, N, base=base)
    
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

    def _correct_size(self, l, p):
        if l.size != p.size:
            return l * np.ones(p.size)
        return l

    def get_data(self):
        """ Return lists of coordinates for plotting the line.

        Returns
        =======
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates

            z: list (optional)
                List of z-coordinates in case of Parametric3DLineSeries
            
            param : list (optional)
                List containing the parameter, in case of Parametric2DLineSeries
                and Parametric3DLineSeries.
        """
        points = self.get_points()
        if self.steps is True:
            if self.is_2Dline:
                x, y = points[0], points[1]
                x = np.array((x, x)).T.flatten()[1:]
                y = np.array((y, y)).T.flatten()[:-1]
                if self.is_parametric:
                    points = (x, y, points[2])
                else:
                    points = (x, y)
            elif self.is_3Dline:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                points = (x, y, z, points[3])
        return points

    def get_segments(self):
        SymPyDeprecationWarning(
                feature="get_segments",
                issue=21329,
                deprecated_since_version="1.9",
                useinstead="MatplotlibBackend.get_segments").warn()

        points = type(self).get_data(self)
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def get_color_array(self):
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

        return (np.array(x_coords), np.array(y_coords))

    def _uniform_sampling(self):
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
        return np.linspace(self.start, self.end, num=self.n)

    def _uniform_sampling(self):
        param = self.get_parameter_points()
        fx = vectorized_lambdify([self.var], self.expr_x)
        fy = vectorized_lambdify([self.var], self.expr_y)
        list_x = fx(param)
        list_y = fy(param)
        # expr_x or expr_y may be scalars. This allows scalar components
        # to be plotted as well
        list_x = self._correct_size(list_x, param)
        list_y = self._correct_size(list_y, param)
        return list_x, list_y, param

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
        param = []

        def sample(param_p, param_q, p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            random = 0.45 + np.random.rand() * 0.1
            param_new = param_p + random * (param_q - param_p)
            xnew = f_x(param_new)
            ynew = f_y(param_new)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

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
                param.append(param_p)

        f_start_x = f_x(self.start)
        f_start_y = f_y(self.start)
        start = [f_start_x, f_start_y]
        f_end_x = f_x(self.end)
        f_end_y = f_y(self.end)
        end = [f_end_x, f_end_y]
        x_coords.append(f_start_x)
        y_coords.append(f_start_y)
        param.append(self.start)
        sample(self.start, self.end, start, end, 0)

        return np.array(x_coords), np.array(y_coords), np.array(param)


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
    
    is_parametric = True

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
        return np.linspace(self.start, self.end, num=self.n)

    def get_points(self):
        param = self.get_parameter_points()
        fx = vectorized_lambdify([self.var], self.expr_x)
        fy = vectorized_lambdify([self.var], self.expr_y)
        fz = vectorized_lambdify([self.var], self.expr_z)

        list_x = fx(param)
        list_y = fy(param)
        list_z = fz(param)

        # expr_x, expr_y or expr_z may be scalars. This allows scalar components
        # to be plotted as well
        list_x = self._correct_size(list_x, param)
        list_y = self._correct_size(list_y, param)
        list_z = self._correct_size(list_z, param)

        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        list_z = np.array(list_z, dtype=np.float64)

        list_x = np.ma.masked_invalid(list_x)
        list_y = np.ma.masked_invalid(list_y)
        list_z = np.ma.masked_invalid(list_z)

        return list_x, list_y, list_z, param


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

    def __str__(self):
        return ('cartesian surface: %s for'
                ' %s over %s and %s over %s') % (
                    str(self.expr),
                    str(self.var_x),
                    str((self.start_x, self.end_x)),
                    str(self.var_y),
                    str((self.start_y, self.end_y)))

    def _correct(self, a, b):
        """ If the provided expression is a scalar, we need to
        convert its dimension to the appropriate grid size.
        """
        if a.shape != b.shape:
            return b * np.ones_like(a)
        return b

    def get_meshes(self):
        mesh_x, mesh_y = np.meshgrid(np.linspace(self.start_x, self.end_x,
                                                 num=self.n1),
                                     np.linspace(self.start_y, self.end_y,
                                                 num=self.n2))
        f = vectorized_lambdify((self.var_x, self.var_y), self.expr)
        mesh_z = f(mesh_x, mesh_y)
        mesh_z = self._correct(mesh_x, mesh_z).astype(np.float64)
        mesh_z = np.ma.masked_invalid(mesh_z)
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
        return np.meshgrid(np.linspace(self.start_u, self.end_u,
                                       num=self.n1),
                           np.linspace(self.start_v, self.end_v,
                                       num=self.n2))

    def get_meshes(self):

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
    """ Representation for Implicit plot

    References
    ==========

    .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
    Mathematical Formulae with Two Free Variables.

    .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
    Arithmetic. Master's thesis. University of Toronto, 1996
    
    """
    is_implicit = True

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="",
            **kwargs):
        super().__init__()
        expr, has_equality = self._has_equality(sympify(expr))
        self.expr = expr
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self.get_points = self.get_raster
        self.has_equality = has_equality
        self.n = kwargs.get("n", 1000)
        self.label = label
        self.adaptive = kwargs.get("adaptive", False)

        if isinstance(expr, BooleanFunction):
            self.adaptive = True
            warnings.warn(
                "The provided expression contains Boolean functions. " +
                "In order to plot the expression, the algorithm " +
                "automatically switched to an adaptive sampling."
            )

        # Check whether the depth is greater than 4 or less than 0.
        depth = kwargs.get("depth", 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        self.depth = 4 + depth
    
    def _has_equality(self, expr):
        # Represents whether the expression contains an Equality, GreaterThan 
        # or LessThan
        has_equality = False

        def arg_expand(bool_expr):
            """
            Recursively expands the arguments of an Boolean Function
            """
            for arg in bool_expr.args:
                if isinstance(arg, BooleanFunction):
                    arg_expand(arg)
                elif isinstance(arg, Relational):
                    arg_list.append(arg)

        arg_list = []
        if isinstance(expr, BooleanFunction):
            arg_expand(expr)
            # Check whether there is an equality in the expression provided.
            if any(isinstance(e, (Equality, GreaterThan, LessThan))
                    for e in arg_list):
                has_equality = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_equality = True

        return expr, has_equality

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
            if self.adaptive:
                warnings.warn("Adaptive meshing could not be applied to the"
                            " expression. Using uniform meshing.")
            self.adaptive = False

        if self.adaptive:
            return self._get_raster_interval(func)
        else:
            return self._get_meshes_grid()

    def _get_raster_interval(self, func):
        """ Uses interval math to adaptively mesh and obtain the plot"""
        k = self.depth
        interval_list = []
        #Create initial 32 divisions
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
    return np.mean(np.vstack((array[:-1], array[1:])), 0)


def centers_of_faces(array):
    return np.mean(np.dstack((array[:-1, :-1],
                             array[1:, :-1],
                             array[:-1, 1:],
                             array[:-1, :-1],
                             )), 2)


def flat(x, y, z, eps=1e-3):
    """Checks whether three points are almost collinear"""
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
    """ Represent an interactive series, in which the expressions can be either
    a line or a surface (parametric or not). On top of the usual ranges (x, y or
    u, v, which must be provided), the expressions can use any number of 
    parameters.

    This class internally convert the expressions to a lambda function, which is
    evaluated by calling update_data(params), passing in all the necessary
    parameters. Once update_data(params) has been executed, then get_data()
    can be used.
    """

    is_interactive = True

    def __init__(self, exprs, ranges, label="", **kwargs):
        # free symbols of the parameters
        params = kwargs.get("params", dict())
        # number of discretization points
        self.n1 = kwargs.get("n1", 250)
        self.n2 = kwargs.get("n2", 250)
        self.n3 = kwargs.get("n3", 250)
        n = [self.n1, self.n2, self.n3]

        self.xscale = kwargs.get('xscale', 'linear')
        self.yscale = kwargs.get('yscale', 'linear')
        self.label = label
        nexpr, npar = len(exprs), len(ranges)

        if nexpr == 0:
            raise ValueError("At least one expression must be provided." +
                "\nReceived: {}".format((exprs, ranges, label)))
        # # TODO: do I really need this?
        # if npar > 2:
        #     raise ValueError(
        #             "Depending on the backend, only 2D and 3D plots are " +
        #             "supported (1 or 2 ranges at most). The provided " +
        #             "expressions uses {} ranges.".format(npar))

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
            if kwargs.get("threed", False):
                self.is_3Dsurface = True
            else:
                self.is_contour = True
        elif (nexpr == 3) and (npar == 2):
            self.is_3Dsurface = True
            self.is_parametric = True
        elif (nexpr == 2) and (npar == 2):
            self.is_vector = True
            self.is_slice = False
            self.is_2Dvector = True
        elif (nexpr == 3) and (npar == 3):
            self.is_vector = True
            self.is_3Dvector = True
            self.is_slice = False

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = set().union(*[e.free_symbols for e in exprs])
        fs = fs.difference(params.keys()).difference([r[0] for r in ranges])
        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n" +
                "Expression: {}\n".format((exprs, ranges, label)) +
                "Specify what these symbols represent: {}\n".format(fs) +
                "Are they ranges or parameters?")
        
        # if we are dealing with parametric expressions, we pack them into a
        # Tuple so that it can be lambdified
        self.expr = exprs[0] if len(exprs) == 1 else Tuple(*exprs, sympify=False)
        # generate the lambda function
        signature, f = get_lambda(self.expr)
        self.signature = signature
        self.function = f

        # Discretize the ranges. In the following dictionary self.ranges:
        #    key: symbol associate to this particular range
        #    val: the numpy array representing the discretization
        discr_symbols = []
        discretizations = []
        for i, r in enumerate(ranges):
            discr_symbols.append(r[0])
            scale = self.xscale
            if i == 1: # y direction
                scale = self.yscale
            discretizations.append(
                self._discretize(float(r[1]), float(r[2]), n[i], scale=scale))

        if len(ranges) == 1:
            # 2D or 3D lines
            self.ranges = {k: v for k, v in zip(discr_symbols, discretizations)}
        else:
            _slice = kwargs.get("slice", None)
            if _slice is not None:
                # sliced 3D vector fields: the discretizations are provided by
                # the plane or the surface
                self.is_slice = True
                kwargs2 = kwargs.copy()
                kwargs2 = _set_discretization_points(kwargs2, SliceVector3DSeries)
                slice_surf = GeometricPlaneSeries(
                    _slice, *ranges, "", **kwargs2)
                self.ranges = {k: v for k, v in
                    zip(discr_symbols, slice_surf.get_data())}
            else:
                # surfaces: needs mesh grids
                meshes = np.meshgrid(*discretizations)
                self.ranges = {k: v for k, v in zip(discr_symbols, meshes)}
        
        self.data = None
        if len(params) > 0:
            self.update_data(params)
    
    def _evaluate(self, params):
        """ Update the data based on the values of the parameters.

        Parameters
        ==========

            params : dict
                key: symbol associated to the parameter
                val: the value

        """
        args = []
        for s in self.signature:
            print("CIS.update_data -> symbol", s)
            if s in params.keys():
                args.append(params[s])
            else:
                args.append(self.ranges[s])
    
        return self.function(*args)

    def update_data(self, params):
        """ Update the data based on the values of the parameters.

        Parameters
        ==========

            params : dict
                key: symbol associated to the parameter
                val: the value

        """
        results = self._evaluate(params)

        if (self.is_contour or
            (self.is_3Dsurface and (not self.is_parametric)) or
            (self.is_2Dline and (not self.is_parametric))):
            # in the case of single-expression 2D lines of 3D surfaces
            results = [*self.ranges.values(), results]
            self.data = results
            print("WTF1")
    
        elif (self.is_parametric and (self.is_3Dline or self.is_2Dline)):
            # also add the parameter
            results = [*results, *self.ranges.values()]
            print("WTF2")
            self.data = results
        
        elif self.is_vector:
            # in order to plot a vector, we also need the discretized region
            self.data = [*self.ranges.values(), *results]
            print("WTF4")
        else:
            self.data = results
            print("WTF5")
        
    def get_data(self):
        # if the expression depends only on the ranges, the user can call get_data
        # directly without calling update_data
        if (self.data is None) and (len(self.signature) == len(self.ranges)):
            self.update_data(dict())
        if self.data is None:
            raise ValueError(
                "To generate the numerical data, call update_data(params), " +
                "providing the necessary parameters.")
        return self.data
    
    def __str__(self):
        ranges = [(k, v[0], v[-1]) for k, v in self.ranges.items()]
        return ('interactive expression: %s with ranges'
                ' %s and parameters %s') % (
                    str(self.expr),
                    ", ".join([str(r) for r in ranges]),
                    str(self.signature))


class ComplexSeries(BaseSeries):
    """ Represent a complex number or a complex function.
    """
    is_complex = True
    is_point = False
    is_domain_coloring = False

    def __init__(self, expr, r, label, **kwargs):
        expr = sympify(expr)
        nolist = False
        if isinstance(expr, (list, tuple, Tuple)):
            self.expr = expr
            self.is_2Dline = True
            self.is_point = True
            self.var = None
            self.start = None
            self.end = None
        else:
            # we are not plotting list of complex points, but real/imag or
            # magnitude/argument plots
            nolist = True
        
        self._init_attributes(expr, r, label, nolist, **kwargs)
    
    def _init_attributes(self, expr, r, label, nolist, **kwargs):
        """ This method reduces code repetition between ComplexSeries and
        ComplexInteractiveSeries.
        """
        if nolist:
            self.var = sympify(r[0])
            self.start = complex(r[1])
            self.end = complex(r[2])
            if self.start.imag == self.end.imag:
                self.is_2Dline = True
                if kwargs.get('absarg', False):
                    self.is_parametric = True
            elif kwargs.get("threed", False):
                self.is_3Dsurface = True
            else:
                self.is_domain_coloring = True
        
        self.expr = sympify(expr)
        self.n1 = kwargs.get('n1', 300)
        self.n2 = kwargs.get('n2', 300)
        self.xscale = kwargs.get('xscale', 'linear')
        self.yscale = kwargs.get('yscale', 'linear')

        # these will be passed to cplot.get_srgb1
        self.alpha = kwargs.get('alpha', 1)
        self.colorspace = kwargs.get('colorspace', 'cam16')

        self.real = kwargs.get('real', True)
        self.imag = kwargs.get('imag', True)

        if self.is_parametric:
            self.label = "Abs(%s)" % label
        elif self.real and self.imag:
            self.label = label
        elif self.real:
            self.label = "re(%s)" % label
        elif self.imag:
            self.label = "im(%s)" % label
        else:
            self.label = label

    def _correct_output(self, x, r):
        """ Obtain the correct output depending the initialized settings.

        This method reduces code repetition between ComplexSeries and
        ComplexInteractiveSeries.

        Parameters
        ==========
            x : np.ndarray
                Discretized domain. Can be a complex line or a complex region.

            r : np.ndarray
                Numerical evaluation result.
        """
        print("ComplexSeries._correct_output", x.shape, r.shape)

        if self.start.imag == self.end.imag:
            if self.is_parametric:
                print("ComplexSeries._correct_output -> parametric")
                return np.real(x), np.absolute(r), np.angle(r)
            elif self.real and self.imag:
                print("ComplexSeries._correct_output -> real and imag")
                return np.real(x), np.real(r), np.imag(r)
            elif self.real:
                print("ComplexSeries._correct_output -> real")
                return np.real(x), np.real(r)
            elif self.imag:
                print("ComplexSeries._correct_output -> imag")
                return np.real(x), np.imag(r)
            print("ComplexSeries._correct_output -> something else")
            return x, r
        
        print("ComplexSeries._correct_output -> domain coloring or 3D")
        # Domain coloring / 3D
        return np.real(x), np.imag(x), r, np.absolute(r), np.angle(r)

    
    def get_data(self):
        if isinstance(self.expr, (list, tuple, Tuple)):
            # list of complex points
            x_list, y_list = [], []
            for p in self.expr:
                x_list.append(float(re(p)))
                y_list.append(float(im(p)))
            return x_list, y_list
        
        # TODO: do I need this???
        from sympy import lambdify

        if np.imag(self.start) == np.imag(self.end):
            # compute the real/imaginary part/magnitude/argument of the complex 
            # function
            x = self._discretize(self.start, self.end, self.n1, self.xscale)
            f = lambdify([self.var], self.expr)
            y = f(x + np.imag(self.start) * 1j)
            return self._correct_output(x, y)
        
        # Domain coloring
        start_x = np.real(self.start)
        end_x = np.real(self.end)
        start_y = np.imag(self.start)
        end_y = np.imag(self.end)
        x = self._discretize(start_x, end_x, self.n1, self.xscale)
        y = self._discretize(start_y, end_y, self.n2, self.yscale)
        xx, yy = np.meshgrid(x, y)
        f = lambdify([self.var], self.expr)
        domain = xx + 1j * yy
        zz = f(domain)
        return self._correct_output(domain, zz)


class ComplexInteractiveSeries(InteractiveSeries, ComplexSeries):

    def __init__(self, expr, r, label="", **kwargs):
        params = kwargs.get("params", dict())
        
        self._init_attributes(expr, r, label, True, **kwargs)
        self.xscale = kwargs.get('xscale', 'linear')
        self.yscale = kwargs.get('yscale', 'linear')
        
        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = expr.free_symbols
        fs = fs.difference(params.keys()).difference(set([r[0]]))
        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n" +
                "Expression: {}\n".format((expr, r, label)) +
                "Specify what these symbols represent: {}\n".format(fs) +
                "Are they ranges or parameters?")
        
        # generate the lambda function
        signature, f = get_lambda(self.expr)
        self.signature = signature
        self.function = f

        # Discretize the ranges. In the following dictionary self.ranges:
        #    key: symbol associate to this particular range
        #    val: the numpy array representing the discretization
        if complex(r[1]).imag != complex(r[2]).imag:
            # domain coloring
            x = self._discretize(complex(r[1]).real, complex(r[2]).real,
                    self.n1, scale=self.xscale)
            y = self._discretize(complex(r[1]).imag, complex(r[2]).imag,
                    self.n2, scale=self.yscale)
            xx, yy = np.meshgrid(x, y)
            zz = xx + 1j * yy
            self.ranges = {self.var: zz}
        else:
            # line plot
            x = self._discretize(complex(r[1]).real, complex(r[2]).real,
                    self.n1, scale=self.xscale)
            self.ranges = {self.var: x + 0j}
        
        self.data = None
        if len(params) > 0:
            self.update_data(params)
        
    def update_data(self, params):
        results = self._evaluate(params)
        print("CIS.update_data len(results)", len(results))
        self.data = self._correct_output(self.ranges[self.var], results)


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
    elif pt in [SurfaceOver2DRangeSeries, ContourSeries, ComplexSeries,
            ParametricSurfaceSeries, Vector2DSeries, ComplexInteractiveSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            kwargs["n2"] = kwargs["n"]
    elif pt in [Vector3DSeries, SliceVector3DSeries, InteractiveSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            kwargs["n2"] = kwargs["n"]
            kwargs["n3"] = kwargs["n"]
    return kwargs


class VectorBase(BaseSeries):
    """ Represent a vector field.
    """
    is_vector = True
    is_2D = False
    is_3D = False
    is_slice = False

    def _correct(self, a, b):
        """ If one of the provided vector components is a scalar, we need to
        convert its dimension to the appropriate grid size.
        """
        if a.shape != b.shape:
            return b * np.ones_like(a)
        return b


class Vector2DSeries(VectorBase):
    """ Represents a 2D vector field.
    """
    is_2Dvector = True

    def __init__(self, u, v, range1, range2, label, **kwargs):
        self.u = SurfaceOver2DRangeSeries(u, range1, range2, **kwargs)
        self.v = SurfaceOver2DRangeSeries(v, range1, range2, **kwargs)
        self.label = label

    def get_data(self):
        x, y, u = self.u.get_data()
        _, _, v = self.v.get_data()
        return x, y, self._correct(x, u), self._correct(x, v)


class Vector3DSeries(VectorBase):
    """ Represents a 3D vector field.
    """
    is_3D = True
    is_3Dvector = True

    def __init__(self, u, v, w, range_x, range_y, range_z, label, **kwargs):
        self.u = sympify(u)
        self.v = sympify(v)
        self.w = sympify(w)
        self.var_x = sympify(range_x[0])
        self.start_x = float(range_x[1])
        self.end_x = float(range_x[2])
        self.var_y = sympify(range_y[0])
        self.start_y = float(range_y[1])
        self.end_y = float(range_y[2])
        self.var_z = sympify(range_z[0])
        self.start_z = float(range_z[1])
        self.end_z = float(range_z[2])
        self.label = label
        self.n1 = kwargs.get('n1', 10)
        self.n2 = kwargs.get('n2', 10)
        self.n3 = kwargs.get('n3', 10)

    def _discretize(self):
        """ This method allows to reduce code repetition.
        """
        return np.meshgrid(
            np.linspace(self.start_x, self.end_x, num=self.n1),
            np.linspace(self.start_y, self.end_y, num=self.n2),
            np.linspace(self.start_z, self.end_z, num=self.n3))

    def get_data(self):
        x, y, z = self._discretize()
        fu = vectorized_lambdify((self.var_x, self.var_y, self.var_z), self.u)
        fv = vectorized_lambdify((self.var_x, self.var_y, self.var_z), self.v)
        fw = vectorized_lambdify((self.var_x, self.var_y, self.var_z), self.w)
        uu = fu(x, y, z)
        vv = fv(x, y, z)
        ww = fw(x, y, z)
        uu = self._correct(x, uu)
        vv = self._correct(y, vv)
        ww = self._correct(z, ww)
        def _convert(a):
            a = np.array(a, dtype=np.float64)
            return np.ma.masked_invalid(a)
        return x, y, z, _convert(uu), _convert(vv), _convert(ww)


class SliceVector3DSeries(Vector3DSeries):
    """ Represents a 3D vector field plotted over a slice, which can be a slice
    plane or a slice surface.
    """
    is_slice = True

    def __init__(self, plane, u, v, w, range_x, range_y, range_z, label,
            **kwargs):
        if isinstance(plane, Plane):
            self.plane = GeometricPlaneSeries(sympify(plane), 
                    range_x, range_y, range_z, **kwargs)
        else:
            self.plane = SurfaceOver2DRangeSeries(plane,
                    range_x, range_y, **kwargs)
        super().__init__(u, v, w, range_x, range_y, range_z, label, **kwargs)
    
    def _discretize(self):
        """ This method allows to reduce code repetition.
        """
        return self.plane.get_data()


class GeometricPlaneSeries(SurfaceBaseSeries):
    """ Represents a plane in a 3D domain.
    """
    is_3Dsurface = True

    def __init__(self, plane, x_range, y_range, z_range, label="", **kwargs):
        self.plane = sympify(plane)
        if not isinstance(self.plane, Plane):
            raise TypeError(
                "`plane` must be an instance of sympy.geometry.Plane")
        self.x_range = sympify(x_range)
        self.y_range = sympify(y_range)
        self.z_range = sympify(z_range)
        self.label = label
        self.n1 = kwargs.get("n1", 20)
        self.n2 = kwargs.get("n2", 20)
        self.n3 = kwargs.get("n3", 20)
    
    def get_data(self):
        x, y, z = symbols("x, y, z")
        fs = self.plane.equation(x, y, z).free_symbols
        xx, yy, zz = None, None, None
        if fs == set([x]):
            # parallel to yz plane (normal vector (1, 0, 0))
            s = SurfaceOver2DRangeSeries(self.plane.p1[0], 
                    (x, *self.z_range[1:]), (y, *self.y_range[1:]), "", 
                    n1=self.n3, n2=self.n2)
            xx, yy, zz = s.get_data()
            xx, yy, zz = zz, yy, xx
        elif fs == set([y]):
            # parallel to xz plane (normal vector (0, 1, 0))
            s = SurfaceOver2DRangeSeries(self.plane.p1[1], 
                    (x, *self.x_range[1:]), (y, *self.z_range[1:]), "", 
                    n1=self.n1, n2=self.n3)
            xx, yy, zz = s.get_data()
            xx, yy, zz = xx, zz, yy
        else:
            # parallel to xy plane, or any other plane
            eq = self.plane.equation(x, y, z)
            if z in eq.free_symbols:
                eq = solve(eq, z)[0]
            s = SurfaceOver2DRangeSeries(eq, 
                    (x, *self.x_range[1:]), (y, *self.y_range[1:]), "", 
                    n1=self.n1, n2=self.n2)
            xx, yy, zz = s.get_data()
        return xx, yy, zz
