from sympy import sympify, Tuple, symbols, solve, re, im, Add, Mul, Expr
from sympy.geometry import (
    Plane, Polygon, Circle, Ellipse, Line, Segment, Ray,
    Line3D, Curve, Point2D, Point3D,
)
from sympy.geometry.entity import GeometryEntity
from sympy.geometry.line import LinearEntity2D, LinearEntity3D
from sympy.core.relational import (
    Equality, GreaterThan, LessThan,
    Relational, StrictLessThan, StrictGreaterThan,
)
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.experimental_lambdify import (
    vectorized_lambdify,
    lambdify,
    experimental_lambdify,
)
from sympy.plotting.intervalmath import interval
from spb.utils import get_lambda
import warnings
import numpy as np

old_lambdify = lambdify

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

    is_3Dline = False

    is_3Dsurface = False

    is_contour = False

    is_implicit = False
    # Different from is_contour as the colormap in backend will be
    # different

    is_parametric = False

    is_interactive = False
    # An interactive series can update its data.

    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    # Represents a 2D or 3D vector

    is_complex = False
    # Represent a complex expression
    is_domain_coloring = False

    is_point = False
    # If True, the rendering will use points, not lines.

    is_geometry = False
    # If True, it represents an object of the sympy.geometry module

    def __init__(self):
        super().__init__()

    @property
    def is_3D(self):
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    @staticmethod
    def _discretize(start, end, N, scale="linear"):
        if scale == "linear":
            return np.linspace(start, end, N)
        return np.geomspace(start, end, N)

    @staticmethod
    def _correct_size(a, b):
        """If `a` is a scalar, we need to convert its dimension to the
        appropriate grid size given by `b`.
        """
        if not isinstance(a, np.ndarray):
            # `a` is a scalar (int or float)
            a = np.array(a)

        if a.shape != b.shape:
            return a * np.ones_like(b)
        return a

    def get_data(self):
        """All child series should implement this method to return the
        numerical data which can be used by a plotting library.
        """
        raise NotImplementedError

    def _evaluate_mpmath(self, f, args):
        """ Use the multiprocessing module to run a parallel evaluation of a
        lambda function using mpmath.
        """
        # TODO: this is likely a horrible solution to the following problem:
        # evaluating a function with mpmath is much slower than using Numpy,
        # but (IMHO) it provides better results with complex functions in
        # comparison to Numpy. That's because the two libraries deals with
        # branch cuts differently.
        # In order to improve performance, I'd like to use the
        # multiprocessing module and use the available cores. However,
        # pool.map requires the scope of the function to be in the global
        # namespace. Since I'm inside the scope of instance method of a
        # class, I have not identified a better way to achieve that,
        # so I'm relying on the global keyword hack.

        global _wrapper_complex_func
        def _wrapper_complex_func(args):
            try:
                r = f(*args)
            except ZeroDivisionError as err:
                # TODO: in LineOver1DRangeSerie._uniform_sampling, if I convert
                # the numbers to mpmath.mpc the following evaluation might
                # produce a ZeroDivisionError. Instead, if I use standard
                # complex numbers, it doesn't. Can I realiably use standard
                # complex numbers instead of mpc?
                warnings.warn(
                    "Dealing with {} with Mpmath evaluation ".format(err) +
                    "at the following locations: {}".format(args))
                r = np.nan
            return r


        from multiprocessing import Pool, cpu_count
        pool = Pool(processes = cpu_count())
        return pool.map(_wrapper_complex_func, args)


### 2D lines
class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines."""

    is_2Dline = True

    def __init__(self, **kwargs):
        super().__init__()
        self.label = None
        self.steps = kwargs.get("steps", False)
        self.only_integers = kwargs.get("only_integers", False)
        self.is_point = kwargs.get("is_point", False)

    def get_data(self):
        """Return lists of coordinates for plotting the line.

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
        points = [np.array(p, dtype=np.float64) for p in points]

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


class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    def __init__(self, list_x, list_y, label="", **kwargs):
        super().__init__()
        self.list_x = np.array(list_x, dtype=np.float64)
        self.list_y = np.array(list_y, dtype=np.float64)
        self.label = label
        self.is_point = kwargs.get("is_point", False)

    def __str__(self):
        return "list plot"

    def get_points(self):
        return (self.list_x, self.list_y)


class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = sympify(expr)
        self.label = label
        self.var = sympify(var_start_end[0])

        self.is_complex = kwargs.get("is_complex", False)
        if (isinstance(var_start_end[1], (Add, Mul)) or
                isinstance(var_start_end[2], (Add, Mul)) or self.is_complex):
            self.start = complex(var_start_end[1])
            self.end = complex(var_start_end[2])
        else:
            self.start = float(var_start_end[1])
            self.end = float(var_start_end[2])

        self.n = kwargs.get("n", 1000)
        self.adaptive = kwargs.get("adaptive", True)
        self.depth = kwargs.get("depth", 9)
        self.xscale = kwargs.get("xscale", "linear")
        self.polar = kwargs.get("polar", False)
        self.modules = kwargs.get("modules", None)
        self.absarg = kwargs.get("absarg", None)
        if self.absarg is not None:
            self.is_parametric = True

    def __str__(self):
        return "cartesian line: %s for %s over %s" % (
            str(self.expr),
            str(self.var),
            str((self.start, self.end)),
        )

    def adaptive_sampling(self, f, start, end, max_depth=9, xscale="linear"):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        Parameters
        ==========
            f : callable
                The function to be numerical evaluated

            start, end : floats
                start and end values of the discretized domain

            max_depth : int
                Controls the smootheness of the overall evaluation. The higher
                the number, the smoother the function, the more memory will be
                used by this recursive procedure. Default value is 9.

            xscale : str
                Discretization strategy. Can be "linear" or "log". Default to
                "linear".

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
               Luiz Henrique de Figueiredo.
        """
        from mpmath import mpf
        x_coords = []
        y_coords = []

        def _func(t, extract=True):
            if t is None:
                return t
            if extract:
                return t.real if not isinstance(t, mpf) else float(t)
            return t if not isinstance(t, mpf) else float(t)

        def sample(p, q, depth):
            """Samples recursively if three points are almost collinear.
            For depth < max_depth, points are added irrespective of whether
            they satisfy the collinearity condition or not. The maximum
            depth allowed is max_depth.
            """
            # Randomly sample to avoid aliasing.
            random = 0.45 + np.random.rand() * 0.1
            if xscale == "log":
                xnew = 10 ** (
                    np.log10(p[0]) + random * (np.log10(q[0]) - np.log10(p[0]))
                )
            else:
                xnew = p[0] + random * (q[0] - p[0])

            ynew = f(xnew)
            new_point = np.array([xnew, _func(ynew, False)])

            # Maximum depth
            if depth > max_depth:
                x_coords.append(q[0] if q[0] is None else q[0].real)
                y_coords.append(q[1] if q[1] is None else q[1].real)

            # Sample irrespective of whether the line is flat till the
            # depth of 6. We are not using linspace to avoid aliasing.
            elif depth < max_depth:
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif p[1] is None and q[1] is None:
                if xscale == "log":
                    xarray = np.logspace(p[0], q[0], 10)
                else:
                    xarray = np.linspace(p[0], q[0], 10)

                yarray = list(map(f, xarray))
                if any(y is not None for y in yarray):
                    for i in range(len(yarray) - 1):
                        if yarray[i] is not None or yarray[i + 1] is not None:
                            sample(
                                [xarray[i], yarray[i]],
                                [xarray[i + 1], yarray[i + 1]],
                                depth + 1,
                            )

            # Sample further if one of the end points in None (i.e. a
            # complex value) or the three points are not almost collinear.
            elif (
                p[1] is None
                or q[1] is None
                or new_point[1] is None
                or not flat(p, new_point, q)
            ):
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)
            else:
                x_coords.append(q[0] if q[0] is None else q[0].real)
                y_coords.append(q[1] if q[1] is None else q[1].real)

        f_start = f(start)
        f_start = _func(f_start)
        f_end = f(end)
        f_end = _func(f_end)
        x_coords.append(start)
        y_coords.append(f_start)
        sample(np.array([start, f_start]), np.array([end, f_end]), 0)
        return x_coords, y_coords

    def get_points(self):
        """Return lists of coordinates for plotting. Depending on the
        `adaptive` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        Returns
        =======
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates
        """
        
        # TODO: this wrapper function is a very awful hack in order to pass 
        # test inside test_plot.py. Need a better and more reliable lambdify...
        # Once that's done, we must remove this wrapper function and the 
        # following try/except.
        def _doit(g):
            if self.only_integers or not self.adaptive:
                x, y = self._uniform_sampling(g)
                x, y = np.array(x), np.array(y)
            else:
                x, y = self.adaptive_sampling(
                    g, self.start, self.end, self.depth, self.xscale
                )
            return x, y

        try:
            from sympy import lambdify
            f = lambdify([self.var], self.expr, self.modules)
            x, y = _doit(f)
        except:
            if self.only_integers or not self.adaptive:
                f2 = vectorized_lambdify([self.var], self.expr)
            else:
                f2 = old_lambdify([self.var], self.expr)
            x, y = _doit(f2)

        if self.is_complex and (self.absarg is not None):
            # compute the argument at the x locations. Right now, x contains the
            # real part of the discretization line. To compute the argument, we
            # also need the imaginary part.
            # NOTE: this is clearly a far from optimal approach, as we are going
            # to evaluate the function again. However, it reuses code.

            # TODO: can I store another list of x coordinates containing also
            # the imaginary part in the adaptive_sampling algorithm? In that way
            # I could remove this interpolation step.

            # interpolation line
            m = (self.end.imag - self.start.imag) / (self.end.real - self.start.real)
            re = np.array(x)
            im = m * re - self.start.imag

            from sympy import lambdify
            f2 = lambdify([self.var], self.absarg, self.modules)

            if self.modules == "mpmath":
                from mpmath import arg, mpc
                w = self._evaluate_mpmath(f2, [[mpc(r, i), ] for r, i in zip(re, im)])
                angle = np.array([float(arg(t)) for t in w])
            else:
                w = f2(re + im*1j)
                angle = np.angle(w)
            return np.real(x), y, angle

        if self.polar:
            t = x.copy()
            x = y * np.cos(t)
            y = y * np.sin(t)

        return np.real(x), y

    def _uniform_sampling(self, f):
        start, end, N = self.start, self.end, self.n
        if self.only_integers is True:
            start, end = int(start), int(end)
            N = end - start + 1
        x = self._discretize(start, end, N, scale=self.xscale)

        if self.is_complex and (self.modules == "mpmath"):
            from mpmath import mpc
            y = self._evaluate_mpmath(f,
                    # [[t] for t in x])
                    [[mpc(t.real, t.imag)] for t in x])
            y = [float(t.real) for t in y]
        else:
            y = f(x)

        y = self._correct_size(y, x)
        return x, y


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
        self.n = kwargs.get("n", 300)
        self.adaptive = kwargs.get("adaptive", True)
        self.depth = kwargs.get("depth", 9)
        self.scale = kwargs.get("xscale", "linear")

    def __str__(self):
        return "parametric cartesian line: (%s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.var),
            str((self.start, self.end)),
        )

    def _uniform_sampling(self):
        param = self._discretize(self.start, self.end, self.n, scale=self.scale)
        fx = vectorized_lambdify([self.var], self.expr_x)
        fy = vectorized_lambdify([self.var], self.expr_y)
        list_x = fx(param)
        list_y = fy(param)
        # expr_x or expr_y may be scalars. This allows scalar components
        # to be plotted as well
        list_x = self._correct_size(list_x, param)
        list_y = self._correct_size(list_y, param)
        return list_x, list_y, param

    @staticmethod
    def adaptive_sampling(fx, fy, start, end, max_depth=9):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        Parameters
        ==========
            fx : callable
                The function to be numerical evaluated in the horizontal
                direction.

            fy : callable
                The function to be numerical evaluated in the vertical
                direction.

            start, end : floats
                start and end values of the discretized domain

            max_depth : int
                Controls the smootheness of the overall evaluation. The higher
                the number, the smoother the function, the more memory will be
                used by this recursive procedure. Default value is 9.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
               Luiz Henrique de Figueiredo.
        """
        x_coords = []
        y_coords = []
        param = []

        def sample(param_p, param_q, p, q, depth):
            """Samples recursively if three points are almost collinear.
            For depth < max_depth, points are added irrespective of whether
            they satisfy the collinearity condition or not. The maximum
            depth allowed is max_depth.
            """
            # Randomly sample to avoid aliasing.
            random = 0.45 + np.random.rand() * 0.1
            param_new = param_p + random * (param_q - param_p)
            xnew = fx(param_new)
            ynew = fy(param_new)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > max_depth:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

            # Sample irrespective of whether the line is flat till the
            # depth of 6. We are not using linspace to avoid aliasing.
            elif depth < max_depth:
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif (p[0] is None and q[1] is None) or (p[1] is None and q[1] is None):
                param_array = np.linspace(param_p, param_q, 10)
                x_array = list(map(fx, param_array))
                y_array = list(map(fy, param_array))
                if any(
                    x is not None and y is not None for x, y in zip(x_array, y_array)
                ):
                    for i in range(len(y_array) - 1):
                        if (x_array[i] is not None and y_array[i] is not None) or (
                            x_array[i + 1] is not None and y_array[i + 1] is not None
                        ):
                            point_a = [x_array[i], y_array[i]]
                            point_b = [x_array[i + 1], y_array[i + 1]]
                            sample(
                                param_array[i],
                                param_array[i],
                                point_a,
                                point_b,
                                depth + 1,
                            )

            # Sample further if one of the end points in None (i.e. a complex
            # value) or the three points are not almost collinear.
            elif (
                p[0] is None
                or p[1] is None
                or q[1] is None
                or q[0] is None
                or not flat(p, new_point, q)
            ):
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

        f_start_x = fx(start)
        f_start_y = fy(start)
        start_array = [f_start_x, f_start_y]
        f_end_x = fx(end)
        f_end_y = fy(end)
        end_array = [f_end_x, f_end_y]
        x_coords.append(f_start_x)
        y_coords.append(f_start_y)
        param.append(start)
        sample(start, end, start_array, end_array, 0)
        return x_coords, y_coords, param

    def get_points(self):
        """Return lists of coordinates for plotting. Depending on the
        `adaptive` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        Returns
        =======
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates
        """
        if not self.adaptive:
            return self._uniform_sampling()

        fx = lambdify([self.var], self.expr_x)
        fy = lambdify([self.var], self.expr_y)
        x_coords, y_coords, param = self.adaptive_sampling(
            fx, fy, self.start, self.end, self.depth
        )
        return (x_coords, y_coords, param)


### 3D lines
class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""

    is_2Dline = False
    is_3Dline = True

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
        self.n = kwargs.get("n", 300)
        self.scale = kwargs.get("xscale", "linear")

    def __str__(self):
        return "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end)),
        )

    def get_points(self):
        param = self._discretize(self.start, self.end, self.n, scale=self.scale)
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

    def _discretize(self, s1, e1, n1, scale1, s2, e2, n2, scale2):
        mesh_x = super()._discretize(s1, e1, n1, scale1)
        mesh_y = super()._discretize(s2, e2, n2, scale2)
        return np.meshgrid(mesh_x, mesh_y)


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a sympy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__()
        self.is_complex = kwargs.get("is_complex", False)
        func = float if not self.is_complex else complex
        self.expr = sympify(expr)
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = func(var_start_end_x[1])
        self.end_x = func(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = func(var_start_end_y[1])
        self.end_y = func(var_start_end_y[2])
        self.label = label
        self.n1 = kwargs.get("n1", 50)
        self.n2 = kwargs.get("n2", 50)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.modules = kwargs.get("modules", None)

    def __str__(self):
        return ("cartesian surface: %s for" " %s over %s and %s over %s") % (
            str(self.expr),
            str(self.var_x),
            str((self.start_x, self.end_x)),
            str(self.var_y),
            str((self.start_y, self.end_y)),
        )

    def get_data(self):
        mesh_x, mesh_y = self._discretize(
            self.start_x, self.end_x, self.n1, self.xscale,
            self.start_y, self.end_y, self.n2, self.yscale)

        from sympy import lambdify
        f = lambdify((self.var_x, self.var_y), self.expr, modules=self.modules)

        if self.is_complex and (self.modules == "mpmath"):
            mesh_z = self._evaluate_mpmath(f, [(x, y) for x, y in zip(mesh_x.flatten(), mesh_y.flatten())])
            mesh_z = np.array([float(a) for a in mesh_z]).reshape(mesh_x.shape)
        else:
            mesh_z = f(mesh_x, mesh_y)
            mesh_z = self._correct_size(mesh_z, mesh_x)

        mesh_z = mesh_z.astype(np.float64)
        mesh_z = np.ma.masked_invalid(mesh_z)

        if self.is_complex:
            return np.real(mesh_x), np.real(mesh_y), mesh_z
        return mesh_x, mesh_y, mesh_z


class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric sympy
    expressions and a range."""

    is_parametric = True

    def __init__(
        self,
        expr_x,
        expr_y,
        expr_z,
        var_start_end_u,
        var_start_end_v,
        label="",
        **kwargs
    ):
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
        self.n1 = kwargs.get("n1", 50)
        self.n2 = kwargs.get("n2", 50)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")

    def __str__(self):
        return (
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s"
        ) % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var_u),
            str((self.start_u, self.end_u)),
            str(self.var_v),
            str((self.start_v, self.end_v)),
        )

    def get_data(self):
        mesh_u, mesh_v = self._discretize(
            self.start_u,
            self.end_u,
            self.n1,
            self.xscale,
            self.start_v,
            self.end_v,
            self.n2,
            self.yscale,
        )

        fx = vectorized_lambdify((self.var_u, self.var_v), self.expr_x)
        fy = vectorized_lambdify((self.var_u, self.var_v), self.expr_y)
        fz = vectorized_lambdify((self.var_u, self.var_v), self.expr_z)

        mesh_x = fx(mesh_u, mesh_v)
        mesh_y = fy(mesh_u, mesh_v)
        mesh_z = fz(mesh_u, mesh_v)

        mesh_x = self._correct_size(np.array(mesh_x, dtype=np.float64), mesh_u)
        mesh_y = self._correct_size(np.array(mesh_y, dtype=np.float64), mesh_u)
        mesh_z = self._correct_size(np.array(mesh_z, dtype=np.float64), mesh_u)

        mesh_x = np.ma.masked_invalid(mesh_x)
        mesh_y = np.ma.masked_invalid(mesh_y)
        mesh_z = np.ma.masked_invalid(mesh_z)

        return mesh_x, mesh_y, mesh_z


class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False
    is_contour = True

    def __str__(self):
        return ("contour: %s for " "%s over %s and %s over %s") % (
            str(self.expr),
            str(self.var_x),
            str((self.start_x, self.end_x)),
            str(self.var_y),
            str((self.start_y, self.end_y)),
        )


class ImplicitSeries(BaseSeries):
    """Representation for Implicit plot

    References
    ==========

    .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
    Mathematical Formulae with Two Free Variables.

    .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
    Arithmetic. Master's thesis. University of Toronto, 1996
    """

    is_implicit = True

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__()
        expr, has_equality = self._has_equality(sympify(expr))
        self.expr = expr
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self.has_equality = has_equality
        self.n1 = kwargs.get("n1", 1000)
        self.n2 = kwargs.get("n2", 1000)
        self.label = label
        self.adaptive = kwargs.get("adaptive", False)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")

        if isinstance(expr, BooleanFunction):
            self.adaptive = True
            warnings.warn(
                "The provided expression contains Boolean functions. "
                + "In order to plot the expression, the algorithm "
                + "automatically switched to an adaptive sampling."
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
            if any(isinstance(e, (Equality, GreaterThan, LessThan)) for e in arg_list):
                has_equality = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_equality = True

        return expr, has_equality

    def __str__(self):
        return ("Implicit equation: %s for " "%s over %s and %s over %s") % (
            str(self.expr),
            str(self.var_x),
            str((self.start_x, self.end_x)),
            str(self.var_y),
            str((self.start_y, self.end_y)),
        )

    def get_data(self):
        func = experimental_lambdify(
            (self.var_x, self.var_y), self.expr, use_interval=True
        )
        xinterval = interval(self.start_x, self.end_x)
        yinterval = interval(self.start_y, self.end_y)
        try:
            func(xinterval, yinterval)
        except AttributeError:
            # XXX: AttributeError("'list' object has no attribute 'is_real'")
            # That needs fixing somehow - we shouldn't be catching
            # AttributeError here.
            if self.adaptive:
                warnings.warn(
                    "Adaptive meshing could not be applied to the"
                    " expression. Using uniform meshing."
                )
            self.adaptive = False

        if self.adaptive:
            return self._get_raster_interval(func)
        else:
            return self._get_meshes_grid()

    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
        k = self.depth
        interval_list = []
        # Create initial 32 divisions
        xsample = np.linspace(self.start_x, self.end_x, 33)
        ysample = np.linspace(self.start_y, self.end_y, 33)

        # Add a small jitter so that there are no false positives for equality.
        # Ex: y==x becomes True for x interval(1, 2) and y interval(1, 2)
        # which will draw a rectangle.
        jitterx = (
            (np.random.rand(len(xsample)) * 2 - 1)
            * (self.end_x - self.start_x)
            / 2 ** 20
        )
        jittery = (
            (np.random.rand(len(ysample)) * 2 - 1)
            * (self.end_y - self.start_y)
            / 2 ** 20
        )
        xsample += jitterx
        ysample += jittery

        xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1], xsample[1:])]
        yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1], ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        # recursive call refinepixels which subdivides the intervals which are
        # neither True nor False according to the expression.
        def refine_pixels(interval_list):
            """Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:

                # Convert the array indices to x and y values
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                # The expression is valid in the interval. Change the contour
                # array values to 1.
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    # Subdivide
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
        # Check whether the expression represents an equality
        # If it represents an equality, then none of the intervals
        # would have satisfied the expression due to floating point
        # differences. Add all the undecided values to the plot.
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return plot_list, "fill"

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        expr, equality = self._preprocess_meshgrid_expression(self.expr)
        xarray = self._discretize(self.start_x, self.end_x, self.n1, self.xscale)
        yarray = self._discretize(self.start_y, self.end_y, self.n2, self.yscale)
        x_grid, y_grid = np.meshgrid(xarray, yarray)
        func = vectorized_lambdify((self.var_x, self.var_y), expr)
        z_grid = func(x_grid, y_grid)
        z_grid, ones = self._postprocess_meshgrid_result(z_grid, x_grid)
        if equality:
            return xarray, yarray, z_grid, ones, "contour"
        else:
            return xarray, yarray, z_grid, ones, "contourf"

    @staticmethod
    def _preprocess_meshgrid_expression(expr):
        """If the expression is a Relational, rewrite it as a single
        expression. This method reduces code repetition.

        Returns
        =======
            expr : Expr
                The rewritten expression

            equality : Boolean
                Wheter the original expression was an Equality or not.
        """
        equality = False
        if isinstance(expr, Equality):
            expr = expr.lhs - expr.rhs
            equality = True

        elif isinstance(expr, (GreaterThan, StrictGreaterThan)):
            expr = expr.lhs - expr.rhs

        elif isinstance(expr, (LessThan, StrictLessThan)):
            expr = expr.rhs - expr.lhs
        else:
            raise NotImplementedError(
                "The expression is not supported for "
                "plotting in uniform meshed plot."
            )
        return expr, equality

    @staticmethod
    def _postprocess_meshgrid_result(z_grid, x_grid):
        """Bound the result to -1, 1. This method reduces code repetition.
        While with Matplotlib we can directly plot the result z_grid and set the
        contour levels, this is not possible with Plotly. Hence, Plotly will
        use the ones matrix. The result will be slightly different: while
        Matplotlib will render smooth lines, Plotly will looks
        square-ish/segmented.
        """
        z_grid = ImplicitSeries._correct_size(z_grid, x_grid)
        # ones contains data useful to plot regions, or in case of Plotly,
        # contour lines too.
        ones = np.ones_like(z_grid, dtype=np.int8)
        ones[np.ma.where(z_grid < 0)] = -1
        return z_grid, ones


##############################################################################
# Finding the centers of line segments or mesh faces
##############################################################################


def centers_of_segments(array):
    return np.mean(np.vstack((array[:-1], array[1:])), 0)


def centers_of_faces(array):
    return np.mean(
        np.dstack(
            (
                array[:-1, :-1],
                array[1:, :-1],
                array[:-1, 1:],
                array[:-1, :-1],
            )
        ),
        2,
    )


def flat(x, y, z, eps=1e-3):
    """Checks whether three points are almost collinear"""
    # Workaround plotting piecewise (#8577):
    #   workaround for `lambdify` in `.experimental_lambdify` fails
    #   to return numerical values in some cases. Lower-level fix
    #   in `lambdify` is possible.
    x, y, z = [np.real(t) for t in [x, y, z]]
    vector_a = (x - y).astype(np.float64)
    vector_b = (z - y).astype(np.float64)
    dot_product = np.dot(vector_a, vector_b)
    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)
    cos_theta = dot_product / (vector_a_norm * vector_b_norm)
    return abs(cos_theta + 1) < eps


class InteractiveSeries(BaseSeries):
    """Represent an interactive series, in which the expressions can be either
    a line or a surface (parametric or not). On top of the usual ranges (x, y or
    u, v, which must be provided), the expressions can use any number of
    parameters.

    This class internally convert the expressions to a lambda function, which is
    evaluated by calling update_data(params), passing in all the necessary
    parameters. Once update_data(params) has been executed, then get_data()
    can be used.

    NOTE: the __init__ method expects every expression to be already sympified.
    """

    is_interactive = True

    def __new__(cls, *args, **kwargs):
        if isinstance(args[0][0], Plane):
            return PlaneInteractiveSeries(*args, **kwargs)
        elif isinstance(args[0][0], GeometryEntity) and (
            not isinstance(args[0][0], Curve)
        ):
            return GeometryInteractiveSeries(*args, **kwargs)
        return object.__new__(cls)

    def __init__(self, exprs, ranges, label="", **kwargs):
        # take care of Curve from sympy.geometry, which can be seen as
        # parametric series
        if isinstance(exprs[0], Curve):
            c = exprs[0]
            exprs = c.functions
            ranges = [c.limits]

        # free symbols of the parameters
        params = kwargs.get("params", dict())
        # number of discretization points
        self.n1 = kwargs.get("n1", 250)
        self.n2 = kwargs.get("n2", 250)
        self.n3 = kwargs.get("n3", 250)
        n = [self.n1, self.n2, self.n3]

        # TODO / NOTE: even though we have the ComplexSeries and
        # ComplexInteractiveSeries classes, they are already doing a lot of work.
        # For the moment, we are going to allow InteractiveSeries to be able
        # to use complex discretization. In doing so, we can create 2D lines or
        # 3D surfaces of the real/imaginary/absolute value of a function.
        self.is_complex = kwargs.get("is_complex", False)
        castfunc = float if not self.is_complex else complex
        self.modules = kwargs.get("modules", None)
        self.absarg = kwargs.get("absarg", None)

        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.label = label
        nexpr, npar = len(exprs), len(ranges)

        if nexpr == 0:
            raise ValueError(
                "At least one expression must be provided."
                + "\nReceived: {}".format((exprs, ranges, label))
            )
        # # TODO: do I really need this?
        # if npar > 2:
        #     raise ValueError(
        #             "Depending on the backend, only 2D and 3D plots are " +
        #             "supported (1 or 2 ranges at most). The provided " +
        #             "expressions uses {} ranges.".format(npar))

        # set series attributes
        if (nexpr == 1) and (exprs[0].has(BooleanFunction) or exprs[0].has(Relational)):
            self.is_implicit = True
            exprs = list(exprs)
            exprs[0], self.equality = ImplicitSeries._preprocess_meshgrid_expression(
                exprs[0]
            )
        elif (nexpr == 1) and (npar == 1):
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


        if self.is_2Dline and self.is_complex and (self.absarg is not None):
            # here we are dealing with a complex line plot with absarg=True.
            # The series should return x, abs, arg so that the line can be
            # colored by the argument.
            self.is_parametric = True

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = set().union(*[e.free_symbols for e in exprs])
        fs = fs.difference(params.keys()).difference([r[0] for r in ranges])
        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n"
                + "Expression: {}\n".format((exprs, ranges, label))
                + "Specify what these symbols represent: {}\n".format(fs)
                + "Are they ranges or parameters?"
            )

        # Generate the lambda function.
        # If we are dealing with parametric expressions, we pack them into a
        # Tuple so that it can be lambdified
        self.expr = exprs[0] if len(exprs) == 1 else Tuple(*exprs, sympify=False)
        signature, f = get_lambda(self.expr, modules=self.modules)
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
            if i == 1:  # y direction
                scale = self.yscale

            discretizations.append(
                self._discretize(castfunc(r[1]), castfunc(r[2]), n[i], scale=scale)
            )

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
                slice_surf = _build_plane_series(_slice, ranges, **kwargs2)
                self.ranges = {
                    k: v for k, v in zip(discr_symbols, slice_surf.get_data())
                }
            else:
                # surfaces: needs mesh grids
                meshes = np.meshgrid(*discretizations)
                self.ranges = {k: v for k, v in zip(discr_symbols, meshes)}

        self.data = None
        if len(params) > 0:
            self.update_data(params)

    def _evaluate(self, params):
        """Update the data based on the values of the parameters.

        Parameters
        ==========

            params : dict
                key: symbol associated to the parameter
                val: the value

        """
        # discretized ranges all have the same shape. Take the first!
        discr = list(self.ranges.values())[0]

        if self.is_complex and (self.modules == "mpmath"):
            args = []
            for s in self.signature:
                if s in params.keys():
                    args.append((params[s] * np.ones_like(discr)).flatten())
                else:
                    args.append(self.ranges[s].flatten())

            args = np.array(args).T
            args = [list(t) for t in args]
            results = self._evaluate_mpmath(self.function, args)
            results = [float(r.real) for r in results]
            results = np.array(results).reshape(discr.shape)
        else:
            args = []
            for s in self.signature:
                if s in params.keys():
                    args.append(params[s])
                else:
                    args.append(self.ranges[s])
            results = self.function(*args)


        if self.is_complex and (self.absarg is not None):
            from sympy import lambdify, arg as argument
            f = lambdify(self.signature, argument(self.absarg), modules=self.modules)
            if self.modules != "mpmath":
                angle = f(*args)
            else:
                angle = self._evaluate_mpmath(f, args)
                angle = np.array([float(a) for a in angle]).reshape(discr.shape)
            results = [results, angle]

        if isinstance(results, (list, tuple)):
            results = list(results)
            for i, r in enumerate(results):
                results[i] = self._correct_size(
                    # the evaluation might produce an int/float. Need this conversion!
                    np.array(r),
                    discr,
                )
        elif isinstance(results, (int, float)):
            results = self._correct_size(
                # the evaluation might produce an int/float. Need this conversion!
                np.array(results),
                discr,
            )
        return results

    def update_data(self, params):
        """Update the data based on the values of the parameters.

        Parameters
        ==========

            params : dict
                key: symbol associated to the parameter
                val: the value

        """
        results = self._evaluate(params)

        if (
            self.is_contour
            or (self.is_3Dsurface and (not self.is_parametric))
            or (self.is_2Dline and (not self.is_parametric))
        ):
            # in the case of single-expression 2D lines or 3D surfaces
            if self.is_complex:
                results = [*self.ranges.values(), results]
                results = [np.real(r) for r in results]
            else:
                results = [*self.ranges.values(), results]
            self.data = results

        elif self.is_implicit:
            ranges = list(self.ranges.values())
            xr = ranges[0]
            yr = ranges[1]
            results = ImplicitSeries._postprocess_meshgrid_result(results, xr)
            results = [
                xr[0, :],
                yr[:, 0],
                *results,
                "contour" if self.equality else "contourf",
            ]
            self.data = results

        elif (self.is_parametric and self.is_2Dline and self.is_complex and
                (self.absarg is not None)):
            results = [*self.ranges.values(), *results]
            results = [np.real(r) for r in results]
            self.data = results

        elif self.is_parametric and (self.is_3Dline or self.is_2Dline):
            # also add the parameter
            results = [*results, *self.ranges.values()]
            self.data = results

        elif self.is_vector:
            # in order to plot a vector, we also need the discretized region
            self.data = [*self.ranges.values(), *results]

        else:
            self.data = results

    def get_data(self):
        # if the expression depends only on the ranges, the user can call get_data
        # directly without calling update_data
        if (self.data is None) and (len(self.signature) == len(self.ranges)):
            self.update_data(dict())
        if self.data is None:
            raise ValueError(
                "To generate the numerical data, call update_data(params), "
                + "providing the necessary parameters."
            )
        return self.data

    def __str__(self):
        ranges = [(k, v[0], v[-1]) for k, v in self.ranges.items()]
        return ("interactive expression: %s with ranges" " %s and parameters %s") % (
            str(self.expr),
            ", ".join([str(r) for r in ranges]),
            str(self.signature),
        )


class ComplexPointSeries(BaseSeries):
    is_2Dline = True
    is_point = True

    def __init__(self, expr, label="", **kwargs):
        self._init_attributes(expr, label)

    def _init_attributes(self, expr, label):
        print("expr", expr, type(expr))
        if isinstance(expr, (list, tuple)):
            self.expr = Tuple(*expr)
        elif isinstance(expr, Expr):
            self.expr = Tuple(expr)
        else:
            self.expr = expr
        self.label = label

    @staticmethod
    def _evaluate(points):
        x_list, y_list = [], []
        for p in points:
            x_list.append(float(re(p)))
            y_list.append(float(im(p)))
        return np.array(x_list), np.array(y_list)

    def get_data(self):
        return self._evaluate(self.expr)

class ComplexPointInteractiveSeries(InteractiveSeries, ComplexPointSeries):
    def __init__(self, expr, label="", **kwargs):
        self._init_attributes(expr, label)

        params = kwargs.get("params", dict())
        self.modules = kwargs.get("modules", None)

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = self.expr.free_symbols
        fs = fs.difference(params.keys())
        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n"
                + "Expression: {}\n".format((expr, r, label))
                + "Specify what these symbols represent: {}\n".format(fs)
                + "Are they ranges or parameters?"
            )

        # generate the lambda function
        signature, f = get_lambda(self.expr, modules=self.modules)
        self.signature = signature
        self.function = f

        # I need this line to reuse InteractiveSeries.get_data
        self.ranges = np.zeros(len(signature))

        self.data = None
        if len(params) > 0:
            self.update_data(params)

    def update_data(self, params):
        results = self.expr.subs(params)
        self.data = ComplexPointSeries._evaluate(results)

class ComplexSeries(BaseSeries):
    """Represent a complex number or a complex function."""

    is_complex = True
    is_point = False
    is_domain_coloring = False

    def __init__(self, expr, r, label="", **kwargs):
        expr = sympify(expr)
        self._init_attributes(expr, r, label, **kwargs)

    def _init_attributes(self, expr, r, label, **kwargs):
        """This method reduces code repetition between ComplexSeries and
        ComplexInteractiveSeries.
        """
        self.absarg = None
        self.modules = kwargs.get("modules", None)
        self.is_domain_coloring = kwargs.get("domain_coloring", False)

        self.var = sympify(r[0])
        self.start = complex(r[1])
        self.end = complex(r[2])
        if kwargs.get("threed", False):
            self.is_3Dsurface = True

        # TODO: do I need this???
        from sympy import lambdify
        self.function = lambdify([self.var], expr, modules=self.modules)

        self.expr = sympify(expr)
        self.n1 = kwargs.get("n1", 300)
        self.n2 = kwargs.get("n2", 300)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.label = label

        # domain coloring mode
        self.coloring = kwargs.get("coloring", "a")
        if not isinstance(self.coloring, str):
            raise ValueError("`coloring` must be of type string.")
        self.coloring = self.coloring.lower()
        self.phaseres = kwargs.get("phaseres", 20)
        # these will be passed to cplot.get_srgb1
        self.abs_scaling = kwargs.get("abs_scaling", "h-1")
        self.colorspace = kwargs.get("colorspace", "cam16")

    def _correct_output(self, x, r):
        """Obtain the correct output depending the initialized settings.

        This method reduces code repetition between ComplexSeries and
        ComplexInteractiveSeries.

        Parameters
        ==========
            x : np.ndarray
                Discretized domain. Can be a complex line or a complex region.

            r : np.ndarray
                Numerical evaluation result.
        """
        r = self._correct_size(np.array(r), np.array(x))

        if not self.is_domain_coloring:
            return np.real(x), np.imag(x), np.real(r)

        # 2D or 3D domain coloring
        return (
            np.real(x),
            np.imag(x),
            np.dstack([np.absolute(r), np.angle(r)]),
            *self._domain_coloring(r),
        )

    def _domain_coloring(self, w):
        from spb.ccomplex.hsv_color_grading import color_grading
        from spb.ccomplex.wegert import (
            bw_stripes_phase,
            bw_stripes_mag,
            domain_coloring,
            enhanced_domain_coloring,
            enhanced_domain_coloring_phase,
            enhanced_domain_coloring_mag,
            bw_stripes_imag,
            bw_stripes_real,
            cartesian_chessboard,
            polar_chessboard,
        )
        from cplot import get_srgb1

        _mapping = {
            "a": domain_coloring,
            "b": enhanced_domain_coloring,
            "c": enhanced_domain_coloring_mag,
            "d": enhanced_domain_coloring_phase,
            "e": color_grading,
            "f": None,
            "g": bw_stripes_mag,
            "h": bw_stripes_phase,
            "i": bw_stripes_real,
            "j": bw_stripes_imag,
            "k": cartesian_chessboard,
            "l": polar_chessboard,
        }
        colorscale = None
        if not self.coloring in _mapping.keys():
            raise KeyError(
                "`coloring` must be one of the following: {}".format(_mapping.keys())
            )

        if self.coloring == "f":
            zn = 1 * np.exp(1j * np.linspace(0, 2 * np.pi, 256))
            colorscale = get_srgb1(zn, self.abs_scaling, self.colorspace)
            colorscale = (colorscale * 255).astype(np.uint8)
            # shift the argument from [0, 2*pi] to [-pi, pi]
            colorscale = np.roll(colorscale, int(len(colorscale) / 2), axis=0)
            rgb = (get_srgb1(w, self.abs_scaling, self.colorspace) * 255).astype(
                np.uint8
            )
            return rgb, colorscale

        if self.coloring <= "e":
            from matplotlib.colors import hsv_to_rgb

            H = np.linspace(0, 1, 256)
            S = V = np.ones_like(H)
            colorscale = hsv_to_rgb(np.dstack([H, S, V]))
            colorscale = (colorscale.reshape((-1, 3)) * 255).astype(np.uint8)
            colorscale = np.roll(colorscale, int(len(colorscale) / 2), axis=0)
        return _mapping[self.coloring](w, phaseres=self.phaseres), colorscale

    def get_data(self):
        if isinstance(self.expr, (list, tuple, Tuple)):
            # list of complex points
            x_list, y_list = [], []
            for p in self.expr:
                x_list.append(float(re(p)))
                y_list.append(float(im(p)))
            return x_list, y_list

        # Domain coloring
        start_x = np.real(self.start)
        end_x = np.real(self.end)
        start_y = np.imag(self.start)
        end_y = np.imag(self.end)
        x = self._discretize(start_x, end_x, self.n1, self.xscale)
        y = self._discretize(start_y, end_y, self.n2, self.yscale)
        xx, yy = np.meshgrid(x, y)
        domain = xx + 1j * yy
        if self.modules != "mpmath":
            zz = self.function(domain)
        else:
            from mpmath import mpc
            zz = self._evaluate_mpmath(self.function,
                    [[mpc(r, i), ] for r, i in zip(xx.flatten(), yy.flatten())])
            zz = np.array([complex(t.real, t.imag) for t in zz]).reshape(domain.shape)
        return self._correct_output(domain, zz)


class ComplexInteractiveSeries(InteractiveSeries, ComplexSeries):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, expr, r, label="", **kwargs):
        params = kwargs.get("params", dict())

        self._init_attributes(expr, r, label, **kwargs)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.modules = kwargs.get("modules", None)

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = expr.free_symbols
        fs = fs.difference(params.keys()).difference(set([r[0]]))
        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n"
                + "Expression: {}\n".format((expr, r, label))
                + "Specify what these symbols represent: {}\n".format(fs)
                + "Are they ranges or parameters?"
            )

        # generate the lambda function
        signature, f = get_lambda(self.expr, modules=self.modules)
        self.signature = signature
        self.function = f

        # Discretize the ranges. In the following dictionary self.ranges:
        #    key: symbol associate to this particular range
        #    val: the numpy array representing the discretization

        # domain coloring
        x = self._discretize(
            complex(r[1]).real, complex(r[2]).real, self.n1, scale=self.xscale
        )
        y = self._discretize(
            complex(r[1]).imag, complex(r[2]).imag, self.n2, scale=self.yscale
        )
        xx, yy = np.meshgrid(x, y)
        zz = xx + 1j * yy
        self.ranges = {self.var: zz}

        self.data = None
        if len(params) > 0:
            self.update_data(params)

    def update_data(self, params):
        results = self._evaluate(params)
        self.data = self._correct_output(self.ranges[self.var], results)




def _set_discretization_points(kwargs, pt):
    """This function allows the user two use the keyword arguments n, n1 and n2
    to specify the number of discretization points in two directions.

    Parameters
    ==========

        kwargs : dict

        pt : type
            The type of the series, which indicates the kind of plot we are
            trying to create: plot, plot_parametric, ...
    """
    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries]:
        if "n1" in kwargs.keys() and ("n" not in kwargs.keys()):
            kwargs["n"] = kwargs["n1"]
    elif pt in [
        SurfaceOver2DRangeSeries,
        ContourSeries,
        ComplexSeries,
        ParametricSurfaceSeries,
        Vector2DSeries,
        ComplexInteractiveSeries,
        ImplicitSeries,
    ]:
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
    """Represent a vector field."""

    is_vector = True
    is_2D = False
    is_3D = False
    is_slice = False


class Vector2DSeries(VectorBase):
    """Represents a 2D vector field."""

    is_2Dvector = True

    def __init__(self, u, v, range1, range2, label, **kwargs):
        self.u = SurfaceOver2DRangeSeries(u, range1, range2, **kwargs)
        self.v = SurfaceOver2DRangeSeries(v, range1, range2, **kwargs)
        self.label = label

    def get_data(self):
        x, y, u = self.u.get_data()
        _, _, v = self.v.get_data()
        return x, y, self._correct_size(u, x), self._correct_size(v, x)


class Vector3DSeries(VectorBase):
    """Represents a 3D vector field."""

    is_3D = True
    is_3Dvector = True

    def __init__(self, u, v, w, range_x, range_y, range_z, label="", **kwargs):
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
        self.n1 = kwargs.get("n1", 10)
        self.n2 = kwargs.get("n2", 10)
        self.n3 = kwargs.get("n3", 10)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")

    def _discretize(self):
        """This method allows to reduce code repetition."""
        x = super()._discretize(self.start_x, self.end_x, self.n1, self.xscale)
        y = super()._discretize(self.start_y, self.end_y, self.n2, self.yscale)
        z = super()._discretize(self.start_z, self.end_z, self.n3, self.zscale)
        return np.meshgrid(x, y, z)

    def get_data(self):
        x, y, z = self._discretize()
        fu = vectorized_lambdify((self.var_x, self.var_y, self.var_z), self.u)
        fv = vectorized_lambdify((self.var_x, self.var_y, self.var_z), self.v)
        fw = vectorized_lambdify((self.var_x, self.var_y, self.var_z), self.w)
        uu = fu(x, y, z)
        vv = fv(x, y, z)
        ww = fw(x, y, z)
        uu = self._correct_size(uu, x)
        vv = self._correct_size(vv, y)
        ww = self._correct_size(ww, z)

        def _convert(a):
            a = np.array(a, dtype=np.float64)
            return np.ma.masked_invalid(a)

        return x, y, z, _convert(uu), _convert(vv), _convert(ww)


def _build_plane_series(plane, ranges, **kwargs):
    """This method reduced code repetition."""
    if isinstance(plane, Plane):
        return PlaneSeries(sympify(plane), *ranges, **kwargs)
    else:
        return SurfaceOver2DRangeSeries(plane, *ranges, **kwargs)


class SliceVector3DSeries(Vector3DSeries):
    """Represents a 3D vector field plotted over a slice, which can be a slice
    plane or a slice surface.
    """

    is_slice = True

    def __init__(self, plane, u, v, w, range_x, range_y, range_z, label="", **kwargs):
        self.plane = _build_plane_series(plane, [range_x, range_y, range_z], **kwargs)
        super().__init__(u, v, w, range_x, range_y, range_z, label, **kwargs)

    def _discretize(self):
        """This method allows to reduce code repetition."""
        return self.plane.get_data()


class PlaneSeries(SurfaceBaseSeries):
    """Represents a plane in a 3D domain."""

    is_3Dsurface = True

    def __init__(
        self, plane, x_range, y_range, z_range=None, label="", params=dict(), **kwargs
    ):
        self.plane = sympify(plane)
        if not isinstance(self.plane, Plane):
            raise TypeError("`plane` must be an instance of sympy.geometry.Plane")
        self.x_range = sympify(x_range)
        self.y_range = sympify(y_range)
        self.z_range = sympify(z_range)
        self.label = label
        self.n1 = kwargs.get("n1", 20)
        self.n2 = kwargs.get("n2", 20)
        self.n3 = kwargs.get("n3", 20)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")
        self.params = params

    def get_data(self):
        x, y, z = symbols("x, y, z")
        plane = self.plane.subs(self.params)
        fs = plane.equation(x, y, z).free_symbols
        xx, yy, zz = None, None, None
        if fs == set([x]):
            # parallel to yz plane (normal vector (1, 0, 0))
            s = SurfaceOver2DRangeSeries(
                self.plane.p1[0],
                (x, *self.z_range[1:]),
                (y, *self.y_range[1:]),
                "",
                n1=self.n3,
                n2=self.n2,
                xscale=self.xscale,
                yscale=self.yscale,
                zscale=self.zscale,
            )
            xx, yy, zz = s.get_data()
            xx, yy, zz = zz, yy, xx
        elif fs == set([y]):
            # parallel to xz plane (normal vector (0, 1, 0))
            s = SurfaceOver2DRangeSeries(
                self.plane.p1[1],
                (x, *self.x_range[1:]),
                (y, *self.z_range[1:]),
                "",
                n1=self.n1,
                n2=self.n3,
                xscale=self.xscale,
                yscale=self.yscale,
                zscale=self.zscale,
            )
            xx, yy, zz = s.get_data()
            xx, yy, zz = xx, zz, yy
        else:
            # parallel to xy plane, or any other plane
            eq = plane.equation(x, y, z)
            if z in eq.free_symbols:
                eq = solve(eq, z)[0]
            s = SurfaceOver2DRangeSeries(
                eq,
                (x, *self.x_range[1:]),
                (y, *self.y_range[1:]),
                "",
                n1=self.n1,
                n2=self.n2,
                xscale=self.xscale,
                yscale=self.yscale,
                zscale=self.zscale,
            )
            xx, yy, zz = s.get_data()
            if len(fs) > 1:
                idx = np.logical_or(zz < self.z_range[1], zz > self.z_range[2])
                zz[idx] = np.nan
        return xx, yy, zz


class PlaneInteractiveSeries(PlaneSeries, InteractiveSeries):
    """Represent a geometric plane.

    NOTE: In the MRO, PlaneSeries has the precedence over InteractiveSeries.
    This is because Numpy and Scipy don't have correspondence with Plane.
    Hence, we got to use get_data() implemented in PlaneSeries.
    """

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, exprs, ranges, label="", **kwargs):
        PlaneSeries.__init__(self, exprs[0], *ranges, label=label, **kwargs)

    def update_data(self, params):
        self.params = params


class GeometrySeries(BaseSeries):
    is_geometry = True

    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], Plane):
            return PlaneSeries(*args, **kwargs)
        elif isinstance(args[0], Curve):
            new_cls = (
                Parametric2DLineSeries
                if len(args[0].functions) == 2
                else Parametric3DLineSeries
            )
            label = [a for a in args if isinstance(a, str)]
            label = label[0] if len(label) > 0 else str(args[0])
            return new_cls(*args[0].functions, args[0].limits, label, **kwargs)
        return object.__new__(cls)

    def __init__(self, expr, _range=None, label="", params=dict(), **kwargs):
        if not isinstance(expr, GeometryEntity):
            raise ValueError(
                "`expr` must be a geomtric entity.\n"
                + "Received: type(expr) = {}\n".format(type(expr))
                + "Expr: {}".format(expr)
            )

        r = expr.free_symbols.difference(set(params.keys()))
        if len(r) > 0:
            raise ValueError(
                "Too many free symbols. Please, specify the values of the "
                + "following symbols with the `params` dictionary: {}".format(r)
            )

        self.expr = expr
        self._range = _range
        self.label = label
        self.params = params
        self.fill = kwargs.get("fill", True)
        if isinstance(expr, (LinearEntity3D, Point3D)):
            self.is_3Dline = True
            self.start = 0
            self.end = 0
            if isinstance(expr, Point3D):
                self.is_point = True
        elif isinstance(expr, LinearEntity2D) or (
            isinstance(expr, (Polygon, Circle, Ellipse)) and (not self.fill)
        ):
            self.is_2Dline = True
        elif isinstance(expr, Point2D):
            self.is_point = True
            self.is_2Dline = True

    def get_data(self):
        expr = self.expr.subs(self.params)
        if isinstance(expr, Point3D):
            return (
                np.array([expr.x], dtype=float),
                np.array([expr.y], dtype=float),
                np.array([expr.z], dtype=float),
                np.array([0], dtype=float),
            )
        elif isinstance(expr, Point2D):
            return np.array([expr.x], dtype=float), np.array([expr.y], dtype=float)
        elif isinstance(expr, Polygon):
            x = [float(v.x) for v in expr.vertices]
            y = [float(v.y) for v in expr.vertices]
            x.append(x[0])
            y.append(y[0])
            return np.array(x), np.array(y)
        elif isinstance(expr, Circle):
            cx, cy = float(expr.center[0]), float(expr.center[1])
            r = float(expr.radius)
            t = np.linspace(0, 2 * np.pi, 200)
            x, y = cx + r * np.cos(t), cy + r * np.sin(t)
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            return x, y
        elif isinstance(expr, Ellipse):
            cx, cy = float(expr.center[0]), float(expr.center[1])
            a = float(expr.hradius)
            e = float(expr.eccentricity)
            x = np.linspace(-a, a, 200)
            y = np.sqrt((a ** 2 - x ** 2) * (1 - e ** 2))
            x += cx
            x, y = np.concatenate((x, x[::-1])), np.concatenate((cy + y, cy - y[::-1]))
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            return x, y
        elif isinstance(expr, LinearEntity3D):
            p1, p2 = expr.points
            x = np.array([p1.x, p2.x], dtype=float)
            y = np.array([p1.y, p2.y], dtype=float)
            z = np.array([p1.z, p2.z], dtype=float)
            param = np.zeros_like(x)
            return x, y, z, param
        elif isinstance(expr, (Segment, Ray)):
            p1, p2 = expr.points
            x = np.array([p1.x, p2.x])
            y = np.array([p1.y, p2.y])
            return x.astype(float), y.astype(float)
        else:  # Line
            p1, p2 = expr.points
            if self._range is None:
                x = np.array([p1.x, p2.x])
                y = np.array([p1.y, p2.y])
            else:
                m = expr.slope
                q = p1[1] - m * p1[0]
                x = np.array([self._range[1], self._range[2]])
                y = m * x + q
            return x.astype(float), y.astype(float)


class GeometryInteractiveSeries(GeometrySeries, InteractiveSeries):
    """Represent a geometry entity.

    NOTE: In the MRO, GeometrySeries has the precedence over InteractiveSeries.
    This is because Numpy and Scipy don't have correspondence with Line,
    Segment, Polygon, ... Hence, we got to use get_data() implemented in
    GeometrySeries.
    """

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, exprs, ranges, label="", **kwargs):
        r = ranges[0] if len(ranges) > 0 else None
        GeometrySeries.__init__(self, exprs[0], _range=r, label=label, **kwargs)

    def update_data(self, params):
        self.params = params
