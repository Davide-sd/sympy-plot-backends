from spb.series import Parametric3DLineSeries, ComplexParametric3DLineSeries
from sympy import Dummy, sin, cos, sympify, symbols, I, re, im
from sympy.external import import_module


def _plot3d_wireframe_helper(surfaces, **kwargs):
    """Create data series representing wireframe lines.

    Parameters
    ==========
    surfaces : list of BaseSeries

    Returns
    =======
    line_series : list of Parametric3DLineSeries
    """
    if not kwargs.get("wireframe", False):
        return []

    np = import_module('numpy')
    lines = []
    wf_n1 = kwargs.get("wf_n1", 10)
    wf_n2 = kwargs.get("wf_n2", 10)
    npoints = kwargs.get("wf_npoints", None)
    wf_rend_kw = kwargs.get("wf_rendering_kw", dict())

    wf_kwargs = dict(
        use_cm=False, show_in_legend=False,
        n=npoints, rendering_kw=wf_rend_kw
    )

    def create_series(expr, ranges, surface_series, **kw):
        expr = [e if callable(e) else sympify(e) for e in expr]
        kw["tx"] = surface_series.tx
        kw["ty"] = surface_series.ty
        kw["tz"] = surface_series.tz
        if hasattr(surface_series, "tp"):
            kw["tp"] = surface_series.tp
        kw["force_real_eval"] = surface_series.force_real_eval
        if "return" not in kw.keys():
            return Parametric3DLineSeries(*expr, *ranges, "__k__", **kw)
        return ComplexParametric3DLineSeries(*expr, *ranges, "__k__", **kw)

    # NOTE: can't use np.linspace because start, end might be
    # symbolic expressions
    def linspace(start, end, n):
        return [start + (end - start) * i / (n - 1) for i in range(n)]

    for s in surfaces:
        param_expr, ranges = [], []

        if s.is_3Dsurface:
            expr = s.expr

            kw = wf_kwargs.copy()
            if s.is_interactive:
                # pass in the original parameters provided by the user, which
                # might contain RangeSlider...
                kw["params"] = s._original_params.copy()

            if s.is_parametric:
                (x, sx, ex), (y, sy, ey) = s.ranges
                is_callable = any(callable(e) for e in expr)

                for uval in linspace(sx, ex, wf_n1):
                    kw["n"] = s.n[1] if npoints is None else npoints
                    if is_callable:
                        # NOTE: closure on lambda functions
                        param_expr = [lambda t, uv=uval, e=e: e(float(uv), t) for e in expr]
                        ranges = [(y, sy, ey)]
                    else:
                        param_expr = [e.subs(x, uval) for e in expr]
                        ranges = [(y, sy, ey)]
                    lines.append(create_series(param_expr, ranges, s, **kw))
                for vval in linspace(sy, ey, wf_n2):
                    kw["n"] = s.n[0] if npoints is None else npoints
                    if is_callable:
                        # NOTE: closure on lambda functions
                        param_expr = [lambda t, vv=vval, e=e: e(t, float(vv)) for e in expr]
                        ranges = [(x, sx, ex)]
                    else:
                        param_expr = [e.subs(y, vval) for e in expr]
                        ranges = [(x, sx, ex)]
                    lines.append(create_series(param_expr, ranges, s, **kw))

            else:
                if not s.is_complex:
                    (x, sx, ex), (y, sy, ey) = s.ranges
                else:
                    x, y = symbols("x, y", cls=Dummy)
                    z, start, end = s.ranges[0]
                    expr = s.expr.subs(z, x + I * y)
                    sx, ex = re(start), re(end)
                    sy, ey = im(start), im(end)
                    kw["return"] = s._return

                if not s.is_polar:
                    for xval in linspace(sx, ex, wf_n1):
                        kw["n"] = s.n[1] if npoints is None else npoints
                        if callable(expr):
                            # NOTE: closure on lambda functions
                            param_expr = [
                                lambda t, xv=xval: xv,
                                lambda t: t,
                                lambda t, xv=xval: expr(float(xv), t)]
                            ranges = [(y, sy, ey)]
                        else:
                            param_expr = [xval, y, expr.subs(x, xval)]
                            ranges = [(y, sy, ey)]
                        lines.append(create_series(param_expr, ranges, s, **kw))
                    for yval in linspace(sy, ey, wf_n2):
                        kw["n"] = s.n[0] if npoints is None else npoints
                        if callable(expr):
                            # NOTE: closure on lambda functions
                            param_expr = [
                                lambda t: t,
                                lambda t, yv=yval: yv,
                                lambda t, yv=yval: expr(t, float(yv))]
                            ranges = [(x, sx, ex)]
                        else:
                            param_expr = [x, yval, expr.subs(y, yval)]
                            ranges = [(x, sx, ex)]
                        lines.append(create_series(param_expr, ranges, s, **kw))
                else:
                    for rval in linspace(sx, ex, wf_n1):
                        kw["n"] = s.n[1] if npoints is None else npoints
                        if callable(expr):
                            param_expr = [
                                lambda t, rv=rval: float(rv) * np.cos(t),
                                lambda t, rv=rval: float(rv) * np.sin(t),
                                lambda t, rv=rval: expr(float(rv), t)]
                            ranges = [(y, sy, ey)]
                        else:
                            param_expr = [rval * cos(y), rval * sin(y), expr.subs(x, rval)]
                            ranges = [(y, sy, ey)]
                        lines.append(create_series(param_expr, ranges, s, **kw))
                    for tval in linspace(sy, ey, wf_n2):
                        kw["n"] = s.n[0] if npoints is None else npoints
                        if callable(expr):
                            param_expr = [
                                lambda p, tv=tval: p * np.cos(float(tv)),
                                lambda p, tv=tval: p * np.sin(float(tv)),
                                lambda p, tv=tval: expr(p, float(tv))]
                            ranges = [(x, sx, ex)]
                        else:
                            param_expr = [x * cos(tval), x * sin(tval), expr.subs(y, tval)]
                            ranges = [(x, sx, ex)]
                        lines.append(create_series(param_expr, ranges, s, **kw))

    return lines


def _plot_sympify(expr):
    if callable(expr):
        return expr
    return sympify(expr)
