# modify the following imports accordingly to your backend preference
from spb.backends.plotly import PB as TWO_D_B
from spb.backends.k3d import KB as THREE_D_B

from spb.plot import (
    plot as p,
    plot_parametric as pp,
    plot_contour as pc,
    plot3d as p3d,
    plot3d_parametric_line as p3dpl,
    plot3d_parametric_surface as p3dps
)
from spb.plot_implicit import plot_implicit as pi

def plot(*args, **kwargs):
    if "backend" not in kwargs.keys():
        kwargs["backend"] = TWO_D_B
    return p(*args, **kwargs)

def plot_parametric(*args, **kwargs):
    if "backend" not in kwargs.keys():
        kwargs["backend"] = TWO_D_B
    return pp(*args, **kwargs)

def plot_contour(*args, **kwargs):
    if "backend" not in kwargs.keys():
        kwargs["backend"] = TWO_D_B
    return pc(*args, **kwargs)

def plot3d(*args, **kwargs):
    if "backend" not in kwargs.keys():
        kwargs["backend"] = THREE_D_B
    return p3d(*args, **kwargs)

def plot3d_parametric_line(*args, **kwargs):
    if "backend" not in kwargs.keys():
        kwargs["backend"] = THREE_D_B
    return p3dpl(*args, **kwargs)

def plot3d_parametric_surface(*args, **kwargs):
    if "backend" not in kwargs.keys():
        kwargs["backend"] = THREE_D_B
    return p3dps(*args, **kwargs)

def plot_implicit(*args, **kwargs):
    if "backend" not in kwargs.keys():
        kwargs["backend"] = TWO_D_B
    return pi(*args, **kwargs)