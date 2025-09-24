"""
The following code comes from ``control 0.10.0``, specifically ``grid.py``.
"""

import mpl_toolkits.axisartist.angle_helper as angle_helper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
    GridHelperCurveLinear


class FormatterDMS(object):
    '''Transforms angle ticks to damping ratios'''
    def __call__(self, direction, factor, values):
        angles_deg = np.asarray(values)/factor
        damping_ratios = np.cos((180-angles_deg) * np.pi/180)
        ret = ["%.2f" % val for val in damping_ratios]
        return ret


class ModifiedExtremeFinderCycle(angle_helper.ExtremeFinderCycle):
    '''Changed to allow only left hand-side polar grid

    https://matplotlib.org/_modules/mpl_toolkits/axisartist/angle_helper.html#ExtremeFinderCycle.__call__
    '''
    def __call__(self, transform_xy, x1, y1, x2, y2):
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))

        with np.errstate(invalid='ignore'):
            if self.lon_cycle is not None:
                lon0 = np.nanmin(lon)
                # Changed from 180 to 360 to be able to span only
                # 90-270 (left hand side)
                lon -= 360. * ((lon - lon0) > 360.)
            if self.lat_cycle is not None:  # pragma: no cover
                lat0 = np.nanmin(lat)
                lat -= 360. * ((lat - lat0) > 180.)

        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

        lon_min, lon_max, lat_min, lat_max = \
            self._add_pad(lon_min, lon_max, lat_min, lat_max)

        # check cycle
        if self.lon_cycle:
            lon_max = min(lon_max, lon_min + self.lon_cycle)
        if self.lat_cycle:  # pragma: no cover
            lat_max = min(lat_max, lat_min + self.lat_cycle)

        if self.lon_minmax is not None:
            min0 = self.lon_minmax[0]
            lon_min = max(min0, lon_min)
            max0 = self.lon_minmax[1]
            lon_max = min(max0, lon_max)

        if self.lat_minmax is not None:
            min0 = self.lat_minmax[0]
            lat_min = max(min0, lat_min)
            max0 = self.lat_minmax[1]
            lat_max = min(max0, lat_max)

        return lon_min, lon_max, lat_min, lat_max


def sgrid_auto(fig=None):
    # From matplotlib demos:
    # https://matplotlib.org/gallery/axisartist/demo_curvelinear_grid.html
    # https://matplotlib.org/gallery/axisartist/demo_floating_axis.html

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degrees
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform(
        # NOTE: https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#applying-theta-transforms-in-polartransform
        apply_theta_transforms=False
    )

    # polar projection, which involves cycle, and also has limits in
    # its coordinates, needs a special method to find the extremes
    # (min, max of the coordinate within the view).

    # 20, 20 : number of sampling points along x, y direction
    sampling_points = 20
    extreme_finder = ModifiedExtremeFinderCycle(
        sampling_points, sampling_points, lon_cycle=360, lat_cycle=None,
        lon_minmax=(90, 270), lat_minmax=(0, np.inf),)

    grid_locator1 = angle_helper.LocatorDMS(10)
    tick_formatter1 = FormatterDMS()
    grid_helper = GridHelperCurveLinear(
        tr, extreme_finder=extreme_finder, grid_locator1=grid_locator1,
        tick_formatter1=tick_formatter1)

    if not fig:
        # Set up an axes with a specialized grid helper
        fig = plt.gcf()

    ax = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    # make ticklabels of right invisible, and top axis visible.
    visible = True
    ax.axis[:].major_ticklabels.set_visible(visible)
    ax.axis[:].major_ticks.set_visible(False)
    ax.axis[:].invert_ticklabel_direction()
    ax.axis[:].major_ticklabels.set_color('gray')

    # Set up internal tickmarks and labels along the real/imag axes
    ax.axis["wnxneg"] = axis = ax.new_floating_axis(0, 180)
    axis.set_ticklabel_direction("-")
    axis.label.set_visible(False)

    ax.axis["wnxpos"] = axis = ax.new_floating_axis(0, 0)
    axis.label.set_visible(False)

    ax.axis["wnypos"] = axis = ax.new_floating_axis(0, 90)
    axis.label.set_visible(False)
    axis.set_axis_direction("right")

    ax.axis["wnyneg"] = axis = ax.new_floating_axis(0, 270)
    axis.label.set_visible(False)
    axis.set_axis_direction("left")
    axis.invert_ticklabel_direction()
    axis.set_ticklabel_direction("-")

    # let left axis shows ticklabels for 1st coordinate (angle)
    ax.axis["left"].get_helper().nth_coord_ticks = 0
    ax.axis["right"].get_helper().nth_coord_ticks = 0
    ax.axis["left"].get_helper().nth_coord_ticks = 0
    ax.axis["bottom"].get_helper().nth_coord_ticks = 0

    fig.add_subplot(ax)
    ax.grid(True, zorder=0, linestyle='dotted')
    return ax
