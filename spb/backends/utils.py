import numpy as np
from PIL import ImageColor
from matplotlib.colors import Colormap
import plotly.colors
from _plotly_utils.basevalidators import ColorscaleValidator

def convert_colormap(cm, to, n=256):
    """ Convert the provided colormap to a format usable by the specified
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
    assert isinstance(to, str)
    to = to.lower()
    assert to in ["matplotlib", "plotly", "k3d", "bokeh"]
    if not isinstance(cm, (str, list, tuple, np.ndarray, Colormap)):
        raise ValueError(
            "`cm` must be either:\n" + 
            "1. a string with the name of a Plotly color scale.\n" +
            "2. a list of string HEX colors (colorcet colormaps).\n" +
            "2. a list of float numbers between 0 and 1 (k3d colormaps).\n" +
            "3. an instance of matplotlib.colors.Colormap.\n" +
            "4. an array of colors extracted from a matplotlib.colors.Colormap.")
    
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
        elif (isinstance(cm, np.ndarray) or 
                all([isinstance(c, (list, tuple)) for c in cm])):
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
        elif (isinstance(cm, np.ndarray) or 
                all([isinstance(c, (list, tuple)) for c in cm])):
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
    elif to == "matplotlib": # to matplotlib
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
        elif (isinstance(cm, np.ndarray) or 
                all([isinstance(c, (list, tuple)) for c in cm])):
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
    else: # to bokeh
        if isinstance(cm, Colormap):
            # matplotlib color map
            discr = np.linspace(0, 1, n)
            colors = (cm(discr) * 255).astype(int)
            r = ['#%02x%02x%02x' % tuple(c[:-1]) for c in colors]
        elif (isinstance(cm, np.ndarray) or 
                all([isinstance(c, (list, tuple)) for c in cm])):
            if isinstance(cm, (list, tuple)):
                cm = np.array(cm)
            colors = (cm * 255).astype(int)

            if cm.shape[1] == 4:
                # matplotlib color map already extracted
                r = ['#%02x%02x%02x' % tuple(c[:-1]) for c in colors]
            else:
                # colorcet color map
                r = ['#%02x%02x%02x' % tuple(c) for c in colors]
        elif all([isinstance(t, (float, int)) for t in cm]):
            # k3d color map
            cm = np.array(cm).reshape(-1, 4)
            colors = (cm[:, 1:] * 255).astype(int)
            r = ['#%02x%02x%02x' % tuple(c) for c in colors]
        else:
            r = cm
    return r

def _get_continuous_color(colorscale, intermed):
    """ Computes the intermediate color for any value in the [0, 1] range of a
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
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")

def get_plotly_colors(colorscale_name, loc):
    """ Extract the color at the specified location from the specified Plotly's
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
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter for our use cae
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
    colorscale = cv.validate_coerce(colorscale_name)
    
    if hasattr(loc, "__iter__"):
        str_colors = [_get_continuous_color(colorscale, x) for x in loc]
        return [[float(t) / 255 for t in s[4:-1].split(",")] for s in str_colors]
    
    str_color = _get_continuous_color(colorscale, loc)
    return [float(t) / 255 for t in str_color[4:-1].split(",")]