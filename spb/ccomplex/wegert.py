"""
The following functions generate color schemes for complex domain coloring,
based on Elias Wegert's book
`"Visual Complex Functions" <https://www.springer.com/de/book/9783034801799>`_.
The book provides the background to better understand the images.

Original code:
https://it.mathworks.com/matlabcentral/fileexchange/44375-phase-plots-of-complex-functions?s_tid=mwa_osa_a

Elias Wegert kindly granted permission to re-license the colorscheme routine
under BSD 3 clauses.
"""

import matplotlib
import numpy as np


def to_rgb_255(func):
    """Convert a Numpy array with values in the range [0, 1] to [0, 255]."""

    def wrapper(*args, **kwargs):
        rgb = func(*args, **kwargs)
        return (rgb * 255).astype(np.uint8)

    return wrapper


def rect_func(x, dx):
    """Create rectangular impulses of length dx."""
    return np.mod(np.floor(x / dx), 2)


def saw_func(x, dx, a, b):
    """Saw tooth function on R with period dx onto [a,b]"""
    x = x / dx - np.floor(x / dx)
    return a + (b - a) * x


def alpha_blending(rgb, brightness):
    """Alpha blending between RGB image with a white image using the inverse
    of the brightness mask: zeros will retain colors, poles go white.
    """
    brightness = np.dstack([brightness, brightness, brightness])
    return (1 - brightness) * rgb + brightness


@to_rgb_255
def domain_coloring(w, phaseres=20, cmap=None, poffset=0, **kwargs):
    """Standard domain coloring."""
    arg = np.angle(w)
    # normalize the argument to [0, 1]
    arg = (arg / (2 * np.pi) + poffset) % 1

    if (cmap is None) or (cmap == matplotlib.cm.hsv):
        H = arg
        S = V = np.ones_like(H)
        rgb = matplotlib.colors.hsv_to_rgb(np.dstack([H, S, V]))
    else:
        rgb = cmap(arg)[:, :, :-1]

    if not kwargs.get("enhance", False):
        return rgb

    mag = np.absolute(w)
    brightness = mag / (mag + 1)
    return alpha_blending(rgb, brightness)


@to_rgb_255
def enhanced_domain_coloring(w, phaseres=20, cmap=None, blevel=0.75,
    poffset=0, **kwargs):
    """Enhanced domain coloring showing iso-lines for magnitude and phase,
    with an optional enhacement to wash poles to white.
    """
    mag, arg = np.absolute(w), np.angle(w)
    # normalize the argument to [0, 1]
    arg = (arg / (2 * np.pi) + poffset) % 1
    blackm = saw_func(np.log(mag), 2 * np.pi / phaseres, blevel, 1)
    blackp = saw_func(arg, 1 / phaseres, blevel, 1)
    black = blackp * blackm

    if (cmap is None) or (cmap == matplotlib.cm.hsv):
        # NOTE: work on HSV space in order to eliminate "fringing"
        H = arg
        S, V = np.ones_like(arg), black
        rgb = matplotlib.colors.hsv_to_rgb(np.dstack([H, S, V]))
    else:
        black = np.dstack([black, black, black])
        rgb = cmap(arg)[:, :, :-1] * black

    if not kwargs.get("enhance", False):
        return rgb

    mag = np.absolute(w)
    brightness = mag / (mag + 1)
    return alpha_blending(rgb, brightness)


@to_rgb_255
def enhanced_domain_coloring_phase(w, phaseres=20, cmap=None, blevel=0.75,
    poffset=0, **kwargs):
    """Enhanced domain coloring showing iso-lines for phase,
    with an optional enhacement to wash poles to white.
    """
    mag, arg = np.absolute(w), np.angle(w)
    # normalize the argument to [0, 1]
    arg = (arg / (2 * np.pi) + poffset) % 1
    blackp = saw_func(arg, 1 / phaseres, blevel, 1)

    if (cmap is None) or (cmap == matplotlib.cm.hsv):
        # NOTE: work on HSV space in order to eliminate "fringing"
        H, S, V = arg, np.ones_like(arg), blackp
        rgb = matplotlib.colors.hsv_to_rgb(np.dstack([H, S, V]))
    else:
        black = np.dstack([blackp, blackp, blackp])
        colors = cmap(arg)[:, :, :-1]
        rgb = colors * black

    if not kwargs.get("enhance", False):
        return rgb

    mag = np.absolute(w)
    brightness = mag / (mag + 1)
    return alpha_blending(rgb, brightness)


@to_rgb_255
def enhanced_domain_coloring_mag(w, phaseres=20, cmap=None, blevel=0.75,
    poffset=0, **kwargs):
    """Enhanced domain coloring showing iso-lines for magnitude,
    with an optional enhacement to wash poles to white.
    """
    mag, arg = np.absolute(w), np.angle(w)
    # normalize the argument to [0, 1]
    arg = (arg / (2 * np.pi) + poffset) % 1
    blackm = saw_func(np.log(mag), 2 * np.pi / phaseres, blevel, 1)

    if (cmap is None) or (cmap == matplotlib.cm.hsv):
        # NOTE: work on HSV space in order to eliminate "fringing"
        H, S, V = arg, np.ones_like(arg), blackm
        rgb = matplotlib.colors.hsv_to_rgb(np.dstack([H, S, V]))
    else:
        black = np.dstack([blackm, blackm, blackm])
        colors = cmap(arg)[:, :, :-1]
        rgb = colors * black

    if not kwargs.get("enhance", False):
        return rgb

    mag = np.absolute(w)
    brightness = mag / (mag + 1)
    return alpha_blending(rgb, brightness)


@to_rgb_255
def bw_magnitude(w, **kwargs):
    """Black and white magnitude: black are the zeros, white are the poles."""
    mag = np.absolute(w)
    brightness = mag / (mag + 1)
    return np.dstack([brightness, brightness, brightness])


@to_rgb_255
def bw_stripes_phase(w, phaseres=20, **kwargs):
    """Alternating black and white stripes corresponding to phase."""
    arg = np.angle(w)
    # normalize the argument to [0, 1]
    arg = (arg / (2 * np.pi)) % 1
    black = saw_func(arg, 1 / phaseres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
def bw_stripes_mag(w, phaseres=20, **kwargs):
    """Alternating black and white stripes corresponding to modulus."""
    mag = np.absolute(w)
    black = saw_func(np.log(mag), 2 * np.pi / phaseres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
def bw_stripes_imag(w, phaseres=20, **kwargs):
    """Alternating black and white stripes corresponding to imaginary part.
    In particular recommended for stream lines of potential flow.
    """
    imres = 10 / phaseres
    black = saw_func(np.imag(w), imres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
def bw_stripes_real(w, phaseres=20, **kwargs):
    """Alternating black and white stripes corresponding to real part.
    In particular recommended for stream lines of potential flow.
    """
    reres = 10 / phaseres
    black = saw_func(np.real(w), reres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
def cartesian_chessboard(w, phaseres=20, **kwargs):
    """Cartesian Chessboard on the complex points space. The result will hide
    zeros.
    """
    blackx = rect_func(np.real(w), 4 / phaseres)
    blacky = rect_func(np.imag(w), 4 / phaseres)
    white = np.mod(blackx + blacky, 2)
    return np.dstack([white, white, white])


@to_rgb_255
def polar_chessboard(w, phaseres=20, **kwargs):
    """Polar Chessboard on the complex points space. The result will show
    conformality.
    """
    mag, arg = np.absolute(w), np.angle(w)
    # normalize the argument to [0, 1]
    arg = (arg / (2 * np.pi)) % 1

    blackp = rect_func(arg, 1 / phaseres)
    blackm = rect_func(np.log(mag), 2 * np.pi / phaseres)
    black = np.mod(blackp + blackm, 2)
    return np.dstack([black, black, black])


@to_rgb_255
def create_colorscale(cmap, poffset=0, N=256):
    """Create a colorscale which will be used to map argument values from
    [-pi, pi] in a colorbar. Zero is associated to the starting color.

    Parameters
    ==========
        cmap :
            The colormap to use.
        N : int
            Number of discretized colors. Default to 256.

    Returns
    =======
        colorscale : np.ndarray [N x 3]
            Each row is an RGB colors (0 <= R,G,B <= 255).
    """
    t = np.linspace(0, 1, N)
    if (cmap is None) or (cmap == matplotlib.cm.hsv):
        H = t
        S = V = np.ones_like(H)
        colorscale = matplotlib.colors.hsv_to_rgb(
            np.dstack([H, S, V]).reshape((-1, 3)))
    else:
        colorscale = cmap(t)[:, :-1]

    # the value 0 must be at the center of the colorscale
    offset = int(len(colorscale) * (0.5 - poffset))
    colorscale = np.roll(colorscale, offset, axis=0)
    return colorscale


def wegert(coloring, w, phaseres=20, cmap="hsv_r", blevel=0.75, poffset=0, N=256):
    """ Choose between different domain coloring options.

    Parameters
    ==========

    coloring : str
        Default to `"a"`. Possible options:

        - ``"a"``: pure phase portrait, showing the argument of the complex
          function.
        - ``"b"``: enhanced domain coloring showing iso-modulus and iso-phase
          lines.
        - ``"c"``: enhanced domain coloring showing iso-modulus lines.
        - ``"d"``: enhanced domain coloring showing iso-phase lines.
        - ``"e"``: alternating black and white stripes corresponding to
          modulus.
        - ``"f"``: alternating black and white stripes corresponding to
          phase.
        - ``"g"``: alternating black and white stripes corresponding to
          real part.
        - ``"h"``: alternating black and white stripes corresponding to
          imaginary part.
        - ``"i"``: cartesian chessboard on the complex points space. The
          result will hide zeros.
        - ``"j"``: polar Chessboard on the complex points space. The result
          will show conformality.
        - ``"k"``: black and white magnitude of the complex function.
          Zeros are black, poles are white.
        - ``"l"``: enhanced domain coloring showing iso-modulus and iso-phase
          lines, blended with the magnitude: poles are white.
        - ``"m"``: enhanced domain coloring showing iso-modulus lines, blended
          with the magnitude: poles are white.
        - ``"n"``: enhanced domain coloring showing iso-phase lines, blended
          with the magnitude: poles are white.
        - ``"o"``: enhanced domain coloring showing iso-phase lines, blended
          with the magnitude: poles are white.

    w : ndarray [n x m]
        Numpy array with the results (complex numbers) of the evaluation of
        a complex function.

    phaseres : int
        Number of constant-phase lines.

    N : int
        Number of discretized color in the colorscale. Default to 256.

    Returns
    =======

    img : np.ndarray [n x m x 3]
        An array of RGB colors (0 <= R,G,B <= 255)

    colorscale : np.ndarray [N x 3] or None
        RGB colors to be used in the colorscale. If the function computes
        black and white colors, `None` will be returned.
    """

    mapping = {
        "a": [domain_coloring, True],
        "b": [enhanced_domain_coloring, True],
        "c": [enhanced_domain_coloring_mag, True],
        "d": [enhanced_domain_coloring_phase, True],
        "e": [bw_stripes_mag, False],
        "f": [bw_stripes_phase, False],
        "g": [bw_stripes_real, False],
        "h": [bw_stripes_imag, False],
        "i": [cartesian_chessboard, False],
        "j": [polar_chessboard, False],
        "k": [bw_magnitude, False],
        "l": [domain_coloring, True],
        "m": [enhanced_domain_coloring, True],
        "n": [enhanced_domain_coloring_mag, True],
        "o": [enhanced_domain_coloring_phase, True],
    }

    if not cmap:
        cmap = matplotlib.cm.hsv

    if isinstance(cmap, str):
        try:
            cmap = matplotlib.cm.get_cmap(cmap)
        except:
            # it might be a plotly colorscale
            from spb.backends.utils import convert_colormap
            cmap = convert_colormap(cmap, "matplotlib")

    if isinstance(cmap, (list, tuple, np.ndarray)):
        cmap = matplotlib.colors.ListedColormap(cmap)
    elif not isinstance(cmap, matplotlib.colors.Colormap):
        raise TypeError("`%s` is not a supported type for a colormap" % type(cmap))

    if coloring not in mapping.keys():
        raise KeyError(
            "`coloring` must be one of the following: {}".format(
                mapping.keys()))

    # normalize the phase offset
    poffset = (poffset / (2 * np.pi)) % 1

    kwargs = dict(phaseres=phaseres, cmap=cmap, blevel=blevel, poffset=poffset)
    if coloring in ["l", "m", "n", "o"]:
        kwargs["enhance"] = True

    func, create_cc = mapping[coloring]
    if create_cc:
        return [func(w, **kwargs),
            create_colorscale(cmap, poffset, N)]
    return func(w, **kwargs), None
