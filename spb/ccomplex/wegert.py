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

from sympy.external import import_module
np = import_module('numpy')


def to_rgb_255(func):
    """Convert a Numpy array with values in the range [0, 1] to [0, 255]."""

    def wrapper(*args, **kwargs):
        rgb = func(*args, **kwargs)
        return (rgb * 255).astype(np.uint8)

    return wrapper


def _hsv_to_rgb_helper(arr):
    matplotlib = import_module(
        'matplotlib',
        import_kwargs={'fromlist': ['colors']},
        min_module_version='1.1.0',
        catch=(RuntimeError,))
    return matplotlib.colors.hsv_to_rgb(arr)


def hsv_to_rgb(func):
    """Convert a Numpy array of HSV values to RGB values."""
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        return _hsv_to_rgb_helper(arr)
    return wrapper


def rect_func(x, dx):
    """Create rectangular impulses of length dx."""
    return np.mod(np.floor(x / dx), 2)


def saw_func(x, dx, a, b):
    """Saw tooth function on R with period dx onto [a,b]"""
    x = x / dx - np.floor(x / dx)
    return a + (b - a) * x


@to_rgb_255
def bw_stripes_phase(w, phaseres=20):
    """Alternating black and white stripes corresponding to phase."""
    arg = np.angle(w)
    # normalize the argument to [0, 1]
    arg[arg < 0] += 2 * np.pi
    arg /= 2 * np.pi
    black = saw_func(arg, 1 / phaseres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
def bw_stripes_mag(w, phaseres=20):
    """Alternating black and white stripes corresponding to modulus."""
    mag = np.absolute(w)
    black = saw_func(np.log(mag), 2 * np.pi / phaseres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
@hsv_to_rgb
def domain_coloring(w, phaseres=20):
    """Standard domain coloring."""
    arg = np.angle(w)
    # normalize the argument to [0, 1]
    arg[arg < 0] += 2 * np.pi
    arg /= 2 * np.pi
    H = arg
    S = V = np.ones_like(H)
    return np.dstack([H, S, V])


@to_rgb_255
@hsv_to_rgb
def enhanced_domain_coloring(w, phaseres=20):
    """Enhanced domain coloring showing iso-lines for magnitude and phase."""
    mag, arg = np.absolute(w), np.angle(w)
    # normalize the argument to [0, 1]
    arg[arg < 0] += 2 * np.pi
    arg /= 2 * np.pi
    blackp = saw_func(arg, 1 / phaseres, 0.75, 1)
    blackm = saw_func(np.log(mag), 2 * np.pi / phaseres, 0.75, 1)
    black = blackp * blackm
    H = arg
    S, V = np.ones_like(H), black
    return np.dstack([H, S, V])


@to_rgb_255
@hsv_to_rgb
def enhanced_domain_coloring_phase(w, phaseres=20):
    """Enhanced domain coloring showing iso-lines for phase."""
    arg = np.angle(w)
    # normalize the argument to [0, 1]
    arg[arg < 0] += 2 * np.pi
    arg /= 2 * np.pi
    blackp = saw_func(arg, 1 / phaseres, 0.75, 1)
    H = arg
    S, V = np.ones_like(H), blackp
    return np.dstack([H, S, V])


@to_rgb_255
@hsv_to_rgb
def enhanced_domain_coloring_mag(w, phaseres=20):
    """Enhanced domain coloring showing iso-lines for magnitude."""
    mag, arg = np.absolute(w), np.angle(w)
    # normalize the argument to [0, 1]
    arg[arg < 0] += 2 * np.pi
    arg /= 2 * np.pi
    blackm = saw_func(np.log(mag), 2 * np.pi / phaseres, 0.75, 1)
    H = arg
    S, V = np.ones_like(H), blackm
    return np.dstack([H, S, V])


@to_rgb_255
def bw_stripes_imag(w, phaseres=20):
    """Alternating black and white stripes corresponding to imaginary part.
    In particular recommended for stream lines of potential flow.
    """
    imres = 10 / phaseres
    black = saw_func(np.imag(w), imres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
def bw_stripes_real(w, phaseres=20):
    """Alternating black and white stripes corresponding to real part.
    In particular recommended for stream lines of potential flow.
    """
    reres = 10 / phaseres
    black = saw_func(np.real(w), reres, 0, 1)
    bmin, bmax = black.min(), black.max()
    black = np.floor(2 * (black - bmin) / (bmax - bmin))
    return np.dstack([black, black, black])


@to_rgb_255
def cartesian_chessboard(w, phaseres=20):
    """Cartesian Chessboard on the complex points space. The result will hide
    zeros.
    """
    blackx = rect_func(np.real(w), 4 / phaseres)
    blacky = rect_func(np.imag(w), 4 / phaseres)
    white = np.mod(blackx + blacky, 2)
    return np.dstack([white, white, white])


@to_rgb_255
def polar_chessboard(w, phaseres=20):
    """
    Polar Chessboard on the complex points space. The result will show
    conformality.
    """
    mag, arg = np.absolute(w), np.angle(w)
    # normalize the argument to [0, 1]
    arg[arg < 0] += 2 * np.pi
    arg /= 2 * np.pi

    blackp = rect_func(arg, 1 / phaseres)
    blackm = rect_func(np.log(mag), 2 * np.pi / phaseres)
    black = np.mod(blackp + blackm, 2)
    return np.dstack([black, black, black])


def create_colorscale(N=256):
    """
    Create a HSV colorscale which will be used to map argument values from
    [-pi, pi] in a colorbar. Red color is associated to zero.

    Parameters
    ==========
        N : int
            Number of discretized colors. Default to 256.

    Returns
    =======
        colorscale : np.ndarray [N x 3]
            Each row is an RGB colors (0 <= R,G,B <= 255).
    """
    H = np.linspace(0, 1, N)
    S = V = np.ones_like(H)
    colorscale = _hsv_to_rgb_helper(np.dstack([H, S, V]))
    colorscale = (colorscale.reshape((-1, 3)) * 255).astype(np.uint8)
    colorscale = np.roll(colorscale, int(len(colorscale) / 2), axis=0)
    return colorscale


def wegert(coloring, w, phaseres=20, N=256):
    """ Choose between different domain coloring options.

    Parameters
    ==========

    coloring : str
        Default to `"a"`. Possible options:

        - `"a"`: standard domain coloring using HSV.
        - `"b"`: enhanced domain coloring using HSV, showing iso-modulus
          and is-phase lines.
        - `"c"`: enhanced domain coloring using HSV, showing iso-modulus
          lines.
        - `"d"`: enhanced domain coloring using HSV, showing iso-phase
          lines.
        - `"e"`: alternating black and white stripes corresponding to
          modulus.
        - `"f"`: alternating black and white stripes corresponding to
          phase.
        - `"g"`: alternating black and white stripes corresponding to
          real part.
        - `"h"`: alternating black and white stripes corresponding to
          imaginary part.
        - `"i"`: cartesian chessboard on the complex points space. The
          result will hide zeros.
        - `"j"`: polar Chessboard on the complex points space. The result
          will show conformality.

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
        "j": [polar_chessboard, False]
    }

    if coloring not in mapping.keys():
        raise KeyError(
            "`coloring` must be one of the following: {}".format(
                mapping.keys())
        )
    func, create_cc = mapping[coloring]
    if create_cc:
        return func(w, phaseres), create_colorscale(N)
    return func(w, phaseres), None
