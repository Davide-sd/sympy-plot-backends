"""
Microsoft Public License (MS-PL)

This license governs use of the accompanying software. If you use the software,
you accept this license. If you do not accept the license, do not use the
software.

1. Definitions
The terms "reproduce," "reproduction," "derivative works," and "distribution"
have the same meaning here as under U.S. copyright law.

A "contribution" is the original software, or any additions or changes to the
software.

A "contributor" is any person that distributes its contribution under this
license.

"Licensed patents" are a contributor's patent claims that read directly on its
contribution.

2. Grant of Rights

(A) Copyright Grant- Subject to the terms of this license, including the license
conditions and limitations in section 3, each contributor grants you a
non-exclusive, worldwide, royalty-free copyright license to reproduce its
contribution, prepare derivative works of its contribution, and distribute its
contribution or any derivative works that you create.

(B) Patent Grant- Subject to the terms of this license, including the license
conditions and limitations in section 3, each contributor grants you a
non-exclusive, worldwide, royalty-free license under its licensed patents to
make, have made, use, sell, offer for sale, import, and/or otherwise dispose of
its contribution in the software or derivative works of the contribution in the
software.

3. Conditions and Limitations

(A) No Trademark License- This license does not grant you rights to use any
contributors' name, logo, or trademarks.

(B) If you bring a patent claim against any contributor over patents that you
claim are infringed by the software, your patent license from such contributor
to the software ends automatically.

(C) If you distribute any portion of the software, you must retain all
copyright, patent, trademark, and attribution notices that are present in the
software.

(D) If you distribute any portion of the software in source code form, you may
do so only under this license by including a complete copy of this license with
your distribution. If you distribute any portion of the software in compiled or
object code form, you may only do so under a license that complies with this
license.

(E) The software is licensed "as-is." You bear the risk of using it. The
contributors give no express warranties, guarantees or conditions. You may have
additional consumer rights under your local laws which this license cannot
change. To the extent permitted under your local laws, the contributors exclude
the implied warranties of merchantability, fitness for a particular purpose and
non-infringement.

Credit to: dawright
Article: https://www.codeproject.com/Articles/80641/Visualizing-Complex-Functions
"""

import numpy as np
from matplotlib.colors import hsv_to_rgb


def hsv_color_grading(w, **kwargs):
    """Compute HSV color grading according to the following article:
    https://www.codeproject.com/Articles/80641/Visualizing-Complex-Functions

    Parameters
    ==========
        w : array (NxM)
            Result of the evaluation of a complex function.

    Returns
    =======
        HSV : array (NxMx3)
            Array with values in the range [0, 1]
    """
    mag, arg = np.absolute(w), np.angle(w)
    # from [-pi, pi] to [0, 2*pi]
    arg[arg < 0] += 2 * np.pi
    # normalize the argument to [0, 1]
    h = arg / (2 * np.pi)

    # TODO: can this function be optmized for Python/Numpy???
    def func(m):
        # map the magnitude logrithmicly into the repeating interval 0 < r < 1
        # this is essentially where we are between countour lines
        r0, r1 = 0, 1
        while m > r1:
            r0, r1 = r1, r1 * np.e
        return r0, r1

    r0 = np.zeros_like(mag)
    r1 = np.zeros_like(mag)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            r0[i, j], r1[i, j] = func(mag[i, j])

    # this puts contour lines at 0, 1, e, e^2, e^3, ...
    r = (mag - r0) / (r1 - r0)

    # determine saturation and value based on r
    # p and q are complementary distances from a countour line
    p = np.zeros_like(r)
    idx = r < 0.5
    p[idx] = 2 * r[idx]
    idx = r >= 0.5
    p[idx] = 2 * (1 - r[idx])
    q = 1.0 - p

    # only let p and q go to zero very close to zero;
    # otherwise they should stay nearly 1
    # this keep the countour lines from getting thick
    p1 = 1 - q * q * q
    q1 = 1 - p * p * p
    # fix s and v from p1 and q1
    s = 0.4 + 0.6 * p1
    v = 0.6 + 0.4 * q1
    return np.dstack([h, s, v])


def color_grading(w, **kwargs):
    """Compute RGB image with color grading for complex functions.

    Parameters
    ==========
        w : array (NxM)
            Result of the evaluation of a complex function.

    Returns
    =======
        RGB : array (NxMx3)
            Values in the range [0, 255].
    """
    rgb = hsv_to_rgb(hsv_color_grading(w, **kwargs))
    return (rgb * 255).astype(np.uint8)
