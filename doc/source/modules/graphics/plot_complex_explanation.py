import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorcet

fig = plt.figure(constrained_layout=True, figsize=(6, 1.75))
subfigs = fig.subfigures(1, 2)

ax0 = subfigs[0].add_subplot()
arg1 = np.linspace(-np.pi, np.pi, 256)
arg2 = (arg1 / (2 * np.pi)) % 1
ax0.plot(arg1, arg2)
ax0.set_xlabel("phase [rad]")
ax0.set_ylabel("normalized phase")
ax0.set_xlim(-np.pi, np.pi)
ax0.set_ylim(0, 1)

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
cmaps = {
    "hsv": "hsv",
    "twilight": "twilight",
    "colorwheel": colorcet.colorwheel,
    "CET-C7": colorcet.CET_C7,
    "viridis": "viridis"
}

ax1 = subfigs[1].subplots(len(cmaps), 1)
for ax, name in zip(ax1, cmaps):
    if isinstance(cmaps[name], str):
        cmap = matplotlib.cm.get_cmap(name)
    else:
        cmap = matplotlib.colors.ListedColormap(cmaps[name])
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
            transform=ax.transAxes)
    ax.axis(False)