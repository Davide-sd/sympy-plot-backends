## TODO

In no particular order:

- [ ] Implement parallel evaluation when `modules="mpmath"` or
  `modules="sympy"`. [Motivational example](https://stackoverflow.com/a/77163667/2329968).

- [ ] `plot3d_implicit` with matplotlib: possible by adding some dependency:
  [PyMCubes](https://github.com/pmneila/PyMCubes) or [scikit-image](https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html).

- [ ] `plot_implicit`: decouple `And` expressions and plot them with contours
  following a strategy similar to sage.

- [ ] 3D vector fields: discretize the volume with two polar parameters, or
  two cylindrical parameters.

- [ ] Control system plotting for Plotly.

- [ ] implement parametric-widgets support for `xlim, ylim, zlim` and
  `xlabel, ylabel, zlabel`.

- [ ] Create an Animation class.

- [ ] Create a way to apply rotation/translation/scaling transformation to 3D
  entities (obviously, with interactive plotting in mind).

- [ ] Custom legend position with keyword argument.

- [ ] Apply `pi` tick format to axis labels.

- [ ] Wireframe lines for iso-modulus, iso-phase on `analytic_landscape`.

- [ ] Asymptotes on 3D plots: probably requires to break mesh connectivity.
