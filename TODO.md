## TODO

- [ ] enable the user to provide existing figures and plot over them.

- [ ] Implement parallel evaluation when `modules="mpmath"` or
  `modules="sympy"`. [Motivational example](https://stackoverflow.com/a/77163667/2329968).

- [ ] `plot3d_implicit` with matplotlib: possible by adding some dependency:
  [PyMCubes](https://github.com/pmneila/PyMCubes) or [scikit-image](https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html).

- [ ] `plot_implicit`: decouple `And` expressions and plot them with contours
  following a strategy similar to sage.

- [ ] 3D vector fields discretizing the volume with two polar parameters, or
  two cylindrical parameters.

- [ ] `plot_nyquist` and `plot_nichols` for Plotly/Bokeh.

- [ ] implement parametric-widgets support for `xlim, ylim, zlim` and
  `xlabel, ylabel, zlabel`.

- [ ] custom legend position with keyword argument.

- [ ] wireframe lines for iso-modulus, iso-phase on `analytic_landscape`.

- [ ] asymptotes on 3D plots: probably requires to break mesh connectivity.
