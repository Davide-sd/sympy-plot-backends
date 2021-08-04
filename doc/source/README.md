# Quirks of This Documentation

To properly build the documentation, Matplotlib must be the default 2D and 3D backend. Also, the custom user settings folder must be deleted before running `make html`, which in Linux OS is located at:

```
/home/user/.local/share/spb
```

Different plotting libraries have different JS dependencies:

* Matplotlib with `ipympl`
* Bokeh
* Plotly
* K3D
* Holoviz's Panel

In particular:

1. `spb.functions`, `spb.vectors`, `spb.ccomplex` uses Matplotlib-Sphinx `plot` extension, hence Matplotlib MUST be the default backend.
2. `spb.interactive` uses the `jupyter_sphinx` extension to render the examples on the documentation. Currently, it doesn't appear to work with K3D: it loads the plot, but not the data.
3. All tutorials are "hand-written" from the respecting notebooks. Why not let Sphinx compiles them and save time? For one reason or another, it is impossible: too many different javascript libraries. Sphinx is not able to deal with them. For example, if `%matplotlib widget` is used, no matplotlib figure is show. K3D doesn't save/load data, ... By taking the long road, rewriting them and hand-inserting the pictures, we get quite a decent documentation.
4. `Tutorials/Parametric-Interactive Plots` is using `jupyter_sphinx` to view the panel's applications.
5. All tutorials containing both Bokeh/Plotly/K3D static images and `iplot`, the latter output images have not been included. At this moment it is impossible even using `jupyter_sphinx`: JS clashes.

To save the plots:
* Matplotlib: just use `ipympl` save button.
* Bokeh: just use `p.save(filename, CDN [optional])`
* Plotly: just use `p.save(filenameinclude_plotlyjs="require", full_html=False)`
* K3D: use:

```
with open('fig-01.html', 'w') as f:
    p.fig.snapshot_include_js = False
    f.write(p.fig.get_snapshot())
```

Once K3D plot has been saved, we need to modify it:

* Open the file in a text editor
* Replace `"canvasTarget"` with a unique ID.
* Remove everything above the body-script tag
* Insert the following at the top:
```
<div style="display: flex; height: 400px;">
<div id="UNIQUE-ID" style="position: relative; width: 100%;"></div>
```
* Remove closing body/html tags with `</div>`.

All images have been moved in the `tutorials/fig` folder in a logical manner.

Generate the documentation with `make clean html`.

Insert `plotly.js` and `k3d.js` into `doc/build/html/tutorials`

Everything is ready to be used/viewed.
