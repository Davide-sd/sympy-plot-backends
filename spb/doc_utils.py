"""A few handy functions to modify code blocks before they are executed on
Sphinx.

An instance of the Plot class contains a figure created with some plotting
library. In case of ordinary plots, Sphinx extensions need to retrieve the
figure in order to save it to a png (and or pdf, html) file. In case of
interactive widget plots, a Sphinx extension needs to retrieve the html
code of the interactive application. So, we need a way to tell Sphinx
extensions where to look for the data they need.

But docstring examples are designed to be executed by the users, so adding
code to retrieve the figure in the code blocks is useless and confusing.

The following functions uses the ast module to preprocess a code block, add
the necessary code to extract data.
"""

import ast


def _modify_plot_expr(expr):
    expr = expr.value
    if not isinstance(expr.func, ast.Name):
        # for example: (p1 + p2).show()
        return

    func_name = expr.func.id
    # look for function calls starting with "plot"
    if (len(func_name) >= 4) and (func_name[:4] == "plot"):
        found_show = False
        # loop over kwargs, if "show" is already present, set its
        # value to False
        for kw in expr.keywords:
            if kw.arg == "show":
                found_show = True
                kw.value.value = False
                break
        # if "show" is not present, then add it
        if not found_show:
            expr.keywords.append(ast.keyword(arg='show',
                value=ast.Constant(value=False)))


def _modify_code(code):
    """In the docstrings, the last command of each example is either:
    1. plot_something(...) # plot command can span multiple rows
    2. (p1 + p2 + ...).show()

    Either way, the ``.. plotly`` directive is unable to extract the Plotly
    figure from the `Plot` object. This function parses the `code` and apply
    a few modifications. In particular:

    1. plot_something(...) will be transformed to:
       myplot = plot_something(..., show=False)
       myplot.fig
    2. (p1 + p2 + ...).show() will be transformed to:
       myplot = p1 + p2 + ...
       myplot.fig

    So, the last command will actually be the Plotly figure. Therefore, the
    sphix_plotly_directive extension will work as expected.

    Parameters
    ==========
    code : str
        The current code block being processed.

    Returns
    =======
    modified_code : str
    """
    tree = ast.parse(code)

    if any(t in code for t in ["backend=KB", "backend = KB"]):
        # plotting some 3D plot with K3D Jupyter.
        # Skip Jupyter Notebook check.
        # Assume import statements are all at the beginning.
        num_import_statements = 0
        for t in tree.body:
            if isinstance(t, ast.Import) or isinstance(t, ast.ImportFrom):
                num_import_statements += 1
        obj = ast.parse("KB.skip_notebook_check=True")
        tree.body.insert(num_import_statements, obj.body[0])

    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            # example: p1 = plot(...)
            _modify_plot_expr(node)
        elif isinstance(node, ast.Expr):
            # example: plot(...)
            _modify_plot_expr(node)

    # last node
    ln = tree.body[-1]
    if isinstance(ln, ast.Expr) and isinstance(ln.value, ast.Call):
        if (isinstance(ln.value.func, ast.Attribute) and
            (ln.value.func.attr == "show")):
            tree.body[-1] = ast.Assign(targets=[ast.Name(id="myplot")],
                value=ln.value.func.value, lineno=ln.lineno)
        else:
            # if the last command is a plot function call (for example,
            # plot(...), modify it to be an assignment: myplot = plot(...)
            tree.body[-1] = ast.Assign(targets=[ast.Name(id="myplot")],
                value=ln.value, lineno=ln.lineno)

    # finally, append myplot.fig to the ast
    tree.body.append(ast.Expr(value=ast.Attribute(
        value=ast.Name(id="myplot"), attr="fig")))
    return ast.unparse(tree)


def _modify_iplot_code(code):
    """What this function does:

    1. Look for the last command to be ipot or plot_something(params={},
       servable=True).
    2. Remove servable
    3. set show=False and imodule="panel"
    4. manually create a template and apply the content.
    5. Returns the template.

    Parameters
    ==========
    code : str
        The current code block being processed.

    Returns
    =======
    modified_code : str
    """
    tree = ast.parse(code)
    ln = tree.body[-1]

    if (isinstance(ln, ast.Expr) and isinstance(ln.value, ast.Call) and
        isinstance(ln.value.func, ast.Name)):
        func_name = tree.body[-1].value.func.id
        if func_name == "iplot" or (func_name[:4] == "plot"):
            params_node, servable_node = None, None
            show_node, imodule_node, backend_node = None, None, None
            for kw in tree.body[-1].value.keywords:
                if kw.arg == "params":
                    params_node = kw
                elif kw.arg == "servable":
                    servable_node = kw
                elif kw.arg == "show":
                    show_node = kw
                elif kw.arg == "imodule":
                    imodule_node = kw
                elif kw.arg == "backend":
                    backend_node = kw
            if imodule_node is not None:
                imodule_node.value.value = "panel"
            else:
                tree.body[-1].value.keywords.append(
                    ast.keyword(arg='imodule', value=ast.Constant(value="panel"))
                )
            is_KB = backend_node and (backend_node.value.id in ["KB", "K3DBackend"])
            if ((params_node is None) or (servable_node is None)) and (not is_KB):
                return ast.unparse(tree)
            if servable_node is None:
                servable_node = ast.keyword(
                    arg='servable_node', value=ast.Constant(value=False))
            if (not servable_node.value.value) and (not is_KB):
                return ast.unparse(tree)
            servable_node.value.value = False
            if show_node is not None:
                show_node.value.value = False
            else:
                tree.body[-1].value.keywords.append(
                    ast.keyword(arg='show', value=ast.Constant(value=False))
                )
            # the default width of the sidebar is good for full screen, but
            # it's too small for demo plots in the docs. Let's increase it.
            tree.body[-1].value.keywords.append(
                ast.keyword(arg='template', value=ast.Constant(
                    value={'sidebar_width': '300px'}))
                )
            tree.body[-1] = ast.Assign(targets=[ast.Name(id="panelplot")],
                value=ln.value, lineno=ln.lineno)

            if is_KB:
                last_command = ast.parse("panelplot.layout_controls()")
            else:
                last_command = ast.parse("panelplot._create_template(show=False)")

            tree.body.append(last_command.body[-1])
    return ast.unparse(tree)


def postprocess_KB_interactive_image(ns, size, img, browser, browser_path,
    driver_path, driver_options=[]):
    import numpy as np
    from PIL import Image
    from spb.interactive.panel import InteractivePlot
    from spb import KB
    from sphinx_k3d_screenshot.utils import get_k3d_screenshot, get_driver

    if "panelplot" not in ns.keys():
        return img

    panelplot = ns["panelplot"]
    if not isinstance(panelplot, InteractivePlot):
        return img
    if not isinstance(panelplot.backend, KB):
        return img

    # guesstimate (in pixel) for the vertical space of each row of widgets
    hr = 50
    nr = int(np.ceil(len(panelplot.backend[0].params) / panelplot._ncols))
    # need to pad before first row and after last row. Guesstimate for padding.
    pad_h = 25
    h = hr * nr + 2 * pad_h
    img = img.crop((0, 0, img.width, h))

    driver = get_driver(
        browser, browser_path, driver_path, driver_options)
    plot = get_k3d_screenshot(driver, size, panelplot.fig)

    # concatenate vertically the two images
    final = Image.new('RGB', (img.width, img.height + plot.height))
    final.paste(img, (0, 0))
    final.paste(plot, (0, img.height))
    return final
