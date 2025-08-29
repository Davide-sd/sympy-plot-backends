import pytest
from spb.defaults import cfg
from spb import MB
from sympy import latex


@pytest.fixture
def label_func():
    def wrapper(use_latex, expr):
        if use_latex:
            return "$%s$" % latex(expr)
        return str(expr)
    return wrapper


@pytest.fixture
def p_options():
    return dict(show=False, backend=MB)


@pytest.fixture
def paf_options(p_options):
    # paf = plot adaptive false
    options = p_options.copy()
    options["n"] = 100
    return options


@pytest.fixture
def pi_options(paf_options, panel_options):
    # pi = plot interactive
    options = paf_options.copy()
    options["imodule"] = panel_options["imodule"]
    return options


@pytest.fixture
def ipywidgets_options():
    return dict(show=False, imodule="ipywidgets")


@pytest.fixture
def panel_options():
    return dict(show=False, imodule="panel")


@pytest.fixture
def default_range():
    return lambda s: (s, cfg["plot_range"]["min"], cfg["plot_range"]["max"])


@pytest.fixture
def default_complex_range():
    _min = cfg["plot_range"]["min"]
    _max = cfg["plot_range"]["max"]
    return lambda s: (s, _min + _min * 1j, _max + _max * 1j)
