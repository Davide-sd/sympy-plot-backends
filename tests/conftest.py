import pytest
from spb.defaults import cfg
from spb import MB


@pytest.fixture
def p_options():
    return dict(show=False, backend=MB)


@pytest.fixture
def paf_options(p_options):
    # paf = plot adaptive false
    options = p_options.copy()
    options["adaptive"] = False
    options["n"] = 100
    return options


@pytest.fixture
def pat_options(p_options):
    # pat = plot adaptive true
    options = p_options.copy()
    options["adaptive"] = True
    options["adaptive_goal"] = 0.05
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
