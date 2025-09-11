import param
import pytest
from spb.doc_utils.ipython import (
    MyParamPager,
    get_public_methods,
    split_docstring,
    _get_parameters_dict,
    modify_graphics_doc
)
from spb.series import LineOver1DRangeSeries
from spb import plot, line


class A:
    a_class_attr = False

    def __init__(self):
        self._property_1 = 0
        self._property_2 = 0

    def method_1(self, a, b):
        return a + b

    def _method_2(self):
        pass

    @property
    def property_1(self):
        return self._property_1

    @property
    def property_2(self):
        return self._property_2

    @property_2.setter
    def property_2(self, val):
        self._property_2 = val


class B(param.Parameterized):
    first = param.Selector(
        default="a",
        doc="first docstring",
        objects={
            "First opt": "a",
            "Second opt": "b",
            "Third opt": "c",
            "Fourth opt": 4,
        })
    second = param.Selector(
        default="a",
        doc="second docstring",
        objects=["a", "b", "c", 4]
    )
    third = param.Selector(
        default=1,
        doc="third docstring",
        objects=[0, 1, 2, 3]
    )
    fourth = param.Integer(
        default=4, doc="fourth docstring")
    fifth = param.Integer(
        default=5, doc="fifth docstring", bounds=(3, 7))
    sixth = param.Integer(
        default=6, doc="sixth docstring", bounds=(3, None))
    a = param.Number(
        default=3.25, doc="docstring a",
        bounds=(3, 5), inclusive_bounds=(True, False))
    b = param.Number(
        default=3.5, doc="docstring b",
        bounds=(3, 5), inclusive_bounds=(False, True))
    c = param.Number(
        default=3.75, doc="docstring c",
        bounds=(3, 5), inclusive_bounds=(False, False))
    d = param.Number(
        default=4.25, doc="docstring d",
        bounds=(3, 5), inclusive_bounds=(True, True))

    def method_1(self, a, b):
        return a + b

    @param.depends("first", "second", watch=True)
    def method_2():
        pass

    def _method_3(self):
        pass


class C_process_docstring:
    """
    Note the new line character after the triple quote. This is mandatory
    for code to process docstring appropriately. Failing to include it
    would make split_docstring to fail, without raising any errors.

    Parameters
    ---------
    p1 : float
        This is a test
    p2 : bool
        Another parameter

    Notes
    =====
    Some random note.

    Methods
    -------

    method_1()
    method_2(*args)
    """
    pass


@pytest.mark.parametrize("cls, expected", [
    (A, ["method_1"]),
    (B, ["method_1", "method_2"])
])
def test_get_public_methods(cls, expected):
    methods = get_public_methods(cls)
    assert set(methods) == set(expected)


def test_MyParamPager():
    param_pager = MyParamPager(metaclass=True)
    docstring = param_pager(B)

    assert "first docstring" in docstring
    assert f"* 'a': First opt" in docstring
    assert f"* 'b': Second opt" in docstring
    assert f"* 'c': Third opt" in docstring
    assert f"* 4: Fourth opt" in docstring
    assert "Default value: 'a'" in docstring

    assert "second docstring" in docstring
    assert f"Possible options: ['a', 'b', 'c', 4]" in docstring

    assert "third docstring" in docstring
    assert f"Possible options: [0, 1, 2, 3]" in docstring
    assert "Default value: 1" in docstring

    assert "fourth docstring" in docstring
    assert "Default value: 4" in docstring

    assert "fifth docstring" in docstring
    assert "It must be: 3 ≤ fifth ≤ 7" in docstring
    assert "Default value: 5" in docstring

    assert "sixth docstring" in docstring
    assert "It must be: 3 ≤ sixth < ∞" in docstring
    assert "Default value: 6" in docstring

    assert "docstring a" in docstring
    assert "It must be: 3 ≤ a < 5" in docstring
    assert "Default value: 3.25" in docstring

    assert "docstring b" in docstring
    assert "It must be: 3 < b ≤ 5" in docstring
    assert "Default value: 3.5" in docstring

    assert "docstring c" in docstring
    assert "It must be: 3 < c < 5" in docstring
    assert "Default value: 3.75" in docstring

    assert "docstring d" in docstring
    assert "It must be: 3 ≤ d ≤ 5" in docstring
    assert "Default value: 4.25" in docstring


def test_split_docstring():
    sections = split_docstring(C_process_docstring.__doc__)
    assert len(sections) == 4
    assert "general" in sections
    assert "Parameters" in sections
    assert "Notes" in sections
    assert "Methods" in sections

    assert len(sections["general"].split("\n")) == 3
    assert "p1 : float" in sections["Parameters"]
    assert "Some random note." in sections["Notes"]
    assert "method_1()" in sections["Methods"]
    assert "method_2(*args)" in sections["Methods"]


@modify_graphics_doc(LineOver1DRangeSeries)
def this_is_a_func(expr, range_x, label=""):
    """
    This is function in which I test the decorator that will aggregate
    the docstring from this function, as well as the provided data series
    in the decorator call.

    Parameters
    ----------
    my_custom_param_1 : int, optional
        This is a test docstring for my_custom_param_1.
    my_custom_param_2 : float
        This is a test docstring for my_custom_param_2.

    Returns
    -------
    It returns something.
    """
    pass


def test_modify_graphics_doc():
    doc = this_is_a_func.__doc__
    assert "expr :" in doc
    assert "range_x : tuple, Tuple" in doc
    assert "my_custom_param_1 : int, optional" in doc
    assert "This is a test docstring for my_custom_param_1." in doc
    assert "my_custom_param_2 : float" in doc
    assert "This is a test docstring for my_custom_param_2." in doc


def test_plot_functions_doc():
    # verify that the documentation contains parameters from the data series
    # and the PlotAttributes class
    doc = plot.__doc__
    assert "expr :" in doc
    assert "range_x : tuple, Tuple" in doc
    assert "aspect : str, tuple, list" in doc
    assert "xlabel :" in doc
    assert "Label of the x-axis. It can be:" in doc
    assert "xscale : str" in doc
    assert "If the backend supports it, the x-direction will use the specified" in doc
    assert "evaluator" not in doc


def test_graphics_function_doc():
    # verify that the documentation contains parameters from the data series
    doc = line.__doc__
    assert "expr :" in doc
    assert "range_x : tuple, Tuple" in doc
    assert "Discretization strategy along the x-direction." in doc
    assert "evaluator" not in doc

def test_Series_doc():
    # verify that data series classes get a modified docstring
    doc = LineOver1DRangeSeries.__doc__
    assert "Parameters" in doc
    assert "Parameters of 'LineOver1DRangeSeries'" not in doc
    assert "C/V= Constant/Variable, RO/RW = ReadOnly/ReadWrite, AN=Allow None" not in doc
    assert "_parametric_ranges" not in doc
