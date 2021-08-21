from typing import Union, List
import numpy as np
from statsmodels import regression
from statsmodels.tools import add_constant
from Classes.documnent_filter import DocumentFilter


class Regression:

    doc_filter: DocumentFilter
    x_column: Union[str, List[str]]
    y_column: str

    def __init__(self, doc_filter, x_column, y_column):

        if type(x_column) == str:
            x = add_constant(np.array(doc_filter.df[x_column], dtype=float))
        else:
            x = np.array(doc_filter.df[x_column], dtype=float)

        y = np.array(doc_filter.df[y_column])
        model = regression.linear_model.OLS(y, x).fit()

        if type(x_column) == str:
            self._constant_coef = model.params[0]
            self._item_coef = model.params[1]
            self._coefficients = None
            self._lower_conf = model.conf_int()[1, :2][0]
            self._upper_conf = model.conf_int()[1, :2][1]
            self._pvalue = model.pvalues[1]
        else:
            self._constant_coef = None
            self._item_coef = None
            self._coefficients = model.params
            self._confidence_bounds = model.conf_int()
            self._lower_conf = None
            self._upper_conf = None
            self._pvalue = model.pvalues

        self._r2 = model.rsquared
        self._resid = model.resid
        self._bse = model.bse
        self._mse = model.mse_model
        self._ssr = model.ssr
        self._ess = model.ess

    def __repr__(self):
        return self._r2

    @property
    def r2(self):
        return self._r2

    @property
    def constant_coefficient(self):
        return self._constant_coef

    @property
    def x_coefficient(self):
        return self._item_coef

    @property
    def lower_confidence(self):
        return self._lower_conf

    @property
    def upper_confidence(self):
        return self._upper_conf

    @property
    def pvalue(self):
        return self._pvalue

    @property
    def residuals(self):
        return self._resid

    @property
    def mse(self):
        return self._mse

    @property
    def ssr(self):
        return self._ssr

    @property
    def ess(self):
        return self._ess

    @property
    def confidence(self):
        return self._coefficients

    @property
    def coefficients(self):
        return self._confidence_bounds
