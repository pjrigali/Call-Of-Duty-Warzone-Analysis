from typing import Union, List
import numpy as np
from statsmodels import regression
from statsmodels.tools import add_constant
from Classes.documnent_filter import DocumentFilter


class Regression:
    """
    Calculate a linear regression.

    Parameters
    ----------
    doc_filter : DocumentFilter
        Input DocumentFilter.
    x_column : str, or List[str]
        Name of column or columns to be used in regression analysis
    y_column : str
        Name of column to be used as y variable in regression.

    Examples
    --------

    >>> from Classes.document_filter import DocumentFilter
    >>> from Classes.regression import Regression
    >>> doc = DocumentFilter(original_df=cod.our_df, map_choice='mp_e', mode_choice='quad')
    >>> model = Regression(doc_filter=doc, x_column='kills', y_column='placementPercent')

    This will return a Regression object with regression resault information.
    """

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
        """Returns R Squared"""
        return self._r2

    @property
    def constant_coefficient(self):
        """Returns Constant Coefficient, if only one x_column is provided"""
        return self._constant_coef

    @property
    def x_coefficient(self):
        """Returns X Coefficient, if only one x_column is provided"""
        return self._item_coef

    @property
    def lower_confidence(self):
        """Returns Lower Confidence Value, if only one x_column is provided"""
        return self._lower_conf

    @property
    def upper_confidence(self):
        """Returns Upper Confidence Value, if only one x_column is provided"""
        return self._upper_conf

    @property
    def pvalue(self):
        """Returns P Value or Values"""
        return self._pvalue

    @property
    def residuals(self):
        """Returns residuals"""
        return self._resid

    @property
    def mse(self):
        """Returns Mean Squared Error"""
        return self._mse

    @property
    def ssr(self):
        """Returns Sum of Squared Residuals"""
        return self._ssr

    @property
    def ess(self):
        """Returns Sum of Squared Error"""
        return self._ess

    @property
    def confidence(self):
        """Returns Confidence Values, if more than one x_column is provided"""
        return self._coefficients

    @property
    def coefficients(self):
        """Returns Coefficient Values, if more than one x_column is provided"""
        return self._confidence_bounds
