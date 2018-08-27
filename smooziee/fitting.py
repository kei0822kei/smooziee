#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# fit data received from peak_search.PeakSearch
###############################################################################

"""
fit data received from peak_search.PeakSearch
"""

import copy
import functools
import joblib
from operator import add
import re

import matplotlib.pyplot as plt
import lmfit

"""
### CODING NOTICE ###

# Naming
function names ... Follow "lmfit.lineshapes" function
                    ('lorentzian', 'gaussian', ...)
parameter names ... Each function's parameters which are noticed
                    in lmfit's documentation of "built-in models"
                    (for lorentzian, 'amplitude', 'center', ...)

"""

epsilon = 1e-8


def load_peaksearch(peaksearch):
    """
    load from peaksearch of a file dumped by joblib

        Parameters
        ----------
        peaksearch : peak_search.PeakSearch obj or hdf5
            PeakSearch.ix_peaks must be set.
    """
    try:
        peak_search = joblib.load(peaksearch)
    except FileNotFoundError:
        peak_search = peaksearch
    # check
    if peak_search.ix_peaks is None:
        ValueError("ix_peaks was None, couldn't find ix_peaks")
    return peak_search


def result_peaksearch(peaksearch, return_peak_num=True):
    """
    check the result of peak search

        Parameters
        ----------
        peaksearch : peak_search.PeakSearch obj or hdf5
            PeakSearch.ix_peaks must be set.
        return_peak_num : bool (default True)
            if True, return the number of peaks in peaksearch

        Returns
        -------
        peak_num : int
            the number of peaks in 'peaksearch'
            if 'return_peak_num' is True, this is returned
    """
    processor = load_peaksearch(peaksearch)
    peak_num = len(processor.ix_peaks)
    print("the number of peaks: %s" % str(peak_num))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    processor.plot(ax)
    plt.show()
    if return_peak_num:
        return peak_num


class Fitting():
    """
    fit data by this class

        Attributes
        ----------
        peaksearch : smooziee.smooziee.peak_search.PeakSearch object
            PeakSearch object including data peaks and peak pairs
        model : lmfit.model.CompositreModel object
            sum of the models such as lorentzian fitted using data peaks
        params : lmfit.parameter.Parameters object
            parameters to fit
        result : lmfit.model.ModelResult
            parameters after fitting
    """

    def __init__(self, peaksearch, peak_funcs):
        """
        set ix_peaks, ix_peakpairs and lmfit.model.

            Parameters
            ----------
            peaksearch : peak_search.PeakSearch obj or hdf5
                PeakSearch.ix_peaks must be set.

            peak_funcs : lst
                ex) ['lorentzian', 'gaussian', ....]
                1. Now, you  can set 'lorentzian' or 'gaussian'.
                2. len(lst) must be the same as len(PeakSearch.ix_peaks)
        """

        # make model function and params
        def models():
            models = [self._model(i, peak_func)
                      for i, peak_func in enumerate(peak_funcs)]
            return models

        # set attributes
        # self.peaksearch = peaksearch
        self.peaksearch = load_peaksearch(peaksearch)
        # print("the number of peak is %s"
        # % str(len(self.peaksearch.ix_peaks)))
        self.model = functools.reduce(add, models())
        self.params = self.model.make_params()
        self.result = None

        self._set_params_value()
        self._set_params_expr()
        self._set_params_min(param_name='amplitude')
        self._set_params_min(param_name='sigma')

    def _model(self, i, peak_func):
        """
        Have to set in order that self._param_name() can convert

        """
        if peak_func == 'lorentzian':
            prefix = 'l' + str(i) + '_'
            return lmfit.models.LorentzianModel(prefix=prefix)
        elif peak_func == 'gaussian':
            prefix = 'g' + str(i) + '_'
            return lmfit.models.GaussianModel(prefix=prefix)

    def _param_name(self, i_peak, param_name):
        """
        ex. i_peak=1, param_name='sigma' => 'g1_sigma'

        """
        r = re.compile('^[a-zA-Z]+%d_%s' % (i_peak, param_name))
        match_names = [mpn for mpn in self.model.param_names if r.match(mpn)]
        if len(match_names) != 1:
            raise ValueError("'%s_%s' match %s"
                             % (i_peak, param_name, match_names))
        return match_names[0]

    def _param_names(self, param_name):
        """
        ex. param_name='sigma' => ['g1_sigma', 'g2_sigma', ...]

        """
        r = re.compile('^[a-zA-Z]+[0-9]+_%s' % param_name)
        match_names = [mpn for mpn in self.model.param_names if r.match(mpn)]
        return match_names

    def _set_params_min(self, param_name, min_=epsilon):
        """
        Inputs
        ------
        param_name: str
            ex. 'amplitude'

        """
        for _param_name in self._param_names(param_name):
            self.params[_param_name].set(min=min_)

    def _set_params_value(self, param_name='center'):
        def value(ix_peak):
            return self.peaksearch.x_data[ix_peak]

        for i, ix_peak in enumerate(self.peaksearch.ix_peaks):
            self.params[self._param_name(i, param_name)].set(
                value=value(ix_peak))

    def _set_params_expr(self, param_names=['sigma']):
        """
        param_names: list of str
            ex. ['sigma']

        """
        def pair_i_peak(ix_peak):
            pair_i_peak = None
            for ix_peakpair in self.peaksearch.ix_peakpairs:
                if ix_peakpair[1] == ix_peak:
                    pair_i_peak = self.peaksearch.ix_peaks.index(
                        ix_peakpair[0])
            return pair_i_peak

        def set_expr(param_name, i, ix_peak):
            expr = self._param_name(pair_i_peak(ix_peak), param_name)
            self.params[self._param_name(i, param_name)].set(expr=expr)

        for param_name in param_names:
            for i, ix_peak in enumerate(self.peaksearch.ix_peaks):
                if pair_i_peak(ix_peak) is not None:
                    set_expr(param_name, i, ix_peak)

    def set_params_vary(self, i_peaks, param_names, vary,
                        all_param=False, onlyif_expr_isnone=None):
        """
        description of this instance

            Parameters
            ----------
            i_peaks : list of int
                index of peaks to fix  ex.[2, 9]
            param_names : int
                ex. ['center']
            vary : bool
            all_param : bool, default False
                If True, set all params' vary by 'vary'.
            onlyif_expr_isnone : bool
                Set vary only if 'expr' is None.
                Defaults to True if vary is True.

            Returns
            -------
            fruit_price : int
                description

            Notes
            -----

            Raises
            ------
            ValueError
                conditions which ValueError occurs
        """
        def _set_vary(param_name):
            if onlyif_expr_isnone:
                if self.params[param_name].expr is None:
                    self.params[param_name].set(vary=vary)
            else:
                self.params[param_name].set(vary=vary)

        if onlyif_expr_isnone is None:
            if vary:
                onlyif_expr_isnone = True

        if all_param:
            for param_name in self.params:
                _set_vary(param_name)
        else:
            for i_peak in i_peaks:
                for param_name in param_names:
                    _set_vary(self._param_name(i_peak, param_name))

    def set_params(self, i_peak, param_name, values):
        """
        set parameters to 'params' attribute

            Parameters
            ----------
            i_peak : int
                index of peak to set
            peak_param : str
                parameter name to set
            values : dict
                values to set
        """
        self.params[self._param_name(i_peak, param_name)].set(**values)

    def fit(self):
        """
        fit data using params

        Notes
        -----
        1. the result is set to 'result' attribute
        2. If you want to set the result to  new parameter,
           you have to conduct 'set_result_param_to_inital()'.
        """
        x = self.peaksearch.x_data
        y = self.peaksearch.y_data
        self.result = self.model.fit(y, self.params, x=x)

    def set_result_param_to_inital(self):
        """
        set result.params to params
        """
        self.params = self.result.params

    def plot(self, show_init=False, numpoints=1000, eval_components=False):
        """
        plot the fitting result

            Parameters
            ----------
            show_init : bool, default False
                plot fitting curve using initial parameter
            numpoints : int, default 1000
                data points to plot fitting curve
            eval_components : bool, default False
                plot each function
        """
        self._base_plot(result=self.result,
                        peaksearch=self.peaksearch,
                        show_init=show_init,
                        numpoints=numpoints,
                        eval_components=eval_components)

    def plot_from_params(self, show_init=False, numpoints=1000,
                         eval_components=False):
        """
        description of this method

            Parameters
            ----------
            show_init : bool, default False
                description
            numpoints : int, default 1000
                description
            eval_components : bool, default False
                description

            Notes
            -----

        """
        def _fix_all(params):
            for param_name in params:
                params[param_name].set(vary=False)

        params = copy.deepcopy(self.params)
        _fix_all(params)
        model = copy.deepcopy(self.model)

        result = model.fit(self.peaksearch.y_data,
                           params,
                           x=self.peaksearch.x_data)
        self._base_plot(result=result,
                        peaksearch=self.peaksearch,
                        show_init=show_init,
                        numpoints=numpoints,
                        eval_components=eval_components)

    def _base_plot(self, result, peaksearch, show_init, numpoints,
                   eval_components):
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_axes((0.1, 0.15, 0.85, 0.55))
        ax2 = fig.add_axes((0.1, 0.73, 0.85, 0.22))
        ax2.set_xticklabels([])
        ax2.set_ylabel('')

        result.plot_fit(ax=ax1,
                        show_init=show_init,
                        numpoints=numpoints)
        result.plot_residuals(ax=ax2)
        peaksearch.plot(ax1)
        ax1.set_title('')
        ax2.set_title('')

        if eval_components:
            x = result.userkws['x']
            for name, y in result.eval_components().items():
                ax1.plot(x, y, label=name)
            plt.show()
