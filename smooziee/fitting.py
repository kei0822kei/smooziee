#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# fit data received from peak_search.PeakSearch
###############################################################################

import matplotlib.pyplot as plt
import re
import joblib

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


class Fitting():

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

        def load_peaksearch(peaksearch):
            """
            load from peaksearch of a file dumped by joblib
            """
            try:
                peak_search = joblib.load(peaksearch)
            except:
                pass
            # check
            if peak_search.ix_peaks is None:
                ValueError("ix_peaks was None, couldn't find ix_peaks")
            return peak_search

        # make model function and params
        def models():
            models = [self._model(i, peak_func)
                      for i, peak_func in enumerate(peak_funcs)]
            return models

        models = models()

        # set attributes
        # self.peaksearch = peaksearch
        self.peaksearch = load_peaksearch(peaksearch)
        self.model = sum(models[1:], models[0])
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
            return self.peaksearch.x[ix_peak]

        for i, ix_peak in enumerate(self.peaksearch.ix_peaks):
            self.params[self._param_name(i, param_name)].set(
                value=value(ix_peak))

    def _set_params_expr(self, param_names=['sigma']):
        """
        Inputs
        ------
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
                        all_param=False):
        """
        i_peaks: list of int
            index of peaks to fix  ex.[2, 9]
        param_names: list of str
            ex. ['center']
        vary: bool
        all_param: bool
            If True, set all params' vary by vary

        """
        if all_param:
            for param_name in self.params:
                self.params[param_name].set(vary=vary)
        else:
            for i_peak in i_peaks:
                for param_name in param_names:
                    self.params[self._param_name(i_peak, param_name)].set(
                        vary=vary)

    def fit(self, x, y, set_bestparams=False):
        self.result = self.model.fit(y, self.params, x=x)
        if set_bestparams:
            self.params = self.result.params

    def plot(self, show_init=False):
        self.result.plot(show_init=show_init)

    def plot_evalcomponents(self):
        self.result.plot()
        x = self.result.userkws['x']
        for name, y in self.result.eval_components().items():
            plt.plot(x, y, label=name)
        plt.legend()
