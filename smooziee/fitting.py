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

        def read_peaksearch(peaksearch):
            """
            read from peaksearch of a file dumped by joblib
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
        models = [self._model(i, peak_func)
                  for i, peak_func in enumerate(peak_funcs)]
        model = models[0]
        for each_model in models[1:]:
            model += each_model

        # params = self.model.make_params()

        # set attributes
        self.peaksearch = self.read_peaksearch(peaksearch)
        self.model = sum(models[1:], models[0])
        self.models = models
        self.params = self.model.make_params()
        self.result = None

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
        ex. i_peak=1, param_name='sigma' -> 'g1_sigma'
        """
        r = re.compile('^[a-zA-Z]+%d_%s' % (i_peak, param_name))
        match_names = [mpn for mpn in self.model.param_names if r.match(mpn)]
        if len(match_names) != 1:
            raise ValueError("'%s_%s' match %s"
                             % (i_peak, param_name, match_names))
        return match_names[0]

    def set_params_expr(self, ix_peaks, ix_peakpairs, param_names):
        """
        Inputs
        ------
        ix_peaks: list of int (must be sorted)
            ex. [36, 62, 97]
        ix_peakpairs: list of list (must be sorted)
            ex. [[36, 97], [...], ...]
        param_names: list of str
            ex. ['amplitude', 'center']
        """
        def pair_i_peak(ix_peak):
            pair_i_peak = None
            for ix_peakpair in ix_peakpairs:
                if ix_peakpair[1] == ix_peak:
                    pair_i_peak = ix_peaks.index(ix_peakpair[0])
            return pair_i_peak

        for param_name in param_names:
            for i, ix_peak in enumerate(ix_peaks):
                if pair_i_peak(ix_peak) is not None:
                    self.params[self._param_name(i, param_name)].set(
                        expr=self._param_name(pair_i_peak(ix_peak),
                                              param_name))

    def set_params_vary(self, i_peaks, param_names, vary):
        """
        i_peaks: list of int
            index of peaks to fix  ex.[2, 9]
        param_names: list of str
            ex. ['amplitude', 'center']
        vary: bool

        """
        for i_peak in i_peaks:
            for param_name in param_names:
                self.params[self._param_name(i_peak, param_name)
                            ].set(vary=vary)

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
