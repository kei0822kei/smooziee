#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# fit data received from peak_search.PeakSearch
###############################################################################

"""
fit data received from peak_search.PeakSearch

### CODING NOTICE ###

# Naming
function names ... Follow "lmfit.lineshapes" function
                    ('lorentzian', 'gaussian', ...)
"""

import copy
import functools
import joblib
import numpy as np
from operator import add
import re
import os
import json

import matplotlib.pyplot as plt
import lmfit

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
    except:
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


def load_json(jsonfile):
    """
    return object written in 'jsonfile'

        Parameters
        ----------
        jsonfile : str
            json file

        Returns
        -------
        obj : various
            object written in 'jsonfile'
    """
    with open(jsonfile) as f:
        obj = json.load(f)
    return obj


def load_fitting(dirname):
    """
    load Fitting class object from 'dirname'

        Parameters
        ----------
        dirname : str
            saved directory name using 'Fitting.save' method

        Returns
        -------
        fitter : Fitting class obj
            loaded Fitting class object
    """
    peaksearch = load_peaksearch(os.path.join(dirname, 'peaksearch.pkl'))
    peak_funcs = load_json(os.path.join(dirname, 'peak_funcs.json'))
    lm_parameter = lmfit.Parameters()
    with open(os.path.join(dirname, 'params.json')) as f:
        params = lm_parameter.load(f)
    fitter = Fitting(peaksearch, peak_funcs)
    fitter.params = params
    return fitter


class Fitting():
    """
    fit data by this class

        Attributes
        ----------
        peaksearch : smooziee.peak_search.PeakSearch object
            PeakSearch object including data peaks and peak pairs
        i_peakpairs : list of list of int
            list of [i j], which means i'th and j'th peaks are pair.
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

            Nones
            -----
            _set_params_value() => set x param (read from peaksearch object)
            _set_params_expr() => set expr (read from peaksearch object)
            _set_params_min(param_name='amplitude') => set mininum amplitude
            _set_params_min(param_name='sigma') => set mininum sigma

            at first, the values of 'amplitude' and 'sigma' are random valuas
        """

        # make model function and params
        def models():
            models = [self._model(i, peak_func)
                      for i, peak_func in enumerate(peak_funcs)]
            return models

        # set attributes
        # self.peaksearch = peaksearch
        self.peaksearch = load_peaksearch(peaksearch)
        self.i_peakpairs = self._i_peakpairs()
        # print("the number of peak is %s"
        # % str(len(self.peaksearch.ix_peaks)))
        self.peak_funcs = peak_funcs
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

    def _i_peakpairs(self):
        def _iter_i_peakpairs():
            for ix_peakpair in self.peaksearch.ix_peakpairs:
                assert len(ix_peakpair) == 2
                yield [self.peaksearch.ix_peaks.index(ix_peakpair[i])
                       for i in range(2)]
        return list(_iter_i_peakpairs())

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
        def _pair_i_peak(i_peak):
            pair_i_peak = None
            for i_peakpair in self.i_peakpairs:
                if i_peakpair[1] == i_peak:
                    pair_i_peak = i_peakpair[0]
            return pair_i_peak

        def set_expr(param_name, i_peak, pair_i_peak):
            expr = self._param_name(pair_i_peak, param_name)
            self.params[self._param_name(i_peak, param_name)].set(expr=expr)

        for i_peak in range(len(self.peaksearch.ix_peaks)):
            for param_name in param_names:
                pair_i_peak = _pair_i_peak(i_peak)
                if pair_i_peak is not None:
                    set_expr(param_name, i_peak, pair_i_peak)

    def set_params_vary(self, i_peaks, param_names, vary):
        """
        Set vary only if expr is None.

            Parameters
            ----------
            i_peaks : list of int
                index of peaks to fix  ex.[2, 9]
            param_names : list of str
                ex. ['center']
            vary : bool

            Raises
            ------
            ValueError
                conditions which ValueError occurs
        """
        def _set_vary(param_name):
            if self.params[param_name].expr is None:
                self.params[param_name].set(vary=vary)

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
                The final and initial fit curves are evaluated not only at
                data points, but refined to contain numpoints points in total.
            eval_components : bool, default False
                Whether to show the each component plots.
        """
        self._base_plot(result=self.result,
                        peaksearch=self.peaksearch,
                        show_init=show_init,
                        numpoints=numpoints,
                        eval_components=eval_components)

    def plot_from_params(self, numpoints=1000,
                         eval_components=False, filename=None):
        """
        Plot current fitting result from self.params.
        This method doesn't change self.result and self.model.

            Parameters
            ----------
            numpoints : int, default 1000
                The final and initial fit curves are evaluated not only at
                data points, but refined to contain numpoints points in total.
            eval_components : bool, default False
                Whether to show the each component plots.
            filename : str, default None
                If 'filename' is not 'None',
                figure is saved whose name is 'filename'.
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
                        show_init=False,
                        numpoints=numpoints,
                        eval_components=eval_components,
                        filename=filename)

    def _base_plot(self, result, peaksearch, show_init, numpoints,
                   eval_components, filename=None):
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

        x_array = result.userkws['x']
        x_array_dense = np.linspace(min(x_array), max(x_array), numpoints)

        if eval_components:
            for name, y in result.eval_components(
                    **{'x': x_array_dense}).items():
                ax1.plot(x_array_dense, y, label=name)

        ax1.legend()
        if filename is not None:
            fig.savefig(filename)
        plt.show()

    def save(self, dirname):
        """
        description of this method

            Parameters
            ----------
            dirname : str
                save directory name
        """
        os.mkdir(dirname)
        self.peaksearch.save(os.path.join(dirname, "peaksearch.pkl"))
        with open(os.path.join(dirname, "params.json"), 'w') as f:
            self.params.dump(f)
        with open(os.path.join(dirname, "peak_funcs.json"), 'w') as f:
            json.dump(self.peak_funcs, f)

    def output(self, gpifile, i_center, header='result'):
        """
        output the results

            Parameters
            ----------
            paramname1 : int
                description
            paramname2 : int, default var
                description Returns
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
        def _center_shift(i_center):
            """
            return shift value
            """
            center_names = self._param_names('center')
            shift = None
            for peakpair in self.i_peakpairs:
                if i_center in peakpair:
                    center1 = self.params[center_names[peakpair[0]]].value
                    center2 = self.params[center_names[peakpair[1]]].value
                    shift = ((center1 + center2) / 2) * (-1)
                    center_peak = peakpair
                    print("set center using peak pair : %s" % str(peakpair))
                    break
            if shift is None:
                shift = self.params[center_names[i_center]].value * (-1)
                center_peak = i_center
                print("set center using peak : %s" % str(i_center))
            print("shift is %s" % str(shift))
            return shift, center_peak

        def _read_qpoint(gpifile):
            """
            return qpoint
            """
            from smooziee import gpi
            tf_num = 'tf_'+self.peaksearch.name[-1]
            gpi_reader = gpi.GPI_reader(gpifile)
            qp = gpi_reader.qpoint(tf_num)
            return qp

        def _get_params():
            """
            return parameters
            """
            param_names = list(self.params.keys())
            prefixes = set()
            for name in param_names:
                prefixes.add(name[:2])
            prefixes = list(prefixes)
            params = {}
            for prefix in prefixes:
                params[prefix] = {}
                prefix_params = [ names for names in param_names if prefix in names ]
                for name in prefix_params:
                    params[prefix][name[3:]] = self.params[name].value
            return params

        def _param_shift(params, shift):
            """
            return shifted parameters
            """
            for key in params.keys():
                params[key]['center'] = params[key]['center'] + shift
            return params

        results = {}
        shift, center_peak = _center_shift(i_center)
        results['center_shift'] = {'shift':shift, 'center_peak':center_peak}
        results['qpoint'] = _read_qpoint(gpifile)
        results['params'] = _get_params()
        results['shifted_params'] = _param_shift(results['params'], shift)
        joblib.dump(results, header+'.pkl')
        self.plot_from_params(eval_components=True, filename=header+'.png')
        self.save(header+'_dump')
