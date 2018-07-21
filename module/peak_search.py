#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# peak search from 2d data
###############################################################################

import sys
import re
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import lmfit
import numpy as np


epsilon = 1e-8


class PeakSearch():
    """
    deals with phonon scattering experimental data
    """
    def __init__(self, x=None, y=None, name=None):
        """
        input       : x; np.array
                      y; np.array
        """
        # set self
        self.x = x
        self.y = y
        self.name = name
        self.ix_peaks = None  # ex) [36, 62, 97]
        self.ix_peak_pairs = None  # ex) [[36, 97], [...], ...]

    def find_peak(self, order, verbose=True):
        """
        input       : order; int
                      notice; bool => True (default)
        output      : np.array => argrelmax_arr
        description : find peak from data_lst
                      you can change the value of 'order', which is parameter
                      of definition 'scipy.signal.argrelmax'
                      see more about 'scipy.signal.argrelmax'
                      http://jinpei0908.hatenablog.com/entry/2016/11/26/224216
        """
        extrema = argrelmax(self.y, order=order)
        self.ix_peaks = list(extrema[0])
        if verbose:
            print("found %s peaks" % len(self.ix_peaks))

    def add_peak(self, idx, run_mode='test'):
        """
        input       : run_mode; str => 'test' or 'add'
                      idx; int => new index add to self.ix_peaks
        """
        if run_mode == 'test':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self.plot(ax)
            ax.scatter(self.x[idx], self.y[idx],
                       marker="*", c='blue', s=100)
            ax.set_title(self.name)
            plt.show()
            plt.close()

        elif run_mode == 'add':
            if idx in self.ix_peaks:
                raise ValueError("index %s is already in the peak_idx_lst !")

            self.ix_peaks.append(idx)
            self.ix_peaks.sort()

    def remove_peak(self, idx):
        """
        input       : idx; int => remove peak index
        """
        self.ix_peaks.remove(idx)

    def find_peak_pair(self, threshold=6, verbose=True):
        """
        input       : threshold; float or int => threshold=6 (default)
        description : recognize A and B as pair if abs(A) - abs(B) < threshold
        set         : self.peak_pair_idx_lst
        """
        if self.ix_peaks is None:
            raise ValueError("You have to execute find_peak ahead !")

        # condition => stokes anti-stokes
        pairs = []
        flags = []
        for i in range(int(len(self.ix_peaks)/2)+1):
            for j in range(len(self.ix_peaks)-1, i, -1):
                mean = (self.x[self.ix_peaks[i]]
                        + self.x[self.ix_peaks[j]])
                if abs(mean) < threshold \
                        and i not in flags and j not in flags:
                    pairs.append(
                      [self.ix_peaks[i], self.ix_peaks[j]])
                    flags.extend([i, j])

        self.ix_peak_pairs = pairs

        if verbose:
            print("found %s pair" % str(len(self.ix_peak_pairs)))

    def revise_peak_pair(self, ix_peak_pairs, run_mode='test'):
        """
        input       : run_mode; str => 'test' or 'revise'
        """
        if run_mode == 'test':
            self.ix_peak_pairs = ix_peak_pairs
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self.plot(ax)
            plt.title(self.name)
            plt.show()
            plt.close()

        elif run_mode == 'revise':
            self.ix_peak_pairs = ix_peak_pairs

        else:
            raise ValueError("run_mode must be 'test' or 'revise'")

    def plot(self, ax, run_mode=None):
        """
        input         : ax;  ex) ax = fig.add_subplot(111)
                        run_mode; str => 'raw_data', 'peak'
        """
        # raw data
        ax.scatter(self.x, self.y, c='red', s=2)
        ax.set_title(self.name)

        if run_mode == 'raw_data':
            return

        # find peak
        if self.ix_peaks is not None:
            if self.ix_peak_pairs is None:
                c_lst = ['black' for _ in range(len(self.ix_peaks))]
            else:
                c_lst = ['black' for _ in range(len(self.ix_peaks))]
                color_lst = ['green', 'yellow', 'pink', 'purple']
                for i in range(len(self.ix_peak_pairs)):
                    for j in self.ix_peak_pairs[i]:
                        c_lst[self.ix_peaks.index(j)] = color_lst[i]

            ax.scatter(self.x[self.ix_peaks],
                       self.y[self.ix_peaks],
                       c=c_lst, s=30)

        if run_mode == 'peak':
            return

        # smoothing
        # if self.func_info_lst[0]['params']['amplitude'] is not None:
        #     def tot_func(x):
        #         return sum(getattr(lmfit.lineshapes, func_info['function'])
        #                    (x, **func_info['params'])
        #                    for func_info in self.func_info_lst)

        #     curve_x_arr = np.linspace(min(self.x), max(self.x), 200)
        #     ax.plot(curve_x_arr, [tot_func[x] for x in curve_x_arr],
        #             c='blue', linewidth=1., linestyle='--')
