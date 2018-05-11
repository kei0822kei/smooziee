#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# fitting 2d data
###############################################################################

### import modules
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.signal import argrelmax
from scipy.optimize import fmin
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from smooziee.module import function as smooziee_func


###############################################################################
# phonon scattering
###############################################################################

class Processor():
    """
    deals with phonon scattering experimental data
    """

    def __init__(self, x_arr, y_arr):
        """
        input       : x_arr; np.array
                      y_arr; np.array
        """

        ### set self
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.peak_idx_lst = None  ### ex) [36, 62, 97]
        self.peak_pair_idx_lst = None  ### ex) [[36, 97], [...], ...]
        self.best_param_lst = None  ### [[initA_0, initx0_0, initd_0], ...]
        self.function = None


    def find_peak(self, order, notice=True):
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
        argrelmax_return_tuple = argrelmax(self.y_arr, order=order)
        self.peak_idx_lst = list(argrelmax_return_tuple[0])
        if notice:
            print("found %s peaks" % len(self.peak_idx_lst))


    def add_peak(self, idx, run_mode='test'):
        """
        input       : run_mode; str => 'test' or 'add'
        """
        if run_mode == 'test':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self.plot(ax, run_mode='raw_data')
            ax.scatter(self.x_arr[idx,0], self.y_arr[idx,1], marker="*", c='blue', s=100)
            plt.show()

        elif run_mode == 'add':
            if idx in self.peak_idx_lst:
                print("index %s is already in the peak_idx_lst !")
                return

            self.peak_idx_lst.append(idx)


    # def find_peak_pair(self, threshold=6, notice=True):
    #     """
    #     input       : threshold; float or int => threshold=6 (default)
    #     description : recognize A and B as pair if abs(A) - abs(B) < threshold
    #     set         : self.peak_pair_idx_lst
    #     """
    #     if self.peak_idx_lst is None:
    #         print("You have to execute find_peak ahead !")
    #         sys.exit(1)

    #     ### condition => stokes anti-stokes
    #     pair_lst = []
    #     flag_lst = []
    #     for i in range(int(len(self.peak_idx_lst)/2)+1):
    #         #for j in range(len(peak_idx_lst)-1, i, -1):
    #         for j in range(len(self.peak_idx_lst)-1, i, -1):
    #             mean = self.data_df.loc[self.peak_idx_lst[i]]['meV'] + \
    #                        self.data_df.loc[self.peak_idx_lst[j]]['meV']
    #             if abs(mean) < threshold and i not in flag_lst and j not in flag_lst:
    #                 pair_lst.append([self.peak_idx_lst[i], self.peak_idx_lst[j]])
    #                 flag_lst.extend([i, j])

    #     self.peak_pair_idx_lst = pair_lst

    #     if notice:
    #         print("found %s pair" % str(len(self.peak_pair_idx_lst)))


    def save(self, savefile):
        """
        input         : savefile
        description   : save variables
        """
        outfh = h5py.File(savefile, 'w')
        outfh.create_dataset('x_arr', data = self.x_arr)
        outfh.create_dataset('y_arr', data = self.y_arr)
        outfh.create_dataset('peak_idx_lst', data = self.peak_idx_lst)
        outfh.create_dataset('peak_pair_idx_lst', data = self.peak_pair_idx_lst)
        outfh.create_dataset('best_param_lst', data = self.best_param_lst)
        outfh.flush()
        outfh.close()


    def load(self, loadfile):
        infh = h5py.File(loadfile, 'r')

        self.x_arr = np.array(list(infh['x_arr'].value))
        self.y_arr = np.array(list(infh['y_arr'].value))
        self.peak_idx_lst = list(infh['peak_idx_lst'].value)
        self.peak_pair_idx_lst = list(map(list, infh['peak_pair_idx_lst'].value))
        self.best_param_lst = list(map(list, infh['best_param_lst'].value))
        infh.close()

    def plot(self, ax, run_mode=None):
        """
        input         : ax;  ex) ax = fig.add_subplot(111)
                        run_mode; str => 'raw_data', 'peak'
        """
        ### raw data
        ax.scatter(self.x_arr, self.y_arr, c='red', s=2)

        if run_mode == 'raw_data':
            return

        ### find peak
        if self.peak_idx_lst is not None:
            if self.peak_pair_idx_lst is None:
                c_lst = [ 'black' for _ in range(len(self.peak_idx_lst)) ]
            else:
                c_lst = [ 'black' for _ in range(len(self.peak_idx_lst)) ]
                color_lst = ['green', 'yellow', 'pink']
                for i in range(len(self.peak_pair_idx_lst)):
                    for j in self.peak_pair_idx_lst[i]:
                        c_lst[self.peak_idx_lst.index(j)] = color_lst[i]

            ax.scatter(self.x_arr[self.peak_idx_lst], \
                       self.y_arr[self.peak_idx_lst],
                       c=c_lst, s=30)

        if run_mode == 'peak':
            return

        ### smoothing
        if self.best_param_lst is not None:
            curve_x_arr = np.linspace(
                min(self.x_arr), max(self.x_arr), 200)
            curve_y_arr = 0
            for param in self.best_param_lst:
                curve_y_arr += smooziee_func.lorentzian(curve_x_arr,
                                   [param[0], param[1], param[2]]
                               )
            ax.plot(curve_x_arr, curve_y_arr, c='blue', linewidth=1.,
                    linestyle='--')

            for param in self.best_param_lst:
                curve_y_arr = smooziee_func.lorentzian(curve_x_arr,
                                  [param[0], param[1], param[2]])
                ax.plot(curve_x_arr, curve_y_arr, c='blue', linewidth=0.3,
                        linestyle='--')


    def set_function(self):

        def fitting_function(input_param_lst):
            all_param_lst = list(map(list, np.reshape(np.array(input_param_lst), (int(len(input_param_lst)/3), 3))))
            y_fit_arr = 0
            for param_lst in all_param_lst:
                y_fit_arr += smooziee_func.lorentzian(self.x_arr, param_lst)
            error = np.sqrt(mean_squared_error(self.y_arr, y_fit_arr))

            x = np.linspace(min(self.x_arr), max(self.x_arr), 200)
            y = 0
            for param_lst in all_param_lst:
                y += smooziee_func.lorentzian(y, param_lst)

            return error

        self.function = fitting_function


    def fit(self):
        if self.best_param_lst == None:
            param_arr = np.random.rand(len(self.peak_idx_lst) * 3) * 3
            for i in range(int(len(self.peak_idx_lst))):
                param_arr[3*i+1] = self.x_arr[self.peak_idx_lst[i]]
            init_param_lst = list(param_arr)

        else:
            init_param_lst = list(np.array(self.best_param_lst).flatten())

        [xopt, fopt, itera, funcalls, warnflag, allvecs] \
            = fmin(self.function,init_param_lst, retall=True, full_output=True)

        all_param_lst = list(map(list, np.reshape(np.array(xopt), (int(len(xopt)/3), 3))))
        self.best_param_lst = all_param_lst
