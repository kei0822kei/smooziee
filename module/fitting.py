#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# fitting 2d data
###############################################################################
# print hello

### import modules
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.signal import argrelmax
from scipy.optimize import fmin
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from smooziee.module import function as smooziee_func
import lmfit


###############################################################################
# phonon scattering
###############################################################################

class Processor(lmfit.Parameters):
    """
    deals with phonon scattering experimental data
    """

    def __init__(self, x_arr=None, y_arr=None, name=None):
        """
        input       : x_arr; np.array
                      y_arr; np.array
        """

        ### inheritance
        super().__init__()

        ### set self
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.name = name
        self.peak_idx_lst = None  ### ex) [36, 62, 97]
        self.peak_pair_idx_lst = None  ### ex) [[36, 97], [...], ...]
        self.best_param_lst = None  ### [[initA_0, initx0_0, initd_0], ...]
        self.revised_best_param_lst = None
        self.center_move = None
        self.function = None
        self.center_peak = None  ### ex) 62 or [36, 97]
        self.function_name_lst = None
        self.func_info_lst = None


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
            self.plot(ax)
            ax.scatter(self.x_arr[idx], self.y_arr[idx],
                       marker="*", c='blue', s=100)
            ax.set_title(self.name)
            plt.show()
            plt.close()

        elif run_mode == 'add':
            if idx in self.peak_idx_lst:
                print("index %s is already in the peak_idx_lst !")
                return

            self.peak_idx_lst.append(idx)
            self.peak_idx_lst.sort()


    def remove_peak(self, idx):
        """
        input       : idx; int => remove peak index
        """
        self.peak_idx_lst.remove(idx)
        self.best_param_lst = None
        self.revised_best_param_lst = None
        self.center_move = None
        self.function = None
        self.center_peak = None
        print("self.best_param_lst were set None")


    def find_peak_pair(self, threshold=6, notice=True):
        """
        input       : threshold; float or int => threshold=6 (default)
        description : recognize A and B as pair if abs(A) - abs(B) < threshold
        set         : self.peak_pair_idx_lst
        """
        if self.peak_idx_lst is None:
            print("You have to execute find_peak ahead !")
            sys.exit(1)

        ### condition => stokes anti-stokes
        pair_lst = []
        flag_lst = []
        for i in range(int(len(self.peak_idx_lst)/2)+1):
            #for j in range(len(peak_idx_lst)-1, i, -1):
            for j in range(len(self.peak_idx_lst)-1, i, -1):
                mean = self.x_arr[self.peak_idx_lst[i]] + \
                         self.x_arr[self.peak_idx_lst[j]]
                if abs(mean) < threshold \
                        and i not in flag_lst and j not in flag_lst:
                    pair_lst.append(
                      [self.peak_idx_lst[i], self.peak_idx_lst[j]])
                    flag_lst.extend([i, j])

        self.peak_pair_idx_lst = pair_lst

        if notice:
            print("found %s pair" % str(len(self.peak_pair_idx_lst)))


    def revise_peak_pair(self, peak_pair_lst, run_mode='test'):
        """
        input       : run_mode; str => 'test' or 'revise'
        """
        if run_mode == 'test':
            temp_lst = self.peak_pair_idx_lst
            self.peak_pair_idx_lst = peak_pair_lst
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self.plot(ax)
            plt.title(self.name)
            plt.show()
            plt.close()

        elif run_mode == 'revise':
            self.peak_pair_idx_lst = peak_pair_lst

        else:
            print("run_mode must be 'test' or 'revise'")
            sys.exit(1)


    def set_function_info(self, func_name_lst):
        """
        set the type of function
        ex)["Gaussian", "Lonentz"]
        """
        self.function_name_lst = func_name_lst
        if len(func_name_lst) != len(self.peak_idx_lst):
            print("The number of peaks and functions must be the same")
            sys.exit(1)
        

        func_info_lst = []
        for func in self.function_name_lst:
            if func == "lorentzian":
                each_info = {"function" : func, 
                            "params" : {"A" : None, "myu" : None, "sigma" : None},
                            "optimize" : {"A" : True, "myu" : True, "sigma" : None}}
            func_info_lst.append(each_info)

        self.func_info_lst = func_info_lst







    # def save(self, savefile=None):
    #     """
    #     input         : savefile
    #     description   : save variables
    #     """
    #     if savefile == None:
    #         if name == None:
    #             print("Please set savefile name.")
    #             sys.exit(1)
    #         else:
    #             savefile = name

    #     outfh = h5py.File(savefile+'.hdf5', 'w')
    #     outfh.create_dataset('x_arr', data = self.x_arr)
    #     outfh.create_dataset('y_arr', data = self.y_arr)
    #     outfh.create_dataset('peak_idx_lst', data = self.peak_idx_lst)
    #     outfh.create_dataset('peak_pair_idx_lst',
    #                            data = self.peak_pair_idx_lst)
    #     outfh.create_dataset('best_param_lst', data = self.best_param_lst)
    #     try:
    #         outfh.create_dataset('revised_best_param_lst',
    #                                data = self.revised_best_param_lst)
    #         outfh.create_dataset('center_move', data = self.center_move)
    #         outfh.create_dataset('center_peak', data = self.center_peak)
    #     except:
    #         pass
    #     outfh.flush()
    #     outfh.close()

    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111)
    #     # self.plot(ax)
    #     # plt.savefig(self.name+'.png')


    # def load(self, loadfile):
    #     infh = h5py.File(loadfile, 'r')

    #     self.x_arr = np.array(list(infh['x_arr'].value))
    #     self.y_arr = np.array(list(infh['y_arr'].value))
    #     self.peak_idx_lst = list(infh['peak_idx_lst'].value)
    #     self.peak_pair_idx_lst = \
    #         list(map(list, infh['peak_pair_idx_lst'].value))
    #     self.best_param_lst = \
    #         list(map(list, infh['best_param_lst'].value))
    #     try:
    #         self.revised_best_param_lst = \
    #             list(map(list, infh['revised_best_param_lst'].value))
    #         self.center_move = int(infh['center_move'].value)
    #         try:
    #             self.center_peak = int(infh['center_peak'].value)
    #         except:
    #             self.center_peak = list(infh['center_peak'].value)
    #     except:
    #         pass
    #     infh.close()


    def plot(self, ax, run_mode=None):
        """
        input         : ax;  ex) ax = fig.add_subplot(111)
                        run_mode; str => 'raw_data', 'peak'
        """
        ### raw data
        ax.scatter(self.x_arr, self.y_arr, c='red', s=2)
        ax.set_title(self.name)

        if run_mode == 'raw_data':
            return

        ### find peak
        if self.peak_idx_lst is not None:
            if self.peak_pair_idx_lst is None:
                c_lst = [ 'black' for _ in range(len(self.peak_idx_lst)) ]
            else:
                c_lst = [ 'black' for _ in range(len(self.peak_idx_lst)) ]
                color_lst = ['green', 'yellow', 'pink', 'purple']
                for i in range(len(self.peak_pair_idx_lst)):
                    for j in self.peak_pair_idx_lst[i]:
                        c_lst[self.peak_idx_lst.index(j)] = color_lst[i]

            ax.scatter(self.x_arr[self.peak_idx_lst], \
                       self.y_arr[self.peak_idx_lst],
                       c=c_lst, s=30)

        if run_mode == 'peak':
            return

        ### smoothing
        # if self.best_param_lst is not None:
        #     curve_x_arr = np.linspace(
        #         min(self.x_arr), max(self.x_arr), 200)
        #     curve_y_arr = 0
        #     for param in self.best_param_lst:
        #         curve_y_arr += smooziee_func.lorentzian(curve_x_arr,
        #                            [param[0], param[1], param[2]]
        #                        )
        #     ax.plot(curve_x_arr, curve_y_arr, c='blue', linewidth=1.,
        #             linestyle='--')

        #     for param in self.best_param_lst:
        #         curve_y_arr = smooziee_func.lorentzian(curve_x_arr,
        #                           [param[0], param[1], param[2]])
        #         ax.plot(curve_x_arr, curve_y_arr, c='blue', linewidth=0.3,
        #                 linestyle='--')

        # ### set center
        # if self.revised_best_param_lst is not None:
        #     r_curve_y_arr = 0
        #     for param in self.revised_best_param_lst:
        #         r_curve_y_arr += smooziee_func.lorentzian(curve_x_arr,
        #                            [param[0], param[1], param[2]]
        #                        )
        #     ax.plot(curve_x_arr, r_curve_y_arr, c='orange', linewidth=1.,
        #             linestyle='--')

        #     for param in self.best_param_lst:
        #         r_curve_y_arr = smooziee_func.lorentzian(curve_x_arr,
        #                           [param[0], param[1], param[2]])
        #         ax.plot(curve_x_arr, r_curve_y_arr, c='orange', linewidth=0.3,
        #                 linestyle='--')


    # def initial_fit(self, idx_range=5, notice=True):
    #     """
    #     input       : idx_range; int => idx_range = 10 (default)
    #                       peak fit using data_arr[peak_idx-10:peak_idx+10, 0]
    #                       if idx_range = 10
    #     set         : self.best_param_lst
    #     description : make initial fit using self.peak_idx_lst
    #     """
    #     ### check
    #     if self.peak_idx_lst == None:
    #         print("You have to execute find_peak ahead!")
    #         sys.exit(1)
    #     if self.peak_pair_idx_lst == None:
    #         print("You have to execute find_peak_pair ahead!")
    #         sys.exit(1)

    #     if notice:
    #         print("make initial fitting")

    #     best_param_lst = []
    #     for peak_idx in self.peak_idx_lst:
    #         try:
    #             param_lst = curve_fit(
    #                 smooziee_func.lorentzian_for_curve_fit,
    #                 self.x_arr[peak_idx-idx_range:peak_idx+idx_range],
    #                 self.y_arr[peak_idx-idx_range:peak_idx+idx_range],
    #                 ### p0 => initial peak point
    #                 p0=[self.y_arr[peak_idx], self.x_arr[peak_idx], 1.]
    #             )
    #             best_param_lst.append( \
    #                 [param_lst[0][0], param_lst[0][1], param_lst[0][2]])
    #         except:
    #             print("index %s could not make curve_fit" % str(peak_idx))
    #             best_param_lst.append( \
    #                 [self.y_arr[peak_idx], self.x_arr[peak_idx], 1.])

    #     ### stokes anti-stokes revise param d
    #     for idx_pair_lst in self.peak_pair_idx_lst:
    #         mean_d_val = \
    #           (best_param_lst[self.peak_idx_lst.index(idx_pair_lst[0])][2] +
    #            best_param_lst[self.peak_idx_lst.index(idx_pair_lst[1])][2]) / 2
    #         best_param_lst[self.peak_idx_lst.index(idx_pair_lst[0])][2] \
    #             = mean_d_val
    #         best_param_lst[self.peak_idx_lst.index(idx_pair_lst[1])][2] \
    #             = mean_d_val

    #     self.best_param_lst = best_param_lst


    # def revise_best_param(self, revise_lst):
    #     """
    #     set         : self.best_param_lst
    #     input       : revise_lst; list
    #                     =>    [peak_idx, param_idx, val]
    #                        or [[peak_idx_1, peak_idx_2], param_idx, val]
    #                       param_idx => 0 - A  1 - x0  2 - d
    #     description : revise self.best_param_lst
    #     """
    #     ### check peak pair
    #     if type(revise_lst[0]) == int:
    #         revise_lst[0] = [revise_lst[0]]

    #     if revise_lst[1] == 2:
    #         append_lst = []
    #         for arg0_idx in revise_lst[0]:
    #             idx = self.peak_idx_lst[arg0_idx]
    #             for lst in self.peak_pair_idx_lst:
    #                 if idx in lst:
    #                     for i in range(2):
    #                         append_lst.append(self.peak_idx_lst.index(lst[i]))
    #         revise_lst[0].extend(append_lst)
    #         revise_lst[0] = list(set(revise_lst[0]))

    #     param_lst = self.best_param_lst
    #     for i in range(len(revise_lst[0])):
    #         param_lst[revise_lst[0][i]][revise_lst[1]] = \
    #             param_lst[revise_lst[0][i]][revise_lst[1]] + revise_lst[2]
    #     self.best_param_lst = param_lst


    # def set_center(self, peak_idx):
    #     """
    #     set         : self.center_peak
    #                   self.center_move
    #     input       : peak_idx; int => peak index used centering
    #     description : if peak_idx is one of the peak of pair peak,
    #                   the mean of the peaks is used
    #     """
    #     center_peak = int(self.peak_idx_lst[peak_idx])
    #     for peak_pair_lst in self.peak_pair_idx_lst:
    #         if self.peak_idx_lst[peak_idx] in peak_pair_lst:
    #             center_peak = peak_pair_lst
    #             break
    #     self.center_peak = center_peak
    #     print("set center peak: %s" % str(self.center_peak))

    #     if type(self.center_peak) == int:
    #         idx = self.peak_idx_lst.index(self.center_peak)
    #         self.center_move = -self.best_param_lst[idx][1]
    #     else:
    #         x0_lst = []
    #         for peak in self.center_peak:
    #             idx = self.peak_idx_lst.index(peak)
    #             x0_lst.append(self.best_param_lst[idx][1])
    #         self.center_move = -np.mean(np.array(x0_lst))
    #     print("lorentzians were moved: %s" % str(self.center_move))

    #     revised_best_param_lst = []
    #     for lst in self.best_param_lst:
    #         revised_best_param_lst.append(
    #           [lst[0], lst[1]+self.center_move, lst[2]])
    #     self.revised_best_param_lst = revised_best_param_lst
    #     print("revised_best_param_lst were set")
