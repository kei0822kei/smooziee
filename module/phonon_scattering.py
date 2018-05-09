#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# this script deals with phonon scattering experimental data
###############################################################################

### import modules
import os
import sys
import numpy as np
import pandas as pd
import scipy.optimize
import joblib
from scipy.signal import argrelmax
from smooziee.module import math_tools
from sklearn.metrics import mean_squared_error


###############################################################################
# phonon scattering
###############################################################################

class Process():
    """
    deals with phonon scattering experimental data
    """

    def __init__(self, raw_data):
        """
        input       : raw_data; str => raw data file path
                        ex) raw_data="KCl_GXL511_m0p25_RS_4"
        description : read dat file and make DataFrame
        """
        data_df = pd.read_csv(raw_data, sep='\s+')

        ### set self
        self.data_df = data_df
        self.filename = os.path.basename(raw_data)
        self.meV_y_arr = np.array(data_df.loc[:, ['meV', 'y_unitpk']])
        self.peak_idx_lst = None
        self.peak_pair_idx_lst = None  ### [[a, b], [c, d]]
        self.best_param_lst = None  ### [[initA_0, initx0_0, initd_0], ...]
        # self.grid_param_lst = None


    def find_peak(self, order=20):
        """
        input       : data_arr; lst or np.array(1 dimension)
                      order; int => default (order=5)
        output      : np.array => argrelmax_arr
        description : find peak from data_lst
                      you can change the value of 'order', which is parameter
                      of definition 'scipy.signal.argrelmax'
                      see more about 'scipy.signal.argrelmax'
                      http://jinpei0908.hatenablog.com/entry/2016/11/26/224216
        """
        self.peak_idx_lst = argrelmax(np.array(data_lst), order=order)
        print("found %s peaks" % len(self.peak_idx_lst[0]))


    def revise_peak(self, peak_arr):
        """
        input       : peak_arr; np.array => revise self.peak_idx_lst
        description : revise self.peak_idx_lst
                          peak_arr = self.peak_idx_lst
        """
        self.peak_idx_lst = peak_arr


    def find_peak_pair(self, threshold=6):
        """
        input       : threshold; float or int => threshold=6 (default)
        description : recognize A and B as pair if abs(A) - abs(B) < threshold
        set         : self.peak_pair_idx_lst
        """
        if self.peak_idx_lst == None:
            print("You have to execute find_peak ahead !")
            sys.exit(1)

        ### condition => stokes anti-stokes
        pair_lst = []
        flag_lst = []
        for i in range(int(len(peak_idx_lst)/2)):
            #for j in range(len(peak_idx_lst)-1, i, -1):
            for j in range(len(peak_idx_lst)-1, i, -1):
                if abs(abs(init_x0_lst[i])-abs(init_x0_lst[j])) < threshold \
                        and i not in flag_lst and j not in flag_lst:
                    init_d_lst[i] = init_d_lst[j] = \
                      (init_d_lst[i] + init_d_lst[j]) / 2
                    pair_lst.append([i, j])
                    flag_lst.extend([i, j])

        self.peak_pair_idx_lst = pair_lst
        print("found %s pair" % str(len(self.peak_pair_idx_lst)))


    def initial_fit(idx_range=10):
        """
        input       : idx_range; int => idx_range = 10 (default)
                          peak fit using data_arr[peak_idx-10:peak_idx+10, 0]
                          if idx_range = 10
        set         : self.best_param_lst
        description : make initial fit using self.peak_idx_lst
        """
        ### check
        if self.peak_idx_lst == None:
            print("You have to execute find_peak ahead!")
            sys.exit(1)
        if self.peak_pair_idx_lst == None:
            print("You have to execute find_peak_pair ahead!")
            sys.exit(1)

        print("make initial fitting")
        best_param_lst = []
        for peak_idx in peak_idx_lst:
            param_lst = scipy.optimize.curve_fit(
                math_tools.lorentzian,
                data_arr[peak_idx-idx_range:peak_idx+idx_range, 0],
                data_arr[peak_idx-idx_range:peak_idx+idx_range, 1],
                ### p0 => initial peak point
                p0=[data_arr[peak_idx, 1], data_arr[peak_idx, 0], 1.]
            )
            best_param_lst.append( \
                [param_lst[0][0], param_lst[0][1], param_lst[0][2]])

        ### stokes anti-stokes revise param d
        for idx_pair_lst in self.peak_pair_idx_lst:
            mean_d_val = (best_param_lst[idx_pair_lst[0]] +
                          best_param_lst[idx_pair_lst[1]]) / 2
            best_param_lst[idx_pair_lst[0]] = mean_d_val
            best_param_lst[idx_pair_lst[1]] = mean_d_val

        self.best_param_lst = best_param_lst


    def make_grid_param(param_nw_dic=
                            {'A':[3, 0.02], 'x0':[1, 0.5], 'd':[1, 0.02]}):
        """
        input       : idx_range; int => idx_range = 10 (default)
                          peak fit using data_arr[peak_idx-10:peak_idx+10, 0]
                          if idx_range = 10
        set         : self.grid_param_lst
        description : make initial fit using self.peak_idx_lst
        """
        ### check
        if self.peak_idx_lst == None:
            print("You have to execute find_peak ahead!")
            sys.exit(1)
        if self.peak_pair_idx_lst == None:
            print("You have to execute find_peak_pair ahead!")
            sys.exit(1)

        ### make parameter for grid search
        for i in range(len(self.peak_idx_lst)):
            param_info_dic['A_'+str(i)] = \
              [self.peak_pair_idx_lst[i][0], param_nw_dic['A'][0], param_nw_dic['A'][1]]
            param_info_dic['x0_'+str(i)] = \
              [self.peak_pair_idx_lst[i][1], param_nw_dic['x0'][0], param_nw_dic['x0'][1]]
            param_info_dic['d_'+str(i)] = \
              [self.peak_pair_idx_lst[i][2], param_nw_dic['d'][0], param_nw_dic['d'][1]]
        param_lst = math_tools.make_grid_param(param_info_dic)
        print("initial param num is %s" % str(len(param_lst)))

        ### stokes anti-stokes revise param d
        print("if not same d value, remove from param_lst")
        for param_dic in param_lst:
            param_flag = 0
            for pair_idx_lst in peak_pair_idx_lst:
                if param_dic['d_'+str(pair_idx_lst[0])] != \
                        param_dic['d_'+str(pair_idx_lst[1])]:
                    param_flag = 1
            if param_flag == 1:
                param_lst.remove(param_dic)
        print("final param num is %s" % str(len(param_lst)))

        self.grid_param_lst = param_lst


    def grid_search():
        """
        set         : self.best_param_lst
        description : execute grid search using self.param_lst
        """
        ### check
        if self.grid_param_lst == None:
            print("You have to execute make_grid_param ahead!")
            sys.exit(1)

        ### grid search
        score_lst =[]
        for param_dic in self.grid_param_lst:

            smooth_y_arr = 0
            for i in range(len(self.peak_idx_lst)):
                ### condition => 2d must be more than 1.5 meV
                #if param_dic['d_'+str(i)] < 0.75:
                #    param_dic['d_'+str(i)] = 0.75

                smooth_y_arr = smooth_y_arr + math_tools.lorentzian( \
                                   data_arr[:, 0],
                                   param_dic['A_'+str(i)],
                                   param_dic['x0_'+str(i)],
                                   param_dic['d_'+str(i)]
                               )
            score_lst.append(
                mean_squared_error(data_arr[:,1], smooth_y_arr))

        best_score_idx = score_lst.index(min(score_lst))
        print("best score was %s" % str(min(score_lst)))
        final_param_dic = param_lst[best_score_idx]
        best_param_lst = []
        for i in range(len(self.peak_idx_lst)):
            best_param_lst.append(
                [final_param_dic['A_'+str(i)], \
                 final_param_dic['x0_'+str(i)], \
                 final_param_dic['d_'+str(i)]])
        self.best_param_lst = best_param_lst
        print("best param was set to self.best_param_lst")


    def saveobj(self, savefile=self.filename):
        """
        input         : savefile
        description   : joblib.dump(self, savefile)
        """
        joblib.dump(self, savefile)



    def plot(self, ax, param_nw_dic=None, run_mode='raw', threshold=6,
        """
        input         : ax;  ex) ax = fig.add_subplot(111)
                        param_nw_dic; dic => {'A':[grid_num, width], 'x0': ...}
                        run_mode; str => default (run_mode='raw')
                        order; int or float => default (order=5)
        output        : ax
        option        : run_mode; 'raw' or 'peak' or 'smooth'
        description   : return ax which is painted data plot and data peak
        """
        ### check run_mode

        ### raw data
        data_arr = np.array(self.data_df.loc[:, ['meV', 'y_unitpk']])
        ax.scatter(data_arr[:,0], data_arr[:,1], c='red', s=2)

        ### find peak
        if self.best_param_lst == None:
            if self.peak_idx_lst != None:
                ax.scatter(data_arr[self.peak_idx_lst,0], data_arr[self.peak_idx_lst,1],
                           c='black', s=10)
    
        ### smoothing
        else:
            ### scatter
            color_lst = ['pink', 'yellow', 'green']
            c_lst = [ 'black' for _ in range(len(self.peak_idx_lst)) ]
            for i in range(len(self.peak_pair_idx_lst)):
                for j in self.peak_pair_idx_lst[i]:
                    c_lst[j] = color_lst[i]
            print(c_lst)

            for i in range(len(self.peak_pair_idx_lst)):
                ax.scatter(data_arr[self.peak_pair_idx_lst[i],0], \
                           data_arr[self.peak_pair_idx_lst[i],1],
                           c=c_lst[i], s=30)

            curve_x_arr = np.linspace(
                min(data_arr[:,0]), max(data_arr[:,0]), 200)
            curve_y_arr = 0
            for param in range(len(self.best_param_lst)):
                curve_y_arr += math_tools.lorentzian(curve_x_arr,
                                   param[0], param[1], param[2]
                               )
                ax.plot(curve_x_arr, curve_y_arr, c='blue', linewidth=0.3,
                        linestyle='--')

            ### plot
            ax.plot(curve_x_arr, curve_y_arr, c='black', linewidth=0.5)

        ### setting
        ax.set_xlabel('meV', fontsize=fontsize)
        ax.set_ylabel('y_unitpk', fontsize=fontsize)
        ax.set_title(self.filename)
