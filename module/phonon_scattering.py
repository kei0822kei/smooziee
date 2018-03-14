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
from scipy.signal import argrelmax
from smooziee.module import math_tools
from sklearn.metrics import mean_squared_error


###############################################################################
# find peak from input data
###############################################################################

def find_peak(data_lst, order=20):
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
    idx_arr = argrelmax(np.array(data_lst), order=order)
    print("found %s peaks" % len(idx_arr[0]))

    return idx_arr


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
        self.data_df = data_df
        self.filename = os.path.basename(raw_data)

    def meV_y_unitpk(self, ax, param_nw_dic=None, run_mode='raw', threshold=6,
                     get_data=False, order=20, fontsize=10):
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
        if run_mode == 'peak':
            peak_idx_lst = find_peak(
                               data_arr[:,1], order=order)[0]
            ax.scatter(data_arr[peak_idx_lst,0], data_arr[peak_idx_lst,1],
                       c='black', s=10)

        ### smoothing
        if run_mode == 'smooth':
            peak_idx_lst = find_peak(
                               data_arr[:,1], order=order)[0]

            ### initial smoothing
            init_A_lst = []
            init_x0_lst = []
            init_d_lst = []
            for peak_idx in peak_idx_lst:
                init_param_lst = scipy.optimize.curve_fit(
                    math_tools.lorentzian,
                    data_arr[peak_idx-10:peak_idx+10, 0],
                    data_arr[peak_idx-10:peak_idx+10, 1],
                    p0=[data_arr[peak_idx, 1], data_arr[peak_idx, 0], 1.]
                )
                init_A_lst.append(init_param_lst[0][0])
                init_x0_lst.append(init_param_lst[0][1])
                init_d_lst.append(init_param_lst[0][2])

            ### condition => stokes anti-stokes
            pear_lst = []
            flag_lst = []
            for i in range(int(len(peak_idx_lst)/2)):
                #for j in range(len(peak_idx_lst)-1, i, -1):
                for j in range(len(peak_idx_lst)-1, i, -1):
                    if abs(abs(init_x0_lst[i])-abs(init_x0_lst[j])) < threshold \
                            and i not in flag_lst and j not in flag_lst:
                        init_d_lst[i] = init_d_lst[j] = \
                          (init_d_lst[i] + init_d_lst[j]) / 2
                        pear_lst.append([i, j])
                        flag_lst.extend([i, j])

            ### scatter
            color_lst = ['pink', 'yellow', 'green']
            c_lst = [ 'black' for _ in range(len(peak_idx_lst)) ]
            for i in range(len(pear_lst)):
                for j in pear_lst[i]:
                    c_lst[j] = color_lst[i]
            print(c_lst)

            for i in range(len(peak_idx_lst)):
                ax.scatter(data_arr[peak_idx_lst[i],0], data_arr[peak_idx_lst[i],1],
                       c=c_lst[i], s=30)

            ### make parameter for grid search
            param_info_dic = {}
            for i in range(len(init_A_lst)):
                param_info_dic['A_'+str(i)] = \
                  [init_A_lst[i], param_nw_dic['A'][0], param_nw_dic['A'][1]]
                param_info_dic['x0_'+str(i)] = \
                  [init_x0_lst[i], param_nw_dic['x0'][0], param_nw_dic['x0'][1]]
                param_info_dic['d_'+str(i)] = \
                  [init_d_lst[i], param_nw_dic['d'][0], param_nw_dic['d'][1]]
            param_lst = math_tools.make_grid_param(param_info_dic)

            ### grid search
            score_lst =[]
            for param_dic in param_lst:
                for each_pear_lst in pear_lst:
                    if param_dic['d_'+str(each_pear_lst[0])] != param_dic['d_'+str(each_pear_lst[1])]:
                        continue

                smooth_y_arr = 0
                for i in range(len(init_A_lst)):
                    ### condition => 2d must be more than 1.5 meV
                    if param_dic['d_'+str(i)] < 0.75:
                        param_dic['d_'+str(i)] = 0.75

                    smooth_y_arr = smooth_y_arr + math_tools.lorentzian( \
                                       data_arr[:, 0],
                                       param_dic['A_'+str(i)],
                                       param_dic['x0_'+str(i)],
                                       param_dic['d_'+str(i)]
                                   )
                score_lst.append(
                    mean_squared_error(data_arr[:,1], smooth_y_arr))

            best_score_idx = score_lst.index(min(score_lst))
            final_param_dic = param_lst[best_score_idx]
            print("final parameter is "+str(final_param_dic))

            curve_x_arr = np.linspace(
                min(data_arr[:,0]), max(data_arr[:,0]), 200)
            curve_y_arr = 0
            for i in range(len(init_A_lst)):
                curve_y_arr += math_tools.lorentzian(curve_x_arr,
                                   final_param_dic['A_'+str(i)],
                                   final_param_dic['x0_'+str(i)],
                                   final_param_dic['d_'+str(i)]
                               )
                ax.plot(curve_x_arr, curve_y_arr, c='blue', linewidth=0.3,
                        linestyle='--')

            ### plot
            ax.plot(curve_x_arr, curve_y_arr, c='black', linewidth=0.5)

        ### setting
        ax.set_xlabel('meV', fontsize=fontsize)
        ax.set_ylabel('y_unitpk', fontsize=fontsize)
        ax.set_title(self.filename)
