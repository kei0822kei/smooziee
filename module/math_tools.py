#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# preprocess raw data using this script
###############################################################################

### import modules
import sys
import numpy as np
from scipy.optimize import curve_fit


###############################################################################
# functions
###############################################################################

def multi_lorentzian(x, *param_lst):
    """
    input       : x => int or float or np.array
                  param_lst; list => [A_0, x0_0, d_0], [A_1, x0_1, d_1], ...
    output      :
    definition  : f_0(x) = A_0 * ( d_0**2 / ((x-x0_0)**2 + d_0**2) )
                  f_1(x) = A_1 * ( d_1**2 / ((x-x0_1)**2 + d_1**2) )
                                ...

                  f(x) = f_0(x) + f_1(x) + ...
    """
    def lorentzian(x, A, x0, d):
        return A * ( d**2 / ((x-x0)**2 + d**2) )

    y = 0
    for each_param_lst in param_lst:
        if len(each_param_lst) != len(param_lst):
            print("len(each_param_lst) must be 3, found %s" % len(param_lst))
            sys.exit(1)
        y = y + lorentzian(param_lst[0], param_lst[1], param_lst[2])

    return y

def lorentzian_func(A, x0, d):
    return lambda x: A * ( d**2 / ((x-x0)**2 + d**2) )

def lorentzian(x, A, x0, d):
    return A * ( d**2 / ((x-x0)**2 + d**2) )

###############################################################################
# parameter optimize
###############################################################################

def param_optimizer(func, x_arr, y_arr):
    """
    input       : func; function => func=multi_lorentzian
                  x_arr, y_arr; np.array
    output      : 
    definition  : parameter fitting using 'scipy.optimize.curve_fit'
    """
    return curve_fit(func, x_arr, y_arr)
