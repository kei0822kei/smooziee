#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# math tools using smooziee
###############################################################################

### import modules
import sys
import numpy as np
from sklearn.model_selection import ParameterGrid


###############################################################################
# functions
###############################################################################

def lorentzian(x, A, x0, d):
    return A * ( d**2 / ((x-x0)**2 + d**2) )

###############################################################################
# parameter optimize
###############################################################################

#def make_grid_param(name_lst, median_lst, grid_num_lst, width_lst):
def make_grid_param(param_info_dic):
    """
    input       :
    output      :
    definition  :
    """
    dic = {}
    for key in param_info_dic.keys():
        median = param_info_dic[key][0]
        grid_num = param_info_dic[key][1]
        width = param_info_dic[key][2]
        min_val = median - ((grid_num - 1) * width / 2)
        each_param_lst = [ min_val+i*width for i in range(grid_num) ]
        dic[key] = each_param_lst
    param_lst = list(ParameterGrid(dic))

    return param_lst
