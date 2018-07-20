#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# math tools using smooziee
###############################################################################

### import modules
from sklearn.model_selection import ParameterGrid


###############################################################################
# functions
###############################################################################

# def lorentzian(x, p_lst):
#     A = p_lst[0]
#     x0 = p_lst[1]
#     d = p_lst[2]
#     return A * ( d**2 / ((x-x0)**2 + d**2) )

# def lorentzian(x, A, x0, d):
#     return A * ( d**2 / ((x-x0)**2 + d**2) )

###############################################################################
# parameter optimize
###############################################################################

# def make_grid_param(param_info_dic, notice=False):
#     """
#     input       : param_info_dic; dict
#                       => param_info_dic = {'param_1':[median, grid_num, width],
#                                            'param_2':[      ...              ],
#                                                     ......
#                                            }
#     output      : list
#     definition  : make param_lst, which includes all parameter set
#                   for grid search
#     """
#     dic = {}
#     for key in param_info_dic.keys():
#         median = param_info_dic[key][0]
#         grid_num = param_info_dic[key][1]
#         width = param_info_dic[key][2]
#         min_val = median - ((grid_num - 1) * width / 2)
#         each_param_lst = [ min_val+i*width for i in range(grid_num) ]
#         dic[key] = each_param_lst
#     param_lst = list(ParameterGrid(dic))
#     if notice:
#         print("total grid parameter is %s" % len(param_lst))
# 
#     return param_lst
