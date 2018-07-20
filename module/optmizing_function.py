#!/usr/bin/env python
# -*- coding: utf-8 -*-




#return the function to be optimized#
class Optimizing_function():
    def __init__(self):

        #set self#
        self.func_lst = None
        self.param_lst = None
        self.pair_param_lst = None
        self.error_func = None


    def set_function(self, func_lst):
        """
        set the type of function
        ex)["gaussian", "lonentz"]
        """
        self.func_lst = func_lst


    def set_param_lst(self, param_lst):
        """
        set the all parameters of each function
        ex) [[1,1,0], [1,1,2]]
        """
        self.param_lst = param_lst

        if len(self.func_lst) != len(param_lst):
            print("The number of functions and parameters must be the same")



    def set_pair_param_lst(self, pair_index_lst):
        """
        input : pair_index_lst;lst of the which function is pair
        ex)[1,3]
        """


    def split_const_and_variable_params(self, 
