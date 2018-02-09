#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# process raw data using this script
###############################################################################

### import modules
import os
import sys
import numpy as np
import pandas as pd


###############################################################################
# find peak from input data
###############################################################################

def find_peak(data_lst, order=5):
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
    from scipy.signal import argrelmax
    idx_arr = argrelmax(np.array(data_lst), order=order)

    return idx_arr


###############################################################################
# read raw data
###############################################################################

class Process():
    """
    process raw data using this script
    """

    def __init__(self, raw_data):
        """
        input       : raw_data; str => raw data file path
                        ex) raw_data="KCl_GXL511_m0p25_RS_4"
        description : read dat file and make DataFrame
        """
        data_df = pd.read_csv(raw_data, sep='\s+')
        self.data_df = data_df

    def meV_y_unitpk(self):
        """
        output      : pd.DataFrame
        description : return meV and unitpk data
        """
        return self.data_df.loc[:, ['meV', 'y_unitpk']]
