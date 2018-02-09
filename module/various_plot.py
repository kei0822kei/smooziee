#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# plot various figures
###############################################################################

### import modules
from peakfit.module import process


###############################################################################
# plot data with its peak
###############################################################################

def data_peak(ax, data_arr, order=5, xlabel='x', ylabel='y', fontsize=10):
    """
    input         : ax;  ex) ax = fig.add_subplot(111)
                    data_arr; np.array => 2-dimension
                    idx_lst; list => idx_lst=[xlabel, ylabel]
                                     default (idx_lst=['x', 'y'])
                    order; int => default (order=5)
    output        : ax
    description   : return ax which is painted data plot and data peak
    """
    ### find peak
    peak_idx_lst = process.find_peak(data_arr[:,1], order=order)

    ### plot
    ax.scatter(data_arr[:,0], data_arr[:,1], c='red', s=2)
    ax.scatter(data_arr[peak_idx_lst,0], data_arr[peak_idx_lst,1],
               c='black', s=10)

    ### setting
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(xlabel+'-'+ylabel+' plot')
