#!/usr/bin/env python
# -*- coding: utf-8 -*-


###############################################################################
# functions
###############################################################################

import numpy as np

def lorentzian(x, A, x0, d):
    """
    input         : x; float or np.array
                    param_lst; [A, x0, d]
    output        : float or np.array
    description   : lorentzin
    """
    return 2*A / np.pi * ( d / (4*(x-x0)**2 + d**2) )

# def lorentzian_for_curve_fit(x, A, x0, d):
#     return A * ( d**2 / ((x-x0)**2 + d**2) )
