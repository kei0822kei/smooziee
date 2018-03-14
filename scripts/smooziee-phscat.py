#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# deals with phonon scattering experimental data
###############################################################################

### import modules
import os
import sys
import argparse
from matplotlib import pyplot as plt
from smooziee.module import phonon_scattering as ph_scat

### Arg-parser
parser = argparse.ArgumentParser(
    description="")
parser.add_argument('--filename', type=str,
                    help="input file name  ex) KCl_GXL511_m0p25_RT_4")
parser.add_argument('--run_mode', type=str, default='raw',
                    help="choose run mode => 'raw' or 'peak' or 'smooth'")
parser.add_argument('--param_A', type=str, default='3 0.02',
                    help="set parameter 'A' of lorentzian for grid search\
                          first arg; grid num\
                          second arg; grid width\
                          ex) --param_A='3 0.02' (default)")
parser.add_argument('--param_x0', type=str, default='1 0.5',
                    help="set parameter 'x0' of lorentzian for grid search\
                          first arg; grid num\
                          second arg; grid width\
                          ex) --param_x0='1 0.5' (default)")
parser.add_argument('--param_d', type=str, default='1 0.02',
                    help="set parameter 'd' of lorentzian for grid search\
                          first arg; grid num\
                          second arg; grid width\
                          ex) --param_d='1 0.02' (default)")
parser.add_argument('--order', type=int, default=20,
                    help="the more order num is, the less you find its peaks")
parser.add_argument('--threshold', type=int, default=6,
                    help="use this value, finding stokes anti-stokes")
parser.add_argument('--show', type=bool, default=False,
                    help="show figure (default) False")
parser.add_argument('--savefig', type=str,
                    help="directory where figure is saved")
args = parser.parse_args()

### main
fig = plt.figure()
ax = fig.add_subplot(111)
scat = ph_scat.Process(args.filename)

if args.run_mode == 'raw':
    scat.meV_y_unitpk(ax, run_mode=args.run_mode)

if args.run_mode == 'peak':
    scat.meV_y_unitpk(ax, run_mode=args.run_mode, order=args.order)

if args.run_mode == 'smooth':
    A_lst = list(map(float, args.param_A.split()))
    x0_lst = list(map(float, args.param_x0.split()))
    d_lst = list(map(float, args.param_d.split()))
    A_lst[0] = int(A_lst[0])
    x0_lst[0] = int(x0_lst[0])
    d_lst[0] = int(d_lst[0])

    ### check
    if len(A_lst) != 2 or len(x0_lst) != 2 or len(d_lst) != 2:
        print("the number of params is not 2")
        print("A_lst = %s" % A_lst)
        print("x0_lst = %s" % x0_lst)
        print("d_lst = %s" % d_lst)
        sys.exit(1)

    param_nw_dic = {'A': A_lst, 'x0': x0_lst, 'd': d_lst}
    scat.meV_y_unitpk(ax, param_nw_dic=param_nw_dic, run_mode=args.run_mode,
                      order=args.order, threshold=args.threshold)

plt.savefig(args.savefig)

if args.show == True:
    plt.show()
