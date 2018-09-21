#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# deals data from Spring-8 BL35XU
###############################################################################

"""
This script deals data from Spring-8 BL35XU.
"""

import os
import pandas as pd
import numpy as np
from smooziee import gpi

def data2df(filename):
    """
    translate raw data to pd.DataFrame

        Parameters
        ----------
        paramname1 : filename
            BL35XU result filename

        Returns
        -------
        df : pd.DataFrame
            result DataFrame
    """
    df = pd.read_csv(filename, sep='\s+')
    return df

class All_data():
    """
    deals with all data

        Attributes
        ----------
        datafiles : list of str
            data file path list
        gpifiles : list of str
            gpi file path list
        alldata : list of pd.DataFrame
            all data in datafiles
        self.qpoints : list of list of float
            qpoints of each data

        Methods
        -------

        Notes
        -----
    """

    def __init__(self, alldata):
        """
        init

            Parameters
            ----------
            alldata : list
                alldata = [files_1, files_2, ...]
                files_1 = [datafile, gpifile]

            Notes
            -----
            you can use abs file path
        """
        self.datafiles = []
        self.gpifiles = []
        for i in range(len(alldata)):
            self.datafiles.append(alldata[i][0])
            self.gpifiles.append(alldata[i][1])

        self.alldata = []
        self.qpoints = []
        self.datanames = []
        for i in range(len(self.datafiles)):
            data_df = data2df(self.datafiles[i])
            a_name = [ key for key in data_df.keys() if 'a' in key ][0]
            data_df.rename(columns={a_name:'a'}, inplace=True)
            self.alldata.append(data_df)
            dataname = os.path.basename(self.datafiles[i])
            self.datanames.append(dataname)
            tf_num = int(dataname[-1])
            gpi_reader = gpi.GPI_reader(self.gpifiles[i])
            self.qpoints.append(gpi_reader.qpoint(tf_num))
