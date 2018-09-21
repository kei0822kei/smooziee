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
        Attribute1 : int
            description
        Attribute2 : int, default var
            description

        Methods
        -------

        Notes
        -----
    """

    def __init__(alldata):
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
            self.datafiles.append(alldata[0])
            self.gpifiles.append(alldata[1])

        self.alldata = []
        self.qpoints = []
        self.datanames = []
        for i in range(len(self.datafiles)):
            self.alldata.append(data2df(self.datafiles[i]))
            dataname = os.basename(self.datafiles[i])
            self.datanames.append(dataname)
            tf_num = int(dataname[-1])
            gpi_reader = gpi.GPI_reader(self.gpifiles[i], tf_num=tf_num)
            self.qpoints.append(gpi_reader.qpoint(tf_num))
