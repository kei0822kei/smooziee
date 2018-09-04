#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# deals with gpi files (result of the spring8 35XU experiment)
###############################################################################

import os
import re

def file_line_reader_generator(file_path):
    """
    make generator returning each file line

        Parameters
        ----------
        file_path : str
            file path
    """
    with open(file_path, encoding="utf-8") as in_file:
        for line in in_file:
            yield line

class GPI_reader():
    """
    deals with gpi files

        Attributes
        ----------
        file_path : str
            file path
        file_name : str
            file name
    """

    def __init__(self, file_path):
        """
        init

            Parameters
            ----------
            file_path : str
                gpi file path
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

    def _extract_line(self, keyword):
        """
        extract line including keyword

            Parameters
            ----------
            keyword : str
                lines including keyword returns

            Returns
            -------
            match_lines : list of str
                lines including keyword
        """
        generator = file_line_reader_generator(self.file_path)
        match_lines = []
        for i, line in enumerate(generator):
            if keyword in line:
                match_lines.append(line)
        return match_lines

    def _check_list_len(self, lst, num):
        """
        check the length of list

            Parameters
            ----------
            lst : list
                lst which is checked
            num : int
                expected length of list

            Raises
            ------
            ValueError
                the length of lst is not num
        """
        if len(lst) != num:
            ValueError("the length of lst {0} is not num {1}".format(
                str(len(lst)), str(num)))

    def qpoint(self, tf_num):
        """
        get qpoint from gpi file

            Parameters
            ----------
            tf_num : str
                if tf_num='tf_4', get qpoint of tf_4 analizer

            Returns
            -------
            q_point : list of float
                q point of tf_num
        """
        keyword = tf_num+'='
        tf_num_lines = self._extract_line(keyword)
        self._check_list_len(tf_num_lines, 1)
        print(str(tf_num_lines))
        tf_num_line = tf_num_lines[0]
        start = tf_num_line.find("(")
        end = tf_num_line.find(")")
        q_str = tf_num_line[start+1:end]
        q_point = q_str.replace(',', '').split()
        q_point = list(map(float, q_point))
        return q_point
