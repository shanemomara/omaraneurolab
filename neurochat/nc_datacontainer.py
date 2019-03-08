# -*- coding: utf-8 -*-
"""
This module implements a container for the Ndata class to simplify multi experiment analyses.

@author: Sean Martin; martins7 at tcd dot ie
"""

from neurochat.nc_data import NData

# Ideas - set up a file class which stores where the filenames are
# Based on the mode being used
# Then these could be loaded on the fly

# If this is done I could set up child classes for each of the modes, then based
# on the class I could then load appropriately when doing this
# I could even call file.load with a data object passed in
# So that memory can be reused between ndata objects

# Could set up all the analyses to work on a list so that it is easy to work with

# And then calling container.container into these analyses would perform the calcs.

# Loading from excel file


class NDataContainer():
    def __init__(self, LoadOnFly=True):
        """
        Parameters
        ----------
        LoadOnFly : Bool
            If True, load files as they are used and maintain one Ndata object
            Otherwise load in bulk and store many NData objects.

        Attributes
        ----------
        container : List
        file_names : List
        
        """
        self.file_names = []
        self.container = []

    def add(self, data):
        container.append(data)

    def add_file(self)

    def load_from_excel(self, )