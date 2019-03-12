# -*- coding: utf-8 -*-
"""
This module implements a container for the Ndata class to simplify multi experiment analyses.

@author: Sean Martin; martins7 at tcd dot ie
"""

from enum import Enum
import logging

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
    def __init__(self, load_on_fly=True):
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
        self._load_on_fly = load_on_fly
        self._file_names_dict = {}
        self._container = []

    class EFileType(Enum):
        Spike = 1
        Position = 2
        LFP = 3
        HDF = 4

    def get_num_data(self):
        return len(self._container)
    
    def get_file_dict(self):
        return self._file_names_dict

    def add_data(self, data):
        if isinstance(data, NData):
            self._container.append(data)
        else:
            logging.error("Adding incorrect object to data container")
            return

    def get_data(self, index):
        if index >= self.get_num_data():
            logging.error("Input index to get_data out of range")
            return
        return self._container[index]

    def add_files(self, f_type, descriptors):
        filenames, _, _ = descriptors
        if not isinstance(f_type, self.EFileType):
            logging.error("Parameter f_type in add files must be of EFileType")
            return

        # Ensure lists are empty or of equal size    
        for l in descriptors:
            if l is not None:
                if len(l) != len(filenames):
                    logging.error(
                        "add_files called with differing number of filenames and other data"
                    )
                    return

        for idx in range(len(filenames)):
            description = []
            for el in descriptors:
                if el is not None:
                    description.append(el[idx])
                else:
                    description.append(None)
            self._file_names_dict.setdefault(
                f_type.name, []).append(description)

    def _load(self, ndata, key, descriptor):
        key_fn_pairs = {
            "Spike" : [
                getattr(ndata, "set_spike_file"), 
                getattr(ndata, "set_spike_name")]
        }

        filename, objectname, system = descriptor

        if objectname is not None:
            key_fn_pairs[key][1](objectname)

        if system is not None:
            ndata.set_system(system)

        if filename is not None:
            key_fn_pairs[key][0](filename)
            ndata.load_spike()

    def load_all_data(self):
        for key, vals in self.get_file_dict().items():
            for idx, _ in enumerate(vals):
                if idx >= self.get_num_data():
                    self.add_data(NData())
            
            for idx, descriptor in enumerate(vals):
                self._load(
                    self.get_data(idx), key, descriptor)

    def add_files_from_excel(self, file_loc):
        pass