# -*- coding: utf-8 -*-
"""
This module implements a container for the Ndata class to simplify multi experiment analyses.

@author: Sean Martin; martins7 at tcd dot ie
"""

from enum import Enum
import copy
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
    def __init__(self, share_positions=False, load_on_fly=False):
        """
        Bulk load nData objects

        Attributes
        ----------
        _container : List
        _file_names_dict : Dict
        _units : List
        """
        self._file_names_dict = {}
        self._units = []
        self._container = []
        self._unit_count = 0
        self._share_positions = share_positions
        self._load_on_fly = load_on_fly
        self._last_data_pt = (1, None)

    class EFileType(Enum):
        Spike = 1
        Position = 2
        LFP = 3
        HDF = 4

    def get_num_data(self):
        if self._load_on_fly:
            for _, vals in self.get_file_dict().items():
                return len(vals)
        return len(self._container)
    
    def get_file_dict(self):
        return self._file_names_dict

    def get_units(self, index=None):
        if index is None:
            return self._units
        if index >= self.get_num_data() and (not self._load_on_fly):
            logging.error("Input index to get_data out of range")
            return
        return self._units[index]

    def get_data(self, index=None):
        if self._load_on_fly:
            if index is None:
                logging.error("Can't load all data when loading on the fly")
            result = NData()
            for key, vals in self.get_file_dict().items():
                descriptor = vals[index]
                self._load(key, descriptor, ndata=result)
            return result
        if index is None:
            return self._container
        if index >= self.get_num_data():
            logging.error("Input index to get_data out of range")
            return
        return self._container[index]

    def add_data(self, data):
        if isinstance(data, NData):
            self._container.append(data)
        else:
            logging.error("Adding incorrect object to data container")
            return

    def list_all_units(self):
        if self._load_on_fly:
                for key, vals in self.get_file_dict().items():
                    if key == "Spike":
                        for descriptor in vals:
                            result = NData()
                            self._load(key, descriptor, ndata=result)
                            print("units are {}".format(result.get_unit_list()))
        else:
            for data in self._container:
                print("units are {}".format(data.get_unit_list()))

    def add_files(self, f_type, descriptors):
        if isinstance(descriptors, list):
            descriptors = (descriptors, None, None)
        filenames, _, _ = descriptors
        if not isinstance(f_type, self.EFileType):
            logging.error("Parameter f_type in add files must be of EFileType")
            return

        if f_type.name == "Position" and self._share_positions and len(filenames) == 1:
            for _ in range(len(self.get_file_dict()["Spike"]) - 1):
                filenames.append(filenames[0])

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
    
    def set_units(self, units='all'):
        self._units = []
        if units == 'all':
            if self._load_on_fly:
                vals = self.get_file_dict()["Spike"]
                for descriptor in vals:
                    result = NData()
                    self._load("Spike", descriptor, ndata=result)
                    self._units.append(result.get_unit_list())
            else:
                for data in self.get_data():
                    self._units.append(data.get_unit_list())
        elif isinstance(units, list):
            for idx, unit in enumerate(units):
                if unit == 'all':
                    if self._load_on_fly:
                        vals = self.get_file_dict()["Spike"]
                        descriptor = vals[idx]
                        result = NData()
                        self._load("Spike", descriptor, ndata=result)
                        all_units = result.get_unit_list()
                    else:
                        all_units = self.get_data(idx).get_unit_list()
                    self._units.append(all_units)
                elif isinstance(unit, int):
                    self._units.append([unit])
                elif isinstance(unit, list):
                    self._units.append(unit)
                else:
                    logging.error(
                        "Unrecognised type {} passed to set units".format(type(unit)))

        else:
            logging.error(
                "Unrecognised type {} passed to set units".format(type(units)))
        self._unit_count = self._count_num_units()

    def setup(self):
        if self._load_on_fly:
            self._last_data_pt = (1, None)
        else:
            self._load_all_data()

    def add_files_from_excel(self, file_loc):
        pass

    def subsample(self, key):
        result = copy.deepcopy(self)
        
        for k in result._file_names_dict:
            result._file_names_dict[k] = result._file_names_dict[k][key]
            if isinstance(key, int):
                result._file_names_dict[k] = [result._file_names_dict[k]]
        
        if len(result._units) > 0:
            result._units = result._units[key]
            if isinstance(key, int):
                result._units = [result._units]

        if len(result._container) > 0:
            result._container = result._container[key]
            if isinstance(key, int):
                result._container = [result._container[key]]

        result._unit_count = result._count_num_units()
        return result

    def sort_units_spatially(self, should_sort_list=None, mode="vertical"):
        """
        Sorts the units in the collection based on the centroid of the place field
        mode can be horizontal or vertical
        """

        if mode == "vertical":
            h = 1
        elif mode == "horizontal":
            h = 0
        else:
            logging.error("NDataContainer: Only modes horizontal and vertical are supported")

        if should_sort_list is None:
            should_sort_list = [True for _ in range(self.get_num_data())]

        for idx, bool_val in enumerate(should_sort_list):
            if bool_val:
                centroids = []
                data = self.get_data(idx)
                for unit in self.get_units()[idx]:
                    data.set_unit_no(unit)
                    place_info = data.place()
                    centroid = place_info["centroid"]
                    centroids.append(centroid)
                self._units[idx] = [unit for _, unit in sorted(
                    zip(centroids, self.get_units()[idx]), key= lambda pair: pair[0][h])]

    # Methods from here on should be for private class use
    def _load_all_data(self):
        if self._load_on_fly:
            logging.error("Don't load all the data in container if loading on the fly")
        for key, vals in self.get_file_dict().items():
            for idx, _ in enumerate(vals):
                if idx >= self.get_num_data():
                    self.add_data(NData())

            for idx, descriptor in enumerate(vals):
                self._load(key, descriptor, idx=idx)

    def _load(self, key, descriptor, idx=None, ndata=None):
        if ndata is None:
            ndata = self.get_data(idx)
        key_fn_pairs = {
            "Spike" : [
                getattr(ndata, "set_spike_file"), 
                getattr(ndata, "set_spike_name"),
                getattr(ndata, "load_spike")],
            "Position": [
                getattr(ndata, "set_spatial_file"), 
                getattr(ndata, "set_spatial_name"),
                getattr(ndata, "load_spatial")],
            "LFP": [
                getattr(ndata, "set_lfp_file"),
                getattr(ndata, "set_lfp_name"),
                getattr(ndata, "load_lfp")],
        }

        filename, objectname, system = descriptor

        if objectname is not None:
            key_fn_pairs[key][1](objectname)

        if system is not None:
            ndata.set_system(system)

        if key == "Position" and self._share_positions and idx !=0:
            if self._load_on_fly:
                ndata.spatial = self._last_data_pt[1].spatial
            else:
                ndata.spatial = self.get_data(0).spatial
            return

        if filename is not None:
            key_fn_pairs[key][0](filename)
            key_fn_pairs[key][2]()
        
    def __repr__(self):
        string = "NData Container Object with {} objects:\nFiles are {}\nUnits are {}\nSet to Load on Fly? {}".format(
            self.get_num_data(), self.get_file_dict(), self.get_units(), self._load_on_fly)
        return string

    def __getitem__(self, index):
        data_index, unit_index = self._index_to_data_pos(index)
        if self._load_on_fly:
            if data_index == self._last_data_pt[0]:
                result = self._last_data_pt[1]
            else:
                result = NData()
                for key, vals in self.get_file_dict().items():
                    descriptor = vals[data_index]
                    self._load(key, descriptor, idx=data_index, ndata=result)
                self._last_data_pt = (data_index, result)
        else:  
            result = self.get_data(data_index)
        if len(self.get_units()) > 0:
            result.set_unit_no(self.get_units(data_index)[unit_index])
        return result

    def __len__(self):
        counts = self._unit_count
        if counts == 0:
            counts = [1 for _ in range(len(self._container))]  
        return sum(counts)

    def _count_num_units(self):
        counts = []
        for unit_list in self.get_units():
            counts.append(len(unit_list))
        return counts

    def _index_to_data_pos(self, index):
        counts = self._unit_count
        if counts == 0:
            counts = [1 for _ in range(len(self._container))]
        if index >= len(self):
            raise IndexError
        else:
            running_sum, running_idx = 0, 0
            for count in counts:
                if index < (running_sum + count):
                    return running_idx, (index - running_sum)
                else:
                    running_sum += count
                    running_idx += 1

                
        