# -*- coding: utf-8 -*-
"""
This module implements a container for the Ndata class to simplify multi experiment analyses.

@author: Sean Martin; martins7 at tcd dot ie
"""

from enum import Enum
import copy
import logging
import os

import pandas as pd
import numpy as np

from neurochat.nc_data import NData
from neurochat.nc_utils import get_all_files_in_dir
from neurochat.nc_utils import has_ext


class NDataContainer():
    """
    Class for storing multiple file locations for ndata objects.

    Additionally the ndata objects themselves can be stored
    """

    def __init__(self, share_positions=False, load_on_fly=False):
        """
        Initialise the class.

        Parameters
        ----------
        share_positions : bool
            Share the same position file between the data objects
        load_on_fly : bool
            Don't store all the data in memory,
            instead load it as needed, on the fly

        Attributes
        ----------
        _container : List
        _file_names_dict : Dict
        _units : List
        _unit_count : int
        _share_positions : bool
        _load_on_fly : bool
        _smoothed_speed : bool
        _last_data_pt : tuple (int, NData)

        """
        self._file_names_dict = {}
        self._units = []
        self._container = []
        self._unit_count = 0
        self._share_positions = share_positions
        self._load_on_fly = load_on_fly
        self._last_data_pt = (1, None)
        self._smoothed_speed = False

    class EFileType(Enum):
        """The different filetypes that can be added to an object."""

        Spike = 1
        Position = 2
        LFP = 3

    def get_num_data(self):
        """Return the number of Ndata objects in the container."""
        if self._load_on_fly:
            for _, vals in self.get_file_dict().items():
                return len(vals)
        return len(self._container)

    def get_file_dict(self):
        """Return the key value filename dictionary for this collection."""
        return self._file_names_dict

    def get_units(self, index=None):
        """
        Return the units in this collection, optionally at a given index.

        Parameters
        ----------
        index : int
            Optional collection data index to get the units for

        Returns
        -------
        list
            Either a list containing lists of all units in the collection
            or the list of units for the given data index

        """
        if index is None:
            return self._units
        if index >= self.get_num_data() and (not self._load_on_fly):
            logging.error("Input index to get_data out of range")
            return
        return self._units[index]

    def get_data(self, index=None):
        """
        Return the NData objects in this collection, or a specific object.

        Do not call this with no index if loading data on the fly

        Parameters
        ----------
        index : int
            Optional index to get data at

        Returns
        -------
        NData or list of NData objects

        """
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
        """Add an NData object to this container."""
        if isinstance(data, NData):
            self._container.append(data)
        else:
            logging.error("Adding incorrect object to data container")
            return

    def list_all_units(self):
        """Print all the units in the container."""
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
        """
        Add a list of filenames of the given type to the container.

        Parameters
        ----------
        f_type : EFileType:
            The type of file being added (Spike, LFP, Position)
        descriptors : list
            Either a list of filenames, or a list of tuples in the order
            (filenames, obj_names, data_sytem). Filenames should be absolute

        Returns
        -------
        None

        """
        if isinstance(descriptors, list):
            descriptors = (descriptors, None, None)
        filenames, _, _ = descriptors
        if not isinstance(f_type, self.EFileType):
            logging.error(
                "Parameter f_type in add files must be of EFileType\n" +
                "given {}".format(f_type))
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

    def add_all_files(self, spats, spikes, lfps):
        """
        Quickly add a list of positions, spikes and lfps.

        Parameters
        ----------
        spats : list
            The list of spatial files
        spikes : list
            The list of spike files
        lfps : list
            The list of lfp files

        Returns
        -------
        None

        """
        self.add_files(self.EFileType.Position, spats)
        self.add_files(self.EFileType.Spike, spikes)
        self.add_files(self.EFileType.LFP, lfps)

    def set_units(self, units='all'):
        """Set the list of units for the collection."""
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
        """Perform data initialisation based on the input filenames."""
        if self._load_on_fly:
            self._last_data_pt = (1, None)
        else:
            self._load_all_data()

    def add_files_from_excel(self, file_loc, unit_sep=" "):
        """
        Add filepaths from an excel file.

        These should be setup to be in the order:
        directory | position file | spike file | unit numbers | eeg extension

        Parameters
        ----------
        file_loc : str
            Name of the excel file that contains the data specifications
        unit_sep : str
            Optional separator character for unit numbers, default " "

        Returns
        -------
        excel_info :
            The raw info parsed from the excel file for further use

        """
        pos_files = []
        spike_files = []
        units = []
        lfp_files = []
        to_merge = []

        if os.path.exists(file_loc):
            excel_info = pd.read_excel(file_loc, index_col=None)
            if excel_info.shape[1] % 5 != 0:
                logging.error(
                    "Incorrect excel file format, it should be:\n" +
                    "directory | position file | spike file" +
                    "| unit numbers | eeg extension")
                return

            # excel_info = excel_info.iloc[:, 1:] # Can be used to remove index
            count = 0
            for full_row in excel_info.itertuples():
                split = [full_row[i:i + 5]
                         for i in range(1, len(full_row), 5)
                         if not pd.isna(full_row[i])]
                merge = True if len(split) > 1 else False
                merge_list = []
                for row in split:
                    base_dir = row[0]
                    pos_name = row[1]
                    tetrode_name = row[2]

                    if pos_name[-4:] == '.txt':
                        spat_file = base_dir + os.sep + pos_name
                    else:
                        spat_file = base_dir + os.sep + pos_name + '.txt'

                    spike_file = base_dir + os.sep + tetrode_name

                    # Load the unit numbers
                    unit_info = row[3]
                    if unit_info == "all":
                        unit_list = "all"
                    elif isinstance(unit_info, int):
                        unit_list = unit_info
                    elif isinstance(unit_info, float):
                        unit_list = int(unit_info)
                    else:
                        unit_list = [
                            int(x) for x in unit_info.split(" ") if x is not ""]

                    # Load the lfp
                    lfp_ext = row[4]
                    if lfp_ext[0] != ".":
                        lfp_ext = "." + lfp_ext
                    spike_name = spike_file.split(".")[0]
                    lfp_file = spike_name + lfp_ext

                    pos_files.append(spat_file)
                    spike_files.append(spike_file)
                    lfp_files.append(lfp_file)
                    units.append(unit_list)
                    merge_list.append(count)
                    count += 1
                if merge:
                    to_merge.append(merge_list)

            # Complete the file setup based on parsing from the excel file
            self.add_all_files(pos_files, spike_files, lfp_files)
            self.setup()
            self.set_units(units)

            for idx, merge_list in enumerate(to_merge):
                self.merge(merge_list)
                for j in range(idx + 1, len(to_merge)):
                    to_merge[j] = [
                        k - len(merge_list) + 1 for k in to_merge[j]]
            return excel_info
        else:
            logging.error('Excel file does not exist!')
            return None

    # Created by Sean Martin with help from Matheus Cafalchio
    def add_axona_files_from_dir(self, directory, **kwargs):
        """
        Go through a directory, extracting files from it

        Parameters
        ----------
        directory : str
            The directory to parse through

        **kwargs: keyword arguments
            tetrode_list : list
                list of tetrodes to consider
            data_extension : str default .set
            cluster_extension : str default .cut
            pos_extension : str default .txt
            lfp_extension : str default .eeg
        Returns
        -------
        None

        """
        default_tetrode_list = [1, 2, 3, 4, 9, 10, 11, 12]
        tetrode_list = kwargs.get("tetrode_list", default_tetrode_list)
        data_extension = kwargs.get("data_extension", ".set")
        cluster_extension = kwargs.get("cluster_extension", ".cut")
        pos_extension = kwargs.get("pos_extension", ".txt")
        lfp_extension = kwargs.get("lfp_extension", ".eeg")

        files = get_all_files_in_dir(directory, data_extension)
        txt_files = get_all_files_in_dir(directory, pos_extension)
        for filename in files:
            filename = filename[:-len(data_extension)]
            for tetrode in tetrode_list:
                spike_name = filename + '.' + str(tetrode)
                cut_name = filename + '_' + str(tetrode) + cluster_extension
                lfp_name = filename + lfp_extension
                # Don't consider files that have not been clustered
                if not os.path.isfile(os.path.join(directory, cut_name)):
                    logging.info(
                        "Skipping this - no cluster file named {}".format(cut_name))
                    continue

                for fname in txt_files:
                    if fname[:len(filename)] == filename:
                        pos_name = fname
                        break
                else:
                    logging.info(
                        "Skipping this - no position file for {}".format(filename))
                    continue

                self.add_files(NDataContainer.EFileType.Spike, [spike_name])
                self.add_files(NDataContainer.EFileType.Position, [pos_name])
                self.add_files(NDataContainer.EFileType.LFP, [lfp_name])
        self.set_units()

    def merge(self, indices, force_equal_units=True):
        """
        Merge the data from multiple indices together into the first index.

        WORK IN PROGRESS - ONLY FUNCTIONS FOR POSITIONS AND SPIKES CURRENTLY
        Only call this after loading the data, and not while loading on the fly

        Parameters
        ----------
        indices: list
            The list of indices in the data to merge together
        force_equal_units:
            The merged indexes must have the same unit numbers available

        Returns
        -------
        The merged data point

        """
        if self._load_on_fly:
            logging.error("Don't call merge when loading on the fly")
            return

        target_index = indices[0]

        data_to_merge = []
        target_data = self.get_data(target_index)
        for idx in indices[1:]:
            data = self.get_data(idx)
            data_to_merge.append(data)

        units1 = self.get_units(target_index)
        for idx, data in zip(indices[1:], data_to_merge):
            units2 = self.get_units(idx)
            if force_equal_units and (not units1 == units2):
                logging.error(
                    "Can't merge files with unequal units\n" +
                    "Units are {} , {}".format(units1, units2))
                return

            # Merge the spikes based on times (waveforms not done yet)
            new_spike_times = (
                data.spike.get_timestamp() +
                target_data.spike.get_duration())
            new_duration = (
                target_data.spike.get_duration() +
                data.spike.get_duration())
            new_tags = data.spike.get_unit_tags()

            # Merge the spatial information based on times
            new_spat_times = (
                data.spatial._time +
                target_data.spike.get_duration())
            new_pos_x = data.spatial._pos_x
            new_pos_y = data.spatial._pos_y
            new_direction = data.spatial._direction
            new_speed = data.spatial._speed

            target_data.spike._timestamp = np.append(
                target_data.spike._timestamp, new_spike_times)
            target_data.spike._unit_Tags = np.append(
                target_data.spike._unit_Tags, new_tags)
            target_data.spike._set_duration(new_duration)

            target_data.spatial._time = np.append(
                target_data.spatial._time, new_spat_times)
            # NB this may not work properly due to different borders
            target_data.spatial._pos_x = np.append(
                target_data.spatial._pos_x, new_pos_x)
            target_data.spatial._pos_y = np.append(
                target_data.spatial._pos_y, new_pos_y)
            target_data.spatial._direction = np.append(
                target_data.spatial._direction, new_direction)
            target_data.spatial._speed = np.append(
                target_data.spatial._speed, new_speed)

        self._container[target_index] = target_data

        for idx in indices[1:]:
            self._container.pop(idx)
            self._units.pop(idx)
            indices[1:] = [a - 1 for a in indices[1:]]
        self._unit_count = self._count_num_units()

        self._container[target_index].set_unit_no(
            self.get_units(target_index)[0])
        return self.get_data(target_index)

    def subsample(self, key):
        """
        Return a subsample of the original data collection.

        This subsample is not a reference, but a deep copy

        Parameters
        ----------
        key : Slice or int
            How to sample the original collection

        Returns
        -------
        NDataContainer
            The deep copied subsample

        """
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
        Sort the units in the collection based on the place field centroid.

        Parameters
        ----------
        should_sort_list: list
            Optional list of boolean values indicating what objects
        mode: str
            "horizontal" or "vertical", indicating what axis to sort on.

        Returns
        -------
        None

        """
        if mode == "vertical":
            h = 1
        elif mode == "horizontal":
            h = 0
        else:
            logging.error(
                "NDataContainer: "
                + "Only modes horizontal and vertical are supported")

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
                    zip(centroids, self.get_units()[idx]),
                    key=lambda pair: pair[0][h])]

    # Methods from here on should be for private class use
    def _load_all_data(self):
        """Intended private function which loads all the data."""
        if self._load_on_fly:
            logging.error(
                "Don't load all the data in container if loading on the fly")
        for key, vals in self.get_file_dict().items():
            for idx, _ in enumerate(vals):
                if idx >= self.get_num_data():
                    self.add_data(NData())

            for idx, descriptor in enumerate(vals):
                self._load(key, descriptor, idx=idx)

    def _load(self, key, descriptor, idx=None, ndata=None):
        """
        Intended private function which loads data for a specific filetype.

        The NData object loaded into is either passed in, or found by idx.

        Parameters
        ----------
        key : str
            "Spike", "Position", or "LFP", which filetype to load
        descriptor : tuple
            (filename, objectname, system) tuple
        idx : int
            Optional parameter to get corresponding data from _collection
        ndata : NData
            Optional parameter to allow passing in an ndata object to load to

        Returns
        -------
        None

        """
        if ndata is None:
            ndata = self.get_data(idx)
        key_fn_pairs = {
            "Spike": [
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

        if key == "Position" and self._share_positions and idx != 0:
            if self._load_on_fly:
                ndata.spatial = self._last_data_pt[1].spatial
            else:
                ndata.spatial = self.get_data(0).spatial
            return

        if filename is not None:
            key_fn_pairs[key][0](filename)
            key_fn_pairs[key][2]()

    def __repr__(self):
        """Return a string representation of the collection."""
        string = "NData Container Object with {} objects:\nFiles are {}\nUnits are {}\nSet to Load on Fly? {}".format(
            self.get_num_data(), self.get_file_dict(), self.get_units(), self._load_on_fly)
        return string

    def __getitem__(self, index):
        """Return the data object with corresponding unit at index."""
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
        """Return the number of units in the collection."""
        counts = self._unit_count
        if counts == 0:
            counts = [1 for _ in range(len(self._container))]
        return sum(counts)

    def _count_num_units(self):
        """Intended private function to count units in the collection."""
        counts = []
        for unit_list in self.get_units():
            counts.append(len(unit_list))
        return counts

    def _index_to_data_pos(self, index):
        """
        Intended private function to turn an index into a tuple indices.

        Parameters
        ----------
        index : int
            The unit index to convert to a data index and unit index for that

        Returns
        -------
        tuple
            (data collection index, unit index for this data object)

        """
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
