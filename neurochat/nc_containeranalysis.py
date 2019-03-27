import logging
from itertools import compress
from math import floor, ceil

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_data import NData
from neurochat.nc_clust import NClust
from neurochat.nc_utils import smooth_1d, find_true_ranges

import numpy as np

def spike_positions(collection, should_sort=True, mode="vertical"):
    """
    Get the spike positions for a number of units

    Parameters
    ----------
    collection : NDataContainer or NData list or NData object
        The collection to plot spike rasters over

    Returns
    -------
    positions : list of positions of the rat when the cell spiked
    """

    if isinstance(collection, NDataContainer) and should_sort:
        collection.sort_units_spatially(mode=mode)
    
    if isinstance(collection, NData):
        positions = collection.get_event_loc(collection.get_unit_stamp())[1]
        if mode == "vertical":
            positions = positions[1]
        elif mode == "horizontal":
            positions = positions[0]
        else:
            logging.error("nca: mode only supports vertical or horizontal")
    else:
        positions = []
        for data in collection:
            position = data.get_event_loc(data.get_unit_stamp())[1]
            if mode == "vertical":
                position = position[1]
            elif mode == "horizontal":
                position = position[0]
            else:
                logging.error("nca: mode only supports vertical or horizontal")
            positions.append(position)

    return positions

def smooth_speeds(collection, allow_multiple=False):
    if collection._smoothed_speed and not allow_multiple:
        logging.warning(
            "NDataContainer has already been speed smoothed, not smoothing")

    for i in range(collection.get_num_data()):
        data = collection.get_data(i)
        data.smooth_speed()
        collection._smoothed_speed = True

def spike_times(collection, filter_speed=False, **kwargs):
    should_smooth = kwargs.get("should_smooth", False)
    ranges = kwargs.get("ranges", None)

    if isinstance(collection, NData):
        if ranges is not None:
            time_data = collection.get_unit_stamps_in_ranges(ranges)
        elif filter_speed:
            ranges = collection.non_moving_periods(**kwargs)
            time_data = collection.get_unit_stamps_in_ranges(ranges)
        else:
            times = collection.get_unit_stamp()

    else:
        if should_smooth:
            smooth_speeds(collection)
            kwargs["should_smooth"] = False

        times = []
        for data in collection:
            if ranges is not None:
                time_data = data.get_unit_stamps_in_ranges(ranges)
            elif filter_speed:
                ranges = data.non_moving_periods(**kwargs)
                time_data = data.get_unit_stamps_in_ranges(ranges)
            else:
                time_data = data.get_unit_stamp()
            times.append(time_data)
    return times

def multi_unit_activity(
    collection, filter_speed=False, should_smooth=True, **kwargs):
    """
    For each recording in the collection, detect periods of MUA
    returns time ranges for each recording -
    Do not pass ranges and filter speed, only pass one
    bin length is in seconds
    min range is also in seconds
    """
    # TODO could expand later to show which units are contributing
    ranges = kwargs.get("ranges", [] if filter_speed else None)
    bin_length = kwargs.get("bin_length", 0.001)
    min_range = kwargs.get("min_range", 0.01)
    mode = kwargs.get("mode", 2)

    result = {'hists': [], 'bins': [], 'bin_centres': [], 'mua': []}
    for i in range(collection.get_num_data()):
        sub_collection = collection.subsample(i)
        if filter_speed:
            ranges.append(
                sub_collection.get_data(0).get_non_moving_periods(**kwargs))
        spike_times = np.array([])
        for data in sub_collection:
            if ranges is not None:
                new_spikes = np.array(data.get_unit_stamps_in_ranges(ranges))
            else:
                new_spikes = np.array(data.get_unit_stamp())
            spike_times = np.append(spike_times, new_spikes)

        # TODO only works for continuous interval as ranges currently
        if ranges is not None:
            bins = floor(
                (ranges[0][1] - ranges[0][0]) / bin_length)
        else:
            bins = floor(
                sub_collection.get_data(0).get_recording_time() / bin_length)
        flat_array = spike_times
        hist, new_bins = np.histogram(flat_array, bins)
        result['hists'].append(hist) 
        result['bins'].append(new_bins)
        bin_centres = [
            (new_bins[j + 1] + new_bins[j]) / 2 for j in range(len(hist))]
        result['bin_centres'].append(bin_centres)

    # TODO decide which mode is better
    if mode == 1:
        for i, hist in enumerate(result['hists']):
            if should_smooth:
                result['hists'][i] = smooth_1d(hist, filttype='g', filtsize=10)
        
            p95 = np.percentile(result['hists'][i], 95)
            result['mua'].append(
                find_true_ranges(
                    result['bin_centres'][i], 
                    result['hists'][i] > p95, 
                    min_range=min_range)
            )
    if mode == 2:
        for i, hist in enumerate(result['hists']):
            p99 = np.percentile(result['hists'][i], 99)
            large_vals = np.argwhere(result['hists'][i] > p99)
            corresponding_ranges = [
                (result['bins'][i][j], result['bins'][i][j+1])
                for j in large_vals]
            result['mua'].append(corresponding_ranges)
    return result

# Should only be used on a collection of units
def count_units_in_bins(collection, bin_length, in_range):
    num_bins = ceil(in_range[1] - in_range[0] / bin_length)
    arr = np.empty(shape=(len(collection), num_bins))
    for idx, data in enumerate(collection):
        # Check if the unit spikes in the bin
        hist_val, bins = np.histogram(
            data.get_unit_stamps_in_ranges([in_range]), 
            bins=num_bins, range=in_range)
        hist_val = np.clip(hist_val, 0, 1)
        arr[idx] = hist_val
    
    bin_centres = [(bins[j + 1] + bins[j]) / 2 for j in range(num_bins)]
    return np.sum(arr, axis=0), bin_centres

def evaluate_clusters(collection, idx1, idx2):
    nclust1 = NClust()
    nclust2 = NClust()

    sub_col1 = collection.subsample(idx1)
    info1 = sub_col1.get_file_dict()["Spike"][0]
    nclust1.load(filename=info1[0], system=info1[2])

    sub_col2 = collection.subsample(idx2)
    info2 = sub_col2.get_file_dict()["Spike"][0]
    nclust2.load(info2[0], info2[2])

    best_matches = {}
    for unit1 in sub_col1.get_units()[0]:
        best_bc, best_unit = 0, None
        for unit2 in sub_col2.get_units()[0]:
            bc, dh = nclust1.cluster_similarity(nclust2, unit1, unit2)
            print(
                "{} {}: Bhattacharyya {} Hellinger {}".format(
                    unit1, unit2, bc, dh))
            if bc > best_bc:
                best_bc, best_unit = bc, unit2
        best_matches[str(unit1)] = (best_unit, best_bc)
    return best_matches

