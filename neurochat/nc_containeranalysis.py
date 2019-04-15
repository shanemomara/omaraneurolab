"""
This module contains analysis functions for NDataContainer objects.

@author: Sean Martin; martins7 at tcd dot ie
"""

import logging
from itertools import compress
from math import floor, ceil

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_data import NData
from neurochat.nc_clust import NClust
from neurochat.nc_utils import smooth_1d, find_true_ranges
from neurochat.nc_utils import butter_filter
from neurochat.nc_utils import find_peaks
from neurochat.nc_utils import window_rms
from neurochat.nc_utils import distinct_window_rms
from neurochat.nc_plot import replay_summary

import numpy as np
from scipy.optimize import linear_sum_assignment

def spike_positions(collection, should_sort=True, mode="vertical"):
    """
    Get the spike positions for a number of units.

    Parameters
    ----------
    collection : NDataContainer or NData list or NData object


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
    """
    Smooth all the speed data in the collection.

    Parameters
    ----------
    collection : NDataContainer
        Container to get the information from
    allows_multiple : bool
        Allow smoothing multiple times, default False

    Returns
    -------
    None

    """
    if collection._smoothed_speed and not allow_multiple:
        logging.warning(
            "NDataContainer has already been speed smoothed, not smoothing")

    for i in range(collection.get_num_data()):
        data = collection.get_data(i)
        data.smooth_speed()
        collection._smoothed_speed = True

def spike_times(collection, filter_speed=False, **kwargs):
    """
    Return a list of all spike times in the collection.

    Parameters
    ----------
    collection : NDataContainer or NData
        Either the container or data object to get spike times from
    filter_speed : bool
        If true, don't consider spike times when the rat is non moving
    kwargs
        should_smooth : bool
            Smooth the speed data if true
        ranges : list
            List of tuples indicating time ranges to get spikes in

    Returns
    -------
    list
        The list of spike times if collection is NData
        or a 2d list containing a list of times for each collection item

    """
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
    For each recording in the collection, detect periods of MUA.

    WORK IN PROGRESS, NEEDS TO BE MODIFIED BEFORE REAL USE
    Do not pass ranges and filter speed, only pass one.

    Parameters
    ----------
    collection : NDataContainer
        The collection of units to detect Muti unit activity.
    filter_speed : bool
        If true, don't consider spike times when the rat is non moving
    should_smooth : bool
        If true, smooth the histogram of spike counts
    kwargs
        ranges : tuple
            A range of times to consider data in
        bin_length : float
            The length of a time bin in seconds
        min_range : float
            The minimum length of high activity in seconds to be considered MUA
        mode : int
            1 or 2, the type of analysis to use
            1 works based off a

    Returns
    -------
    dict
        hists, bins, bin_centres, mua

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

def count_units_in_bins(collection, bin_length, in_range):
    """
    Count the amount of units that fire in certain bins.

    Should only be used on a collection of units

    Parameters
    ----------
    collection : NDataCollection
        The collection of units to count over
    bin_length : float
        The length of the bins
    in_range : tuple
        The range of time to count units over

    Returns
    -------
    tuple
        (unit counts in bins, bin_centres)

    """
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

def evaluate_clusters(collection, idx1, idx2, set_units=False):
    """
    Find which units are closest in terms of clustering.

    Uses the Hungarian (Munkres) cost optimisation based on Hellinger distance
    between the clusters.

    Parameters
    ----------
    collection : NDataCollection
        The collection to find similar cells in
    idx1 : int
        The first data point in the collection to consider
    idx2 : int
        The second data point in the collection to consider

    Returns
    -------
    dict
        For each unit in data[idx1] (key), a tuple consisting of the
        best matching unit from data[idx2] and the distance for this (val)

    """
    nclust1 = NClust()
    nclust2 = NClust()

    sub_col1 = collection.subsample(idx1)
    info1 = sub_col1.get_file_dict()["Spike"][0]
    nclust1.load(info1[0], info1[2])

    sub_col2 = collection.subsample(idx2)
    info2 = sub_col2.get_file_dict()["Spike"][0]
    nclust2.load(info2[0], info2[2])

    distance_shape = (
        len(sub_col1.get_units()[0]), len(sub_col2.get_units()[0]))
    distances = np.zeros(shape=distance_shape)
    # Build a matrix of distances for each unit
    for idx1, unit1 in enumerate(sub_col1.get_units()[0]):
        for idx2, unit2 in enumerate(sub_col2.get_units()[0]):
            _, dh = nclust1.cluster_similarity(nclust2, unit1, unit2)
            distances[idx1, idx2] = dh
            # print(
            #     "{} {}: Bhattacharyya {} Hellinger {}".format(
            #         unit1, unit2, bc, dh))

    # Solve the linear sum assignment problem based on the Hungarian method
    solution = linear_sum_assignment(distances)
    best_matches = {}
    for i, j in zip(solution[0], solution[1]):
        best_matches[sub_col1.get_units()[0][i]] = (
            sub_col2.get_units()[0][j], distances[i, j])

    print("Best assignment is {}".format(best_matches))

    if set_units:
        run_units = [key for key in best_matches.keys()]
        best_units = [val[0] for _, val in best_matches.items()]
        collection.set_units([run_units, best_units])
    return best_matches

def replay(collection, run_idx, sleep_idx, **kwargs):
    """
    Run and sleep session comparison.

    Set the units of interest in the collection before running.

    Parameters
    ----------
    collection : NDataContainer
        The collection of run and sleep data
    run_idx : int
        The index in the collection for the run data
    sleep_idx : int
        The index in the collection for the sleep data
    kwargs:
        sample_rate : float

    Returns
    -------
    dict
        Graphical and numerical analysis results

    """
    results = {}

    # Parse the kwargs
    min_range = kwargs.get("min_range", 150)
    moving_thresh = kwargs.get("moving_thresh", 2.5)
    swr_lower = kwargs.get("swr_lower", 100)
    swr_higher = kwargs.get("swr_upper", 250)
    rms_window_size_ms = kwargs.get("rms_window_size_ms", 7)
    percentile = kwargs.get("percentile", 99)
    sorting_mode = kwargs.get("sorting_mode", "vertical")

    # Preprocessing step

    # Sort the run data spatially
    truth_arr = [False for i in range(collection.get_num_data())]
    truth_arr[run_idx] = True
    collection.sort_units_spatially(truth_arr, mode=sorting_mode)

    # Match up cells between the recordings
    evaluate_clusters(collection, run_idx, sleep_idx, set_units=True)

    # Find SWS, MUA, SWR
    sleep = collection.get_data(sleep_idx)
    sample_rate = sleep.lfp.get_sampling_rate()

    non_moving_periods = np.array(
            sleep.non_moving_periods(
                min_range=min_range, 
                moving_thresh=moving_thresh)
            ) * sample_rate 
    
    # TODO reconsider - for now get the longest consecutive period.
    sorted_periods = sorted(
        non_moving_periods, key=lambda x : x[1] - x[0], reverse=True)
    longest_sleep_period = sorted_periods[0]

    lfp_samples = sleep.get_samples()[
        int(longest_sleep_period[0]):int(longest_sleep_period[1])]
    lfp_times = sleep.lfp.get_timestamp()[
        int(longest_sleep_period[0]):int(longest_sleep_period[1])]
    filtered_lfp = butter_filter(
        lfp_samples, sample_rate, 10, swr_lower, swr_higher, 'bandpass')
    rms_window_size = floor((rms_window_size_ms / 1000) * sample_rate)
    #rms_envelope = window_rms(filtered_lfp, rms_window_size, mode="same")
    rms_envelope = distinct_window_rms(filtered_lfp, rms_window_size)
    p_val = np.percentile(rms_envelope, percentile)
     # Could also find peaks like this
     # peaks = (longest_sleep_period[0] + np.argwhere(
     #              rms_data > p99).flatten() * rms_window_size) / sample_rate

    # TODO this may still be too many peaks
    _, peaks = find_peaks(rms_envelope, thresh=p_val)
    #peaks = (longest_sleep_period[0] + peaks) / sample_rate
    peaks = (longest_sleep_period[0] + peaks * rms_window_size) / sample_rate

    sleep_subsample = collection.subsample(sleep_idx)

    unit_hist = count_units_in_bins(
        sleep_subsample, 1, 
        longest_sleep_period / sample_rate)

    raw_spike_times = spike_times(
        sleep_subsample, 
        ranges=[longest_sleep_period / sample_rate])

    results["mua histogram"] = unit_hist
    results["lfp times"] = lfp_times
    results["lfp samples"] = filtered_lfp
    results["swr times"] = peaks
    results["spike times"] = raw_spike_times
    results["sample rate"] = sample_rate
    results["num cells"] = len(sleep_subsample)

    # TODO finish this off and look at individual times of overlapping MUA 
    # and sleep etc
    
    return results