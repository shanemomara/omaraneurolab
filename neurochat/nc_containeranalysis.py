import logging
from itertools import compress

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_data import NData

def spike_positions(collection, should_sort=True, mode="vertical"):
    """
    Get the spike positions for a number of units

    Parameters
    ----------
    collection : NDataContainer or NData list or NData object
        The collection to plot spike rasters over

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The spike raster
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
            logging.error("nc_plot: mode only supports vertical or horizontal")
    else:
        positions = []
        for data in collection:
            position = data.get_event_loc(data.get_unit_stamp())[1]
            if mode == "vertical":
                position = position[1]
            elif mode == "horizontal":
                position = position[0]
            else:
                logging.error("nc_plot: mode only supports vertical or horizontal")
            positions.append(position)

    return positions

def spike_times(collection, filter_speed=False, **kwargs):
    if isinstance(collection, NData):
        if filter_speed:
            ranges = collection.non_moving_periods(**kwargs)
            time_data = collection.get_unit_stamps_in_ranges(ranges)
        else:
            times = collection.get_unit_stamp()

    else:
        times = []
        first = True
        for data in collection:
            if filter_speed:
                if first:
                    ranges = data.non_moving_periods(**kwargs)
                first = False
                time_data = data.get_unit_stamps_in_ranges(ranges)
            else:
                time_data = data.get_unit_stamp()
            times.append(time_data)
    return times