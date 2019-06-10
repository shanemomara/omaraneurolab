import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

import csv
import os
from copy import copy


from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_containeranalysis import place_cell_summary
from neurochat.nc_utils import make_dir_if_not_exists, log_exception, oDict


# TODO could do many in this
def save_results_to_csv(filename, in_dicts):
    names = in_dicts[0].keys()
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=names)
            writer.writeheader()
            for in_dict in in_dicts:
                writer.writerow(in_dict)

    except Exception as e:
        log_exception(e, "When {} saving to csv".format(filename))


def main(in_dir, tetrode_list, analysis_flags):
    # Load files from dir in tetrodes x, y, z
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(in_dir, tetrode_list=tetrode_list)
    container.setup()
    spike_names = container.get_file_dict()["Spike"]

    # show summary of place
    if analysis_flags[0]:
        place_cell_summary(container)

    # do burst analysis
    if analysis_flags[1]:
        _results = []
        out_dir = os.path.join(in_dir, "nc_results")
        for i, ndata in enumerate(container):
            split_idx = container._index_to_data_pos(i)
            name = spike_names[split_idx[0]][0]
            parts = os.path.basename(name).split(".")
            end_name = parts[0] + "_burst" + ".csv"
            out_name = os.path.join(
                out_dir, end_name)
            make_dir_if_not_exists(out_name)
            note_dict = oDict()
            note_dict["Tetrode"] = int(parts[1])
            note_dict["Unit"] = ndata.get_unit_no()
            ndata.update_results(note_dict)
            ndata.burst()
            result = copy(ndata.get_results())
            _results.append(result)
        save_results_to_csv(out_name, _results)

    # # can ignore above if multi run
    # if analysis_flags[2]:


if __name__ == "__main__":
    in_dir = r'C:\Users\smartin5\Recordings\11092017'
    tetrode_list = [1, 2, 3, 4, 5, 6, 7, 8]
    analysis_flags = [False, True, True]
    main(in_dir, tetrode_list, analysis_flags)
