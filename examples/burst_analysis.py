import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

import csv
import os


from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_containeranalysis import place_cell_summary
from neurochat.nc_utils import make_dir_if_not_exists, log_exception


# TODO could do many in this
def save_results_to_csv(filename, in_dict):
    names = in_dict.keys()
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=names)
            writer.writeheader()
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
    # # do burst analysis
    # if analysis_flags[1]:
    out_dir = os.path.join(in_dir, "nc_results")
    for i, ndata in enumerate(container):
        iter_num = container._index_to_data_pos(i)
        name = spike_names[iter_num[0]][0]
        out_name = os.path.join(out_dir, name) + "_" + \
            str(iter_num[1]) + ".csv"
        make_dir_if_not_exists(out_name)
        ndata.burst()
        print(ndata.get_results())
        save_results_to_csv(out_name, ndata.get_results())

    # # can ignore above if multi run
    # if analysis_flags[2]:


if __name__ == "__main__":
    in_dir = r'C:\Users\smartin5\Recordings\11092017'
    tetrode_list = [1, 2, 3, 4, 5, 6, 7, 8]
    analysis_flags = [False, True, True]
    main(in_dir, tetrode_list, analysis_flags)
