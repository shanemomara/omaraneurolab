import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

from neurochat.nc_datacontainer import NDataContainer


def main(in_dir, tetrode_list):
    # Load files from dir in tetrodes x, y, z
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir, tetrode_list=tetrode_list)
    container.setup()
    spike_names = container.get_file_dict()["Spike"]
    # show summary of place
    # do burst analysis
    # can ignore above if multi run


if __name__ == "__main__":
    in_dir = r'C:\Users\smartin5\Repos\myNeurochat'
    tetrode_list = [1, 2, 3, 4, 5, 6, 7, 8]
    main(in_dir, tetrode_list)
