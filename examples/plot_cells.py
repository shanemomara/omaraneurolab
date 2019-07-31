import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, r'C:\Users\smartin5\Repos\MatheusNC')

import neurochat.nc_plot as nc_plot
from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_containeranalysis import place_cell_summary

container = NDataContainer(load_on_fly=True)
# container.add_axona_files_from_dir(
#     r"E:\Chapter6\6s_data_and_results\Data",
#     recursive=True,
#     verbose=False)
container.add_axona_files_from_dir(
    r"E:\\Chapter6\\6s_data_and_results\\Data",
    recursive=True)

container.setup()
print(container.string_repr(True))
place_cell_summary(container, dpi=200, out_dirname="nc_spat_plots")
# place_cell_summary(
#     container, dpi=200, out_dirname="nc_plots", filter_place_cells=False,
#     filter_low_freq=False)
