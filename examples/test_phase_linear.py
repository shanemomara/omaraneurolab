import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')
import os

import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_data import NData
from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_plot as nc_plot
from neurochat.nc_utils import make_dir_if_not_exists


def main(dir):
    save_dir = os.path.join(dir, "plots", "phase")

    # spike_file = os.path.join(dir, "060319D_LCA1_linear_15min.2")
    # pos_file = os.path.join(dir, "060319D_LCA1_linear_15min.txt")
    # lfp_file = os.path.join(dir, "060319D_LCA1_linear_15min.eeg")
    # unit_no = 6
    # ndata = NData()
    # ndata.set_spike_file(spike_file)
    # ndata.set_spatial_file(pos_file)
    # ndata.set_lfp_file(lfp_file)
    # ndata.load()
    # ndata.set_unit_no(unit_no)
    # results = ndata.phase_at_spikes(
    #     should_filter=True)
    # positions = results["positions"]
    # phases = results["phases"]
    # good_place = results["good_place"]
    # directions = results["directions"]
    # co_ords = {}
    # co_ords["north"] = np.nonzero(
    #     (45 <= directions) &
    #     (directions < 135))
    # co_ords["south"] = np.nonzero(
    #     (225 <= directions) &
    #     (directions <= 315))
    # if (phases.size != 0) and good_place:
    #     for direction in "north", "south":
    #         dim_pos = positions[1][co_ords[direction]]
    #         directional_phases = phases[co_ords[direction]]
    #         fig, ax = plt.subplots()
    #         ax.scatter(dim_pos, directional_phases)
    #         # ax.hist2d(dim_pos, directional_phases, bins=[10, 90])
    #         parts = os.path.basename(spike_file[0]).split(".")
    #         end_name = (
    #             parts[0] + "_unit" + str(ndata.get_unit_no()) + "_" +
    #             direction + ".png")
    #         out_name = os.path.join(save_dir, end_name)
    #         make_dir_if_not_exists(out_name)
    #         fig.savefig(out_name)
    #         plt.close(fig)

    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir)
    container.setup()
    spike_names = container.get_file_dict()["Spike"]

    for i, ndata in enumerate(container):
        name = spike_names[container._index_to_data_pos(i)[0]]
        results = ndata.phase_at_spikes(
            should_filter=True)
        positions = results["positions"]
        phases = results["phases"]
        good_place = results["good_place"]
        directions = results["directions"]
        co_ords = {}
        co_ords["north"] = np.nonzero(
            (45 <= directions) &
            (directions < 135))
        co_ords["south"] = np.nonzero(
            (225 <= directions) &
            (directions <= 315))
        if (phases.size != 0) and good_place:
            for direction in "north", "south":
                dim_pos = positions[1][co_ords[direction]]
                directional_phases = phases[co_ords[direction]]
                fig, ax = plt.subplots()
                ax.scatter(dim_pos, directional_phases)
                # ax.hist2d(dim_pos, directional_phases, bins=[10, 90])
                parts = os.path.basename(name[0]).split(".")
                end_name = (
                    parts[0] + "_unit" + str(ndata.get_unit_no()) + "_" +
                    direction + ".png")
                out_name = os.path.join(save_dir, end_name)
                make_dir_if_not_exists(out_name)
                fig.savefig(out_name)
                plt.close(fig)


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\Recordings\LCA1'
    main(dir)
