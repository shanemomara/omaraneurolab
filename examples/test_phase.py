import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')
import os

import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_data import NData
from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_plot as nc_plot


def main(dir):
    # container = NDataContainer(load_on_fly=True)
    # container.add_axona_files_from_dir(dir)
    # container.setup()
    # ndata = container[0]
    spike_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms.2")
    pos_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms_2.txt")
    lfp_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms.eeg")
    unit_no = 7
    ndata = NData()
    ndata.set_spike_file(spike_file)
    ndata.set_spatial_file(pos_file)
    ndata.set_lfp_file(lfp_file)
    ndata.load()
    ndata.set_unit_no(unit_no)
    results = ndata.phase_at_spikes(
        should_filter=True)
    boundary = results["boundary"]
    positions = results["positions"]
    phases = results["phases"]
    print(boundary)
    dim_pos = positions[1]
    plt.hist2d(dim_pos, phases, bins=[10, 180])
    plt.savefig("out.png")
    plt.scatter(dim_pos, phases)
    plt.savefig("outscatt.png")


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\recording_example'
    main(dir)
