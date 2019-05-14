import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_plot as nc_plot


def main(dir):
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir)
    container.setup()
    ndata = container[0]
    phases, times, positions = ndata.phase_at_spikes()
    dim_pos = positions[0]
    histo_vals = np.histogram2d(
        dim_pos, phases, bins=[10, 180])
    print(histo_vals)
    plt.hist2d(dim_pos, phases, bins=[10, 180])
    plt.savefig("out.png")


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\recording_example'
    main(dir)
