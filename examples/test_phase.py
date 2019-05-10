import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

import numpy as np

from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_plot as nc_plot


def main(dir):
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir)
    container.setup()
    ndata = container[0]
    # phase_data = ndata.phase_dist()
    # figs = nc_plot.spike_phase(phase_data)
    # for i, fig in enumerate(figs):
    #     fig.savefig("figure{}.png".format(i))

    # TODO for some reason there is slight difference in the shapes
    phases, times, positions = ndata.phase_at_spikes()
    histo_vals = np.histogram2d(positions[0], phases)
    print(histo_vals)


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\recording_example'
    main(dir)
