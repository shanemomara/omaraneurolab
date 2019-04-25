import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_containeranalysis as nca
import neurochat.nc_plot as nc_plot
from matplotlib.pyplot import savefig


def main(dir):
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir)
    for style in ["digitized", "interpolated", "contour"]:
        nc_plot.loc_rate(
            container[0].place(),
            pixel=3,
            chop_bound=0,
            filttype='g', filtsize=5, style=style,
            levels=5)
        savefig("result_place" + style + ".png")


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\recording_example'
    main(dir)
