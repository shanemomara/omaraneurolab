import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_containeranalysis as nca

def main(dir):
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir)
    nca.place_cell_summary(dir, container)
    print("Yay!")

if __name__ == "__main__":
    dir = r'C:\Users\smartin5\recording_example'
    main(dir)