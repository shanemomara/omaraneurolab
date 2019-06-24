import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')
from load_example_data import load_data

import neurochat.nc_plot as nc_plot
from matplotlib.pyplot import savefig


def main(dir):
    ndata = load_data()
    new_data = ndata
    new_data.bandpower_ratio(
        [5, 11], [1.5, 4], 2, relative=False,
        first_name="Theta", second_name="Delta")
    print(new_data.get_results())

    graphData = new_data.spectrum(
        window=2, noverlap=1, nfft=500, ptype='psd', prefilt=False,
        filtset=[10, 1.5, 40, 'bandpass'], fmax=40, db=False, tr=False)
    fig = nc_plot.lfp_spectrum(graphData)
    fig.savefig("spec.png")
    graphData = new_data.spectrum(
        window=2, noverlap=1, nfft=500, ptype='psd', prefilt=False,
        filtset=[10, 1.5, 40, 'bandpass'], fmax=40, db=True, tr=True)
    fig = nc_plot.lfp_spectrum_tr(graphData)
    fig.savefig("spec_tr.png")


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\recording_example'
    main(dir)
