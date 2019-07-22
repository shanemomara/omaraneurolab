import sys
import os
import argparse
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')
from load_example_data import load_data

import neurochat.nc_plot as nc_plot
import matplotlib.pyplot as plt
from neurochat.nc_utils import butter_filter
from neurochat.nc_data import NData


def plot_lfp_signal(lfp, lower, upper, out_name, filt=True):
    fs = lfp.get_sampling_rate()

    if filt:
        filtered_lfp = butter_filter(
            lfp.get_samples(), fs, 10,
            lower, upper, 'bandpass')
    else:
        filtered_lfp = lfp.get_samples()

    plt.plot(lfp.get_timestamp(), filtered_lfp, color='k')
    plt.savefig(out_name)


def lfp_power(new_data, i, max_f, prefilt=False, relative=True):
    # 1.6 or 2 give similar
    filtset = [10, 1.5, max_f, 'bandpass']

    new_data.bandpower_ratio(
        [5, 11], [1.5, 4], 1.6, relative=relative, prefilt=prefilt,
        first_name="Theta", second_name="Delta",
        filtset=filtset)
    print("For {} results are {}".format(i, new_data.get_results()))

    graphData = new_data.spectrum(
        window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
        filtset=filtset, fmax=max_f, db=False, tr=False)
    fig = nc_plot.lfp_spectrum(graphData)
    fig.savefig("spec" + str(i) + ".png")
    graphData = new_data.spectrum(
        window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
        filtset=filtset, fmax=max_f, db=True, tr=True)
    fig = nc_plot.lfp_spectrum_tr(graphData)
    fig.savefig("spec_tr" + str(i) + ".png")
    plt.close("all")


def main(ndata, analysis_flags, parsed):
    max_lfp = parsed.max_freq
    print(max_lfp)
    filt = parsed.prefilt
    relative = parsed.relative
    if analysis_flags[0]:
        plot_lfp_signal(
            ndata.lfp, 1.5, max_lfp, "full_signal.png", filt=filt)

    if analysis_flags[1]:
        lfp_power(ndata, 4, max_lfp, prefilt=True, relative=relative)
        splits = [
            (0, 900),
            (900, 1800)]
        # (1200, 1800)]

        for i, split in enumerate(splits):
            new_data = ndata.subsample(split)
            lfp_power(new_data, i, max_lfp, prefilt=True, relative=relative)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument(
        "--prefilt", "-pf", action="store_true",
        help="Should pre filter lfp before power and spectral analysis")
    parser.add_argument(
        "--relative", "-r", action="store_true",
        help="Should band power be relative to total signal power"
    )
    parser.add_argument(
        "--max_freq", "-mf", type=int, default=40,
        help="The maximum lfp frequency to consider"
    )
    parsed = parser.parse_args()

    # print("Dealing with nt\n")
    # in_dir = r'C:\Users\smartin5\Recordings\ER\26062019-nt'
    # lfp_file = "26062019-nt-LFP.eeg"
    # file = os.path.join(in_dir, lfp_file)
    # ndata = NData()
    # ndata.lfp.load(file)
    # main(ndata, [True, True])

    # in_dir = r'C:\Users\smartin5\Recordings\ER\22062019-nt'
    # lfp_file = "22062019-LFP.eeg"
    # file = os.path.join(in_dir, lfp_file)
    # ndata = NData()
    # ndata.lfp.load(file)
    # main(ndata, [True, True])

    # print("\nDealing with bt\n")
    # in_dir = r'C:\Users\smartin5\Recordings\ER\25062019-bt'
    # lfp_file = "25062019-bt-LFP.eeg"
    # file = os.path.join(in_dir, lfp_file)
    # ndata = NData()
    # ndata.lfp.load(file)
    # main(ndata, [True, True])

    in_dir = r'C:\Users\smartin5\Recordings\ER\23062019-bt'
    lfp_file = "23062019-bt-LFP.eeg"
    file = os.path.join(in_dir, lfp_file)
    ndata = NData()
    ndata.lfp.load(file)
    main(ndata, [True, True], parsed)
