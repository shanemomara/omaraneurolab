import sys
import os
import argparse
import matplotlib.pyplot as plt

sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')
try:
    import neurochat.nc_plot as nc_plot
    from neurochat.nc_utils import butter_filter, make_dir_if_not_exists
    from neurochat.nc_data import NData
except Exception as e:
    print("Could not import neurochat modules with error {}".format(e))


def plot_lfp_signal(lfp, lower, upper, out_name, filt=True, nsamples=None):
    fs = lfp.get_sampling_rate()

    if nsamples is None:
        nsamples = lfp.get_total_samples()

    if filt:
        filtered_lfp = butter_filter(
            lfp.get_samples(), fs, 10,
            lower, upper, 'bandpass')
    else:
        filtered_lfp = lfp.get_samples()

    plt.plot(
        lfp.get_timestamp()[0:nsamples],
        filtered_lfp[0:nsamples], color='k')
    plt.savefig(out_name)
    plt.close()


def lfp_power(new_data, i, max_f, in_dir, prefilt=False):
    # 1.6 or 2 give similar
    filtset = [10, 1.5, max_f, 'bandpass']

    new_data.bandpower_ratio(
        [5, 11], [1.5, 4], 1.6, prefilt=prefilt,
        first_name="Theta", second_name="Delta",
        filtset=filtset)

    graphData = new_data.spectrum(
        window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
        filtset=filtset, fmax=max_f, db=False, tr=False)
    fig = nc_plot.lfp_spectrum(graphData)
    fig.savefig(os.path.join(in_dir, "spec" + str(i) + ".png"))

    graphData = new_data.spectrum(
        window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
        filtset=filtset, fmax=max_f, db=True, tr=True)
    fig = nc_plot.lfp_spectrum_tr(graphData)
    fig.savefig(os.path.join(in_dir, "spec_tr" + str(i) + ".png"))

    voltages = new_data.lfp.get_samples()
    times = new_data.lfp.get_timestamp()
    fig, ax = plt.subplots()
    ax.hist2d(times, voltages, bins=100)
    fig.savefig(os.path.join(in_dir, "time_volt" + str(i) + ".png"))
    plt.close("all")

    return new_data.get_results()


def main(in_dir, lfp_file, analysis_flags, parsed):
    max_lfp = parsed.max_freq
    filt = parsed.prefilt

    f_path = os.path.join(in_dir, lfp_file)
    ndata = NData()
    ndata.lfp.load(f_path)

    print("Saving results to {}".format(
        os.path.join(in_dir, "nc_results")))
    make_dir_if_not_exists(os.path.join(in_dir, "nc_results", "dummy.txt"))

    if analysis_flags[0]:
        out_name = os.path.join(in_dir, "nc_results", "full_signal.png")
        plot_lfp_signal(
            ndata.lfp, 1.5, max_lfp, out_name, filt=False)
        out_name = os.path.join(in_dir, "nc_results", "full_signal_filt.png")
        plot_lfp_signal(
            ndata.lfp, 1.5, max_lfp, out_name, filt=True)

    if analysis_flags[1]:
        splits = [
            (0, 600), (600, 1200),
            (1200, ndata.lfp.get_duration()),
            (0, ndata.lfp.get_duration())]

        for i, split in enumerate(splits):
            new_data = ndata.subsample(split)
            results = lfp_power(
                new_data, i, max_lfp,
                os.path.join(in_dir, "nc_results"),
                prefilt=filt)
            print("For {} results are {}".format(i, results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument(
        "--prefilt", "-pf", action="store_true",
        help="Should pre filter lfp before power and spectral analysis")
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
    # main(ndata, [True, True], parsed)

    # print("\nDealing with bt\n")
    # in_dir = r'C:\Users\smartin5\Recordings\ER\25062019-bt'
    # lfp_file = "25062019-bt-LFP.eeg"
    # file = os.path.join(in_dir, lfp_file)
    # ndata = NData()
    # ndata.lfp.load(file)
    # main(ndata, [True, True], parsed)

    # in_dir = r'C:\Users\smartin5\Recordings\ER\23062019-bt'
    # lfp_file = "23062019-bt-LFP.eeg13"
    # file = os.path.join(in_dir, lfp_file)
    # ndata = NData()
    # ndata.lfp.load(file)
    # main(ndata, [True, True], parsed)

    in_dir = r'C:\Users\smartin5\Recordings\ER\23072019-bt'
    lfp_file = "23072019-bt-LFPsaline.eeg"
    main(in_dir, lfp_file, [True, True], parsed)
