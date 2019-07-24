import sys
import os
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')
try:
    import neurochat.nc_plot as nc_plot
    from neurochat.nc_utils import butter_filter, make_dir_if_not_exists
    from neurochat.nc_data import NData
except Exception as e:
    print("Could not import neurochat modules with error {}".format(e))


def plot_lfp_signal(
        lfp, lower, upper, out_name,
        filt=True, nsamples=None, nsplits=3):
    fs = lfp.get_sampling_rate()

    if nsamples is None:
        nsamples = lfp.get_total_samples()

    if filt:
        filtered_lfp = butter_filter(
            lfp.get_samples(), fs, 10,
            lower, upper, 'bandpass')
    else:
        filtered_lfp = lfp.get_samples()

    fig, axes = plt.subplots(3, 1, figsize=(16, 4))
    for i in range(nsplits):
        start = i * (nsamples // nsplits)
        end = (i + 1) * (nsamples // nsplits)
        axes[i].plot(
            lfp.get_timestamp()[start:end],
            filtered_lfp[start:end], color='k')
        axes[i].set_ylim([-0.7, 0.7])
    plt.tight_layout()
    fig.savefig(out_name)
    plt.close(fig)


def raw_lfp_power(lfp, splits, lower, upper, prefilt=False):
    """
    This can be used to get the power before splitting up the signal.

    Minor differences between this and filtering after splitting
    """

    lfp_samples = lfp.get_samples()
    fs = lfp.get_sampling_rate()
    if prefilt:
        lfp_samples = butter_filter(
            lfp_samples, fs, 10, lower, upper, 'bandpass')

    results = OrderedDict()
    for i, (l, u) in enumerate(splits):
        start_idx = int(l * fs)
        end_idx = int(u * fs)
        sample = lfp_samples[start_idx:end_idx]
        power = np.sum(np.square(sample)) / sample.size
        results["Raw power {}".format(i)] = power
    return results


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

    lfp_samples = new_data.lfp.get_samples()
    fs = new_data.lfp.get_sampling_rate()
    if prefilt:
        lfp_samples = butter_filter(
            lfp_samples, fs, 10, 1.5, max_f, 'bandpass')

    return new_data.get_results()


def lfp_distribution(filename, upper, out_dir, prefilt=False):
    lfp_s_array = []
    lfp_t_array = []
    data = NData()
    avg = OrderedDict()

    for j in range(4):
        avg["Avg power {}".format(j)] = 0

    for i in range(32):
        end = str(i + 1)
        if end == "1":
            load_loc = filename
        else:
            load_loc = filename + end
        data.lfp.load(load_loc)
        lfp_samples = data.lfp.get_samples()
        fs = data.lfp.get_sampling_rate()
        if prefilt:
            lfp_samples = butter_filter(
                lfp_samples, fs, 10, 1.5, upper, 'bandpass')
        lfp_s_array.append(lfp_samples)
        lfp_t_array.append(data.lfp.get_timestamp())

        splits = [
            (0, 600), (600, 1200),
            (1200, data.lfp.get_duration()),
            (0, data.lfp.get_duration())]
        p_result = raw_lfp_power(
            data.lfp, splits, 1.5, upper, prefilt=prefilt)
        for j in range(len(splits)):
            avg["Avg power {}".format(j)] += (
                p_result["Raw power {}".format(j)] / 32)

    samples = np.concatenate(lfp_s_array)
    times = np.concatenate(lfp_t_array)

    fig, ax = plt.subplots()
    h = ax.hist2d(times, samples, bins=100)
    fig.colorbar(h[3])
    fig.savefig(os.path.join(out_dir, "dist.png"))

    return avg


def main(parsed):
    max_lfp = parsed.max_freq
    filt = not parsed.nofilt
    loc = parsed.loc
    eeg_num = parsed.eeg_num
    if not loc:
        print("Please pass a file in through CLI")
        exit(-1)

    if eeg_num != "1":
        load_loc = loc + eeg_num
    else:
        load_loc = loc

    in_dir = os.path.dirname(load_loc)
    ndata = NData()
    ndata.lfp.load(load_loc)
    out_dir = os.path.join(in_dir, "nc_results")

    if ndata.lfp.get_duration() == 0:
        print("Failed to correctly load lfp at {}".format(
            load_loc))
        exit(-1)

    print("Saving results to {}".format(out_dir))
    make_dir_if_not_exists(os.path.join(out_dir, "dummy.txt"))

    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        out_name = os.path.join(out_dir, "full_signal.png")
        plot_lfp_signal(
            ndata.lfp, 1.5, max_lfp, out_name, filt=False)
        out_name = os.path.join(
            in_dir, "nc_results", "full_signal_filt.png")
        plot_lfp_signal(
            ndata.lfp, 1.5, max_lfp, out_name, filt=True)

        splits = [
            (0, 600), (600, 1200),
            (1200, ndata.lfp.get_duration()),
            (0, ndata.lfp.get_duration())]

        p_results = raw_lfp_power(
            ndata.lfp, splits, 1.5, max_lfp, prefilt=filt)
        print("Power results are {}".format(p_results))
        f.write(str(p_results))

        result = lfp_distribution(loc, max_lfp, out_dir, prefilt=filt)
        print(result)
        f.write(str(result))

        for i, split in enumerate(splits):
            new_data = ndata.subsample(split)
            results = lfp_power(
                new_data, i, max_lfp, out_dir, prefilt=filt)
            print("For {} results are {}".format(i, results))
            f.write("{}: {}\n".format(i, results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument(
        "--nofilt", "-nf", action="store_true",
        help="Should pre filter lfp before power and spectral analysis")
    parser.add_argument(
        "--max_freq", "-mf", type=int, default=40,
        help="The maximum lfp frequency to consider"
    )
    parser.add_argument(
        "--loc", type=str, help="Lfp file location"
    )
    parser.add_argument(
        "--eeg_num", "-en", type=str, help="EEG number", default="1"
    )
    parsed = parser.parse_args()

    main(parsed)
