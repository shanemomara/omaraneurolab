import sys
import os
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import entropy

try:
    import neurochat.nc_plot as nc_plot
    from neurochat.nc_utils import butter_filter, make_dir_if_not_exists
    from neurochat.nc_data import NData
except Exception as e:
    print("Could not import neurochat modules with error {}".format(e))


def plot_lfp_signal(
        lfp, lower, upper, out_name,
        filt=True, nsamples=None, offset=0,
        nsplits=3, figsize=(32, 4), ylim=(-0.4, 0.4)):
    fs = lfp.get_sampling_rate()

    if nsamples is None:
        nsamples = lfp.get_total_samples()

    if filt:
        filtered_lfp = butter_filter(
            lfp.get_samples(), fs, 10,
            lower, upper, 'bandpass')
    else:
        filtered_lfp = lfp.get_samples()

    fig, axes = plt.subplots(nsplits, 1, figsize=figsize)
    for i in range(nsplits):
        start = int(offset + i * (nsamples // nsplits))
        end = int(offset + (i + 1) * (nsamples // nsplits))
        if nsplits == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(
            lfp.get_timestamp()[start:end],
            filtered_lfp[start:end], color='k')
        ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(out_name, dpi=400)
    plt.close(fig)
    return filtered_lfp


def raw_lfp_power(lfp_samples, fs, splits, lower, upper, prefilt=False):
    """
    This can be used to get the power before splitting up the signal.

    Minor differences between this and filtering after splitting
    """

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
        [5, 11], [1.5, 4], 1.6, band_total=prefilt,
        first_name="Theta", second_name="Delta",
        totalband=filtset[1:3])

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


def lfp_entropy(
        lfp_samples, fs, splits, lower, upper, prefilt=False, etype="sample"):
    results = OrderedDict()

    if prefilt:
        lfp_samples = butter_filter(
            lfp_samples, fs, 10, lower, upper, 'bandpass')

    for i, (l, u) in enumerate(splits):
        start_idx = int(l * fs)
        end_idx = int(u * fs)
        sample = lfp_samples[start_idx:end_idx]
        if etype == "svd":
            et = entropy.svd_entropy(sample, order=3, delay=1)
        elif etype == "spectral":
            et = entropy.spectral_entropy(
                sample, 100, method='welch', normalize=True)
        elif etype == "sample":
            et = entropy.sample_entropy(sample, order=3)
        elif etype == "perm":
            et = entropy.perm_entropy(sample, order=3, normalize=True)
        else:
            print("Error: unrecognised entropy type {}".format(
                etype
            ))
            exit(-1)
        results["Entropy {}".format(i)] = et

    return results


def lfp_distribution(
        filename, upper, out_dir, splits,
        prefilt=False, get_entropy=False, return_all=False):
    lfp_s_array = []
    lfp_t_array = []
    data = NData()
    avg = OrderedDict()
    ent = OrderedDict()

    if return_all:
        power_arr = np.zeros(shape=(32, len(splits)))
    for j in range(len(splits)):
        avg["Avg power {}".format(j)] = 0
        if get_entropy:
            ent["Avg entropy {}".format(j)] = 0

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

        p_result = raw_lfp_power(
            lfp_samples, fs, splits, 1.5, upper, prefilt=False)
        for j in range(len(splits)):
            avg["Avg power {}".format(j)] += (
                p_result["Raw power {}".format(j)] / 32)
            power_arr[i, j] = p_result["Raw power {}".format(j)]
        if get_entropy:
            p_result = lfp_entropy(
                lfp_samples, fs, splits, 1.5, upper, prefilt=False)
            for j in range(len(splits)):
                ent["Avg entropy {}".format(j)] += (
                    p_result["Entropy {}".format(j)] / 32)

    samples = np.concatenate(lfp_s_array)
    times = np.concatenate(lfp_t_array)

    fig, ax = plt.subplots()
    h = ax.hist2d(times, samples, bins=100)
    fig.colorbar(h[3])
    fig.savefig(os.path.join(out_dir, "dist.png"))

    if return_all:
        return avg, ent, power_arr
    return avg, ent


def result_to_csv(result, out_dir):
    def arr_to_str(name, arr):
        out_str = name
        for val in arr:
            out_str = "{},{:2f}".format(out_str, val)
        return out_str

    with open(os.path.join(out_dir, "results.csv"), "w") as f:
        for key, val in result.items():
            out_str = arr_to_str(key, val.values())
            print(out_str)
            f.write(out_str + "\n")


def main(parsed):
    # Extract parsed args
    max_lfp = parsed.max_freq
    filt = not parsed.nofilt
    loc = parsed.loc
    eeg_num = parsed.eeg_num
    split_s = parsed.splits
    out_loc = parsed.out_loc
    every_min = parsed.every_min
    recording_dur = parsed.recording_dur
    get_entropy = parsed.get_entropy
    return_all = parsed.g_all

    # Do setup
    if not loc:
        print("Please pass a file in through CLI")
        exit(-1)

    if every_min:
        splits = [(60 * i, 60 * (i + 1)) for i in range(recording_dur)]
        splits.append((0, 600))
        splits.append((600, 1200))
        splits.append((1200, 1800))

    else:
        splits = []
        for i in range(len(split_s) // 2):
            splits.append((split_s[i * 2], split_s[i * 2 + 1]))

    splits.append((0, recording_dur * 60))
    if eeg_num != "1":
        load_loc = loc + eeg_num
    else:
        load_loc = loc

    in_dir = os.path.dirname(load_loc)
    ndata = NData()
    ndata.lfp.load(load_loc)
    out_dir = os.path.join(in_dir, out_loc)

    if ndata.lfp.get_duration() == 0:
        print("Failed to correctly load lfp at {}".format(
            load_loc))
        exit(-1)

    print("Saving results to {}".format(out_dir))
    make_dir_if_not_exists(os.path.join(out_dir, "dummy.txt"))

    # Plot signals
    out_name = os.path.join(out_dir, "full_signal.png")
    plot_lfp_signal(
        ndata.lfp, 1.5, max_lfp, out_name, filt=False)
    out_name = os.path.join(
        in_dir, out_dir, "full_signal_filt.png")
    filtered_lfp = plot_lfp_signal(
        ndata.lfp, 1.5, max_lfp, out_name, filt=True)
    if not filt:
        filtered_lfp = ndata.lfp

    # Calculate measures on this tetrode
    fs = ndata.lfp.get_sampling_rate()
    p_results = raw_lfp_power(
        filtered_lfp, fs, splits, 1.5, max_lfp, prefilt=False)

    if get_entropy:
        e_results = lfp_entropy(
            filtered_lfp, fs, splits[-4:], 1.5, max_lfp, prefilt=False)

    # Calculate measures over the dist
    d_result = lfp_distribution(
        loc, max_lfp, out_dir, splits[-4:],
        prefilt=filt, get_entropy=get_entropy, return_all=return_all)

    if get_entropy:
        results = {
            "power": p_results,
            "entropy": e_results,
            "avg_power": d_result[0],
            "avg_entropy": d_result[1]
        }
    else:
        results = {
            "power": p_results,
            "avg_power": d_result[0]
        }

    # Output the results
    result_to_csv(results, out_dir)
    return results, d_result[-1]

    # This is for theta and delta power
    # for i, split in enumerate(splits):
    #     new_data = ndata.subsample(split)
    #     results = lfp_power(
    #         new_data, i, max_lfp, out_dir, prefilt=filt)
    #     print("For {} results are {}".format(i, results))
    #     f.write("{}: {}\n".format(i, results))


def quick_test(load_loc):
    in_dir = os.path.dirname(load_loc)
    ndata = NData()
    ndata.lfp.load(load_loc)
    out_loc = "nc_signal"
    out_dir = os.path.join(in_dir, out_loc)
    out_name = os.path.join(out_dir, "full_signal.png")
    make_dir_if_not_exists(out_name)
    out_name = os.path.join(
        in_dir, out_dir, "full_signal_filt.png")
    filtered_lfp = plot_lfp_signal(
        ndata.lfp, 5, 11, out_name, filt=True,
        offset=ndata.lfp.get_sampling_rate() * 50,
        nsamples=ndata.lfp.get_sampling_rate() * 50,
        nsplits=1, ylim=(-0.3, 0.3),
        figsize=(20, 8))


def main_cfg():
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument(
        "--nofilt", "-nf", action="store_true",
        help="Should not pre filter lfp before power and spectral analysis")
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
    parser.add_argument(
        "--splits", "-s", nargs="*", type=int, help="Splits",
        default=[0, 600, 600, 1200, 1200, 1800]
    )
    parser.add_argument(
        "--out_loc", "-o", type=str, default="nc_results",
        help="Relative name of directory to store results in"
    )
    parser.add_argument(
        "--every_min", "-em", action="store_true",
        help="Calculate lfp every minute"
    )
    parser.add_argument(
        "--recording_dur", "-d", type=int, default=30,
        help="How long in minutes the recording lasted"
    )
    parser.add_argument(
        "--get_entropy", "-e", action="store_true",
        help="Calculate entropy"
    )
    parser.add_argument(
        "--g_all", "-a", action="store_true",
        help="Get all values instead of just average"
    )
    parsed = parser.parse_args()

    main(parsed)


def main_py():
    root = "C:\\Users\\smartin5\\Recordings\\ER\\"
    from types import SimpleNamespace
    args = SimpleNamespace(
        max_freq=40,
        nofilt=False,
        loc=os.path.join(root, "29072019-bt\\29072019-bt-1st30min-LFP.eeg"),
        eeg_num="13",
        splits=[],
        out_loc="firstt",
        every_min=False,
        recording_dur=1800,
        get_entropy=True,
        g_all=True
    )
    _, all1 = main(args)

    args = SimpleNamespace(
        max_freq=40,
        nofilt=False,
        loc=os.path.join(
            root, "30072019-bt\\30072019-bt-1st30min-LFP-DSER.eeg"),
        eeg_num="13",
        splits=[],
        out_loc="firstt",
        every_min=False,
        recording_dur=1800,
        get_entropy=True,
        g_all=True
    )
    _, all2 = main(args)

    args = SimpleNamespace(
        max_freq=40,
        nofilt=False,
        loc=os.path.join(root, "29072019-bt\\29072019-bt-last30min-LFP.eeg"),
        eeg_num="13",
        splits=[],
        out_loc="lastt",
        every_min=False,
        recording_dur=1800,
        get_entropy=True,
        g_all=True
    )
    _, all3 = main(args)

    args = SimpleNamespace(
        max_freq=40,
        nofilt=False,
        loc=os.path.join(
            root, "30072019-bt\\30072019-bt-last30min-LFP-DSER.eeg"),
        eeg_num="13",
        splits=[],
        out_loc="lastt",
        every_min=False,
        recording_dur=1800,
        get_entropy=True,
        g_all=True
    )
    _, all4 = main(args)

    difference = all2 - all1
    print(difference.flatten() * 1000)
    print("Mean difference is {:4f}".format(np.mean(difference)))
    print("Std deviation is {:4f}".format(np.std(difference)))

    difference = all4 - all3
    print(difference.flatten() * 1000)
    print("Mean difference is {:4f}".format(np.mean(difference)))
    print("Std deviation is {:4f}".format(np.std(difference)))
    return


if __name__ == "__main__":
    # main_cfg()
    # main_py()

    root = r"F:\cla-r-07022019"
    name = "cla-r-07022019-L2.eeg"
    load_loc = os.path.join(root, name)
    quick_test(load_loc)
    root = r"F:\cla-r-08022019"
    name = "cla-r-08022019-L2.eeg"
    load_loc = os.path.join(root, name)
    quick_test(load_loc)
