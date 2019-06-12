import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

import csv
import os
from copy import copy

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_containeranalysis import place_cell_summary
from neurochat.nc_utils import make_dir_if_not_exists, log_exception, oDict
import neurochat.nc_plot as nc_plot


def save_results_to_csv(filename, in_dicts):
    names = in_dicts[0].keys()
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=names)
            writer.writeheader()
            for in_dict in in_dicts:
                writer.writerow(in_dict)

    except Exception as e:
        log_exception(e, "When {} saving to csv".format(filename))


def visualise_spacing(N=61, start=5, stop=10000):
    # This is equivalent to np.exp(np.linspace)
    x1 = np.logspace(np.log10(start), np.log10(stop), N, base=10)
    y = np.zeros(N)
    plt.plot(x1, y, 'o')
    plt.ylim([-0.5, 1])
    print(x1)
    plt.show()


def log_isi(ndata, start=0.0005, stop=10, num_bins=60):
    isi_log_bins = np.linspace(
        np.log10(start), np.log10(stop), num_bins + 1)
    hist, _ = np.histogram(
        np.log10(np.diff(ndata.spike.get_unit_stamp())), bins=isi_log_bins, density=True)
    # return ndata.isi(bins=isi_log_bins, density=True), isi_log_bins
    return hist, isi_log_bins


def main(in_dir, tetrode_list, analysis_flags):
    # Load files from dir in tetrodes x, y, z
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(in_dir, tetrode_list=tetrode_list)
    container.setup()
    spike_names = container.get_file_dict()["Spike"]

    # Show summary of place
    if analysis_flags[0]:
        place_cell_summary(container)

    # Do numerical analysis
    if analysis_flags[1]:
        _results = []
        out_dir = os.path.join(in_dir, "nc_results")
        for i, ndata in enumerate(container):
            split_idx = container._index_to_data_pos(i)
            name = spike_names[split_idx[0]][0]
            parts = os.path.basename(name).split(".")
            end_name = parts[0] + "_burst" + ".csv"
            out_name = os.path.join(
                out_dir, end_name)
            make_dir_if_not_exists(out_name)
            note_dict = oDict()
            note_dict["Tetrode"] = int(parts[1])
            note_dict["Unit"] = ndata.get_unit_no()
            ndata.update_results(note_dict)
            ndata.wave_property()
            isi = ndata.isi()
            ndata.burst(burst_thresh=6)
            phase_dist = ndata.phase_dist()
            theta_index = ndata.theta_index()
            # theta_skip_index = ndata.theta_skip_index()
            ndata.bandpower_ratio(
                [5, 11], [1.5, 4], 1.6, relative=True,
                first_name="Theta", second_name="Delta")
            result = copy(ndata.get_results())
            _results.append(result)

            # Do graphical analysis
            if analysis_flags[2]:
                lfp_spectrum = ndata.spectrum()
                # use phase dist

        save_results_to_csv(out_name, _results)

        # Do PCA based analysis
    if analysis_flags[3]:
        ax1, fig1 = nc_plot._make_ax_if_none(None)
        ax2, fig2 = nc_plot._make_ax_if_none(None)
        isi_hist_matrix = np.empty((len(container), 60), dtype=float)
        for i, ndata in enumerate(container):
            split_idx = container._index_to_data_pos(i)
            res_isi, bins = log_isi(ndata)
            isi_hist_matrix[i] = res_isi
            bin_centres = bins[:-1] + np.mean(np.diff(bins))
            print(bins)
            ax1.plot(bin_centres, res_isi)
            ax1.set_xlim([-3, 1])
            ax1.set_xticks([-3, -2, -1, 0])
            ax1.axvline(x=np.log10(0.006))
            isi = ndata.isi()
            ax2.plot(isi['isiBinCentres'] / 1000, isi['isiHist'] / 1000)
            ax2.set_xlim([0.001, 1])
            ax2.set_xticks([0.001, 0.01, 0.1, 1])
        fig1.savefig("logisi.png", dpi=800)
        fig2.savefig("isi.png", dpi=800)
        pca = PCA(0.95)
        after_pca = pca.fit_transform(isi_hist_matrix)
        print(pca.n_components_)
        print(pca.explained_variance_ratio_)

        ax, fig = nc_plot._make_ax_if_none(None)
        dend = shc.dendrogram(shc.linkage(after_pca, method="ward"), ax=ax)
        fig.savefig("dendogram.png", dpi=800)


if __name__ == "__main__":
    in_dir = r'C:\Users\smartin5\Recordings\11092017'
    tetrode_list = [1, 2, 3, 4, 5, 6, 7, 8]
    analysis_flags = [False, False, False, True]
    main(in_dir, tetrode_list, analysis_flags)
