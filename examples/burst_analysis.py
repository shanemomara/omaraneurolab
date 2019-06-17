import sys
sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')

import csv
import os
from copy import copy

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
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
        np.log10(np.diff(ndata.spike.get_unit_stamp())), bins=isi_log_bins, density=False)
    # return ndata.isi(bins=isi_log_bins, density=True), isi_log_bins
    return hist / ndata.spike.get_unit_stamp().size, isi_log_bins
    # return hist, isi_log_bins


def cell_classification_stats(in_dir, container, should_plot=False):
    _results = []
    out_dir = os.path.join(in_dir, "nc_results")
    spike_names = container.get_file_dict()["Spike"]
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

        if should_plot:
            plot_loc = os.path.join(
                in_dir, "nc_plots",
                parts[0] + parts[1] + str(ndata.get_unit_no()) + "phase.png")
            make_dir_if_not_exists(plot_loc)
            fig1, fig2, fig3 = nc_plot.spike_phase(phase_dist)
            fig2.savefig(plot_loc)
            plt.close("all")

    if should_plot:
        plot_loc = os.path.join(in_dir, "nc_plots", parts[0] + "lfp.png")
        make_dir_if_not_exists(plot_loc)

        lfp_spectrum = ndata.spectrum()
        fig = nc_plot.lfp_spectrum(lfp_spectrum)
        fig.savefig(plot_loc)
        plt.close(fig)

    save_results_to_csv(out_name, _results)


def calculate_isi_hist(container):
    ax1, fig1 = nc_plot._make_ax_if_none(None)
    isi_hist_matrix = np.empty((len(container), 60), dtype=float)
    for i, ndata in enumerate(container):
        res_isi, bins = log_isi(ndata)
        isi_hist_matrix[i] = res_isi
        bin_centres = bins[:-1] + np.mean(np.diff(bins)) / 2
        ax1.plot(bin_centres, res_isi)
        ax1.set_xlim([-3, 1])
        ax1.set_xticks([-3, -2, -1, 0])
        ax1.axvline(x=np.log10(0.006))

    fig1.savefig("logisi.png", dpi=400)
    return isi_hist_matrix


def calculate_auto_corr(container):
    ax1, fig1 = nc_plot._make_ax_if_none(None)
    auto_corr_matrix = np.empty((len(container), 20), dtype=float)
    for i, ndata in enumerate(container):
        auto_corr_data = ndata.isi_auto_corr(bins=1, bound=[0, 20])
        auto_corr_matrix[i] = (auto_corr_data["isiCorr"] /
                               ndata.spike.get_unit_stamp().size)
        bins = auto_corr_data['isiAllCorrBins']
        bin_centres = bins[:-1] + np.mean(np.diff(bins)) / 2
        ax1.plot(bin_centres / 1000, auto_corr_matrix[i])
        ax1.set_xlim([0.000, 0.02])
        ax1.set_xticks([0.000, 0.005, 0.01, 0.015, 0.02])
        ax1.axvline(x=0.006)

    fig1.savefig("autocorr.png", dpi=400)
    return auto_corr_matrix


def perform_pca(data, n_components=3, should_scale=True):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    # Standardise the data to improve PCA performance
    if should_scale:
        std_data = scaler.fit_transform(data)
        after_pca = pca.fit_transform(std_data)
    else:
        after_pca = pca.fit_transform(data)

    print(pca.explained_variance_ratio_)
    return after_pca, pca


def ward_clustering(data, plot_dim1=0, plot_dim2=1):
    ax, fig = nc_plot._make_ax_if_none(None)
    dend = shc.dendrogram(
        shc.linkage(data, method="ward", optimal_ordering=True),
        ax=ax)
    fig.savefig("dendogram.png", dpi=400)

    cluster = AgglomerativeClustering(
        n_clusters=2, affinity="euclidean", linkage="ward")
    cluster.fit_predict(data)

    ax, fig = nc_plot._make_ax_if_none(None)
    ax.scatter(
        data[:, plot_dim1],
        data[:, plot_dim2],
        c=cluster.labels_, cmap='rainbow')
    fig.savefig("PCAclust.png", dpi=400)


def pca_clustering(container, n_isi_comps=3, n_auto_comps=2):
    isi_hist_matrix = calculate_isi_hist(container)
    isi_after_pca, _ = perform_pca(
        isi_hist_matrix, n_isi_comps, True)
    auto_corr_matrix = calculate_auto_corr(container)
    corr_after_pca, _ = perform_pca(
        auto_corr_matrix, n_auto_comps, True)
    joint_pca = np.empty(
        (len(container), n_isi_comps + n_auto_comps), dtype=float)
    joint_pca[:, :n_isi_comps] = isi_after_pca
    joint_pca[:, n_isi_comps:n_isi_comps + n_auto_comps] = corr_after_pca
    ward_clustering(joint_pca, 0, 3)


def main(in_dir, tetrode_list, analysis_flags):
    # Load files from dir in tetrodes x, y, z
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(in_dir, tetrode_list=tetrode_list)
    container.setup()

    # Show summary of place
    if analysis_flags[0]:
        place_cell_summary(container)
        plt.close("all")

    # Do numerical analysis
    should_plot = analysis_flags[2]
    if analysis_flags[1]:
        cell_classification_stats(in_dir, container, should_plot=should_plot)

        # Do PCA based analysis
    if analysis_flags[3]:
        pca_clustering(container)


if __name__ == "__main__":
    in_dir = r'C:\Users\smartin5\Recordings\11092017'
    tetrode_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Analysis 0 - summary place cell plot
    # Analysis 1 - csv file of data to classify cells
    # Analysis 2 - more graphical output
    # Analysis 3 - PCA and Dendogram and agglomerative clustering
    analysis_flags = [True, True, True, True]
    main(in_dir, tetrode_list, analysis_flags)
