"""Burst analysis of cells."""
import csv
import os
from copy import copy
import logging

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_containeranalysis import place_cell_summary
from neurochat.nc_utils import make_dir_if_not_exists, log_exception, oDict
from neurochat.nc_utils import remove_extension
import neurochat.nc_plot as nc_plot


def save_results_to_csv(filename, in_dicts):
    """Save a dictionary to a csv"""
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
    """Plots a visual of the logspacing in the ISI"""
    # This is equivalent to np.exp(np.linspace)
    x1 = np.logspace(np.log10(start), np.log10(stop), N, base=10)
    y = np.zeros(N)
    plt.plot(x1, y, 'o')
    plt.ylim([-0.5, 1])
    print(x1)
    plt.show()


def log_isi(ndata, start=0.0005, stop=10, num_bins=60):
    """
    Compute the log_isi from an NData object

    Params
    ------
    start - the start time in seconds for the ISI
    stop - the stop time in seconds for the ISI
    num_bins - the number of bins in the ISI
    """
    isi_log_bins = np.linspace(
        np.log10(start), np.log10(stop), num_bins + 1)
    hist, _ = np.histogram(
        np.log10(np.diff(ndata.spike.get_unit_stamp())),
        bins=isi_log_bins, density=False)
    # return ndata.isi(bins=isi_log_bins, density=True), isi_log_bins
    return hist / ndata.spike.get_unit_stamp().size, isi_log_bins
    # return hist, isi_log_bins


def cell_classification_stats(
        in_dir, container, out_name,
        should_plot=False, opt_end=""):
    """
    Compute a csv of cell stats for each unit in a container

    Params
    ------
    in_dir - the data output/input location
    container - the NDataContainer object to get stats for
    should_plot - whether to save some plots for this
    """
    _results = []
    spike_names = container.get_file_dict()["Spike"]
    for i, ndata in enumerate(container):
        data_idx, unit_idx = container._index_to_data_pos(i)
        name = spike_names[data_idx][0]
        parts = os.path.basename(name).split(".")

        # Setup up identifier information
        note_dict = oDict()
        dir_t = os.path.dirname(name)
        note_dict["FullDir"] = dir_t
        if dir_t != in_dir:
            note_dict["RelDir"] = os.path.dirname(name)[len(in_dir + os.sep):]
        else:
            note_dict["RelDir"] = ""
        note_dict["Recording"] = parts[0]
        note_dict["Tetrode"] = int(parts[-1])
        note_dict["Unit"] = ndata.get_unit_no()
        ndata.update_results(note_dict)

        # Caculate cell properties
        ndata.wave_property()
        ndata.place()
        isi = ndata.isi()
        ndata.burst(burst_thresh=6)
        phase_dist = ndata.phase_dist()
        theta_index = ndata.theta_index()
        ndata.bandpower_ratio(
            [5, 11], [1.5, 4], 1.6, relative=True,
            first_name="Theta", second_name="Delta")
        result = copy(ndata.get_results())
        _results.append(result)

        if should_plot:
            plot_loc = os.path.join(
                in_dir, "nc_plots", parts[0] + "_" + parts[-1] + "_" +
                str(ndata.get_unit_no()) + "_phase" + opt_end + ".png")
            make_dir_if_not_exists(plot_loc)
            fig1, fig2, fig3 = nc_plot.spike_phase(phase_dist)
            fig2.savefig(plot_loc)
            plt.close("all")

            if unit_idx == len(container.get_units(data_idx)) - 1:
                plot_loc = os.path.join(
                    in_dir, "nc_plots", parts[0] + "_lfp" + opt_end + ".png")
                make_dir_if_not_exists(plot_loc)

                lfp_spectrum = ndata.spectrum()
                fig = nc_plot.lfp_spectrum(lfp_spectrum)
                fig.savefig(plot_loc)
                plt.close(fig)

    # Save the cell statistics
    make_dir_if_not_exists(out_name)
    save_results_to_csv(out_name, _results)
    _results.clear()


def calculate_isi_hist(container, in_dir, opt_end="", s_color=False):
    """Calculate a matrix of isi_hists for each unit in a container"""
    ax1, fig1 = nc_plot._make_ax_if_none(None)
    isi_hist_matrix = np.empty((len(container), 60), dtype=float)
    color = iter(cm.gray(np.linspace(0, 0.8, len(container))))
    for i, ndata in enumerate(container):
        res_isi, bins = log_isi(ndata)
        isi_hist_matrix[i] = res_isi
        bin_centres = bins[:-1] + np.mean(np.diff(bins)) / 2
        c = next(color) if s_color else "k"
        ax1.plot(bin_centres, res_isi, c=c)
        ax1.set_xlim([-3, 1])
        ax1.set_xticks([-3, -2, -1, 0])
    ax1.axvline(x=np.log10(0.006), c="k")

    plot_loc = os.path.join(
        in_dir, "nc_plots", "logisi" + opt_end + ".png")
    fig1.savefig(plot_loc, dpi=400)
    return isi_hist_matrix


def calculate_auto_corr(container, in_dir, opt_end="", s_color=False):
    """Calculate a matrix of autocorrs for each unit in a container"""
    ax1, fig1 = nc_plot._make_ax_if_none(None)
    auto_corr_matrix = np.empty((len(container), 20), dtype=float)
    color = iter(cm.gray(np.linspace(0, 0.8, len(container))))
    for i, ndata in enumerate(container):
        auto_corr_data = ndata.isi_auto_corr(bins=1, bound=[0, 20])
        auto_corr_matrix[i] = (auto_corr_data["isiCorr"] /
                               ndata.spike.get_unit_stamp().size)
        bins = auto_corr_data['isiAllCorrBins']
        bin_centres = bins[:-1] + np.mean(np.diff(bins)) / 2
        c = next(color) if s_color else "k"
        ax1.plot(bin_centres / 1000, auto_corr_matrix[i], c=c)
        ax1.set_xlim([0.000, 0.02])
        ax1.set_xticks([0.000, 0.005, 0.01, 0.015, 0.02])
    ax1.axvline(x=0.006, c="k")

    plot_loc = os.path.join(
        in_dir, "nc_plots", "autocorr" + opt_end + ".png")
    fig1.savefig(plot_loc, dpi=400)
    return auto_corr_matrix


def perform_pca(data, n_components=3, should_scale=True):
    """
    Perform PCA on a set of data (e.g. ndarray)

    Params
    ------
    data - input data array
    n_components - the number of PCA components to compute
        if this is a float, uses enough components to reach that much variance
    should_scale - whether to scale the data to unit variance
    """
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    # Standardise the data to improve PCA performance
    if should_scale:
        std_data = scaler.fit_transform(data)
        after_pca = pca.fit_transform(std_data)
    else:
        after_pca = pca.fit_transform(data)

    print("PCA fraction of explained variance", pca.explained_variance_ratio_)
    return after_pca, pca


def ward_clustering(
        data, in_dir, plot_dim1=0, plot_dim2=1, opt_end="", s_color=False):
    """
    Perform heirarchical clustering using ward's method

    Params
    ------
    data - input data array
    in_dir - where to save the result to
    plot_dim1 - the PCA dimension to plot
    plot_dim2 - the other PCA dimesion to plot
    """
    ax, fig = nc_plot._make_ax_if_none(None)
    if s_color:
        shc.set_link_color_palette(["m", "c", "y", "k"])
        atc = '#bcbddc'
    else:
        shc.set_link_color_palette(["k"])
        atc = '#bcbddc'
    dend = shc.dendrogram(
        shc.linkage(data, method="ward", optimal_ordering=True),
        ax=ax, above_threshold_color=atc, orientation="right")
    ax.set_yticks([], [])
    plot_loc = os.path.join(
        in_dir, "nc_plots", "dendogram" + opt_end + ".png")
    fig.savefig(plot_loc, dpi=400)
    shc.set_link_color_palette(None)

    cluster = AgglomerativeClustering(
        n_clusters=2, affinity="euclidean", linkage="ward")
    cluster.fit_predict(data)

    ax, fig = nc_plot._make_ax_if_none(None)
    if s_color:
        ax.scatter(data[:, plot_dim1], data[:, plot_dim2],
                   c=cluster.labels_, cmap='rainbow')
    else:
        markers = list(map(lambda a: "k" if a else "r", cluster.labels_))
        ax.scatter(data[:, plot_dim1], data[:, plot_dim2],
                   c=markers)
    plot_loc = os.path.join(
        in_dir, "nc_plots", "PCAclust" + opt_end + ".png")
    fig.savefig(plot_loc, dpi=400)

    return cluster


def save_pca_res(
        container, fname, n_isi_comps, n_auto_comps, isi_pca, corr_pca, clust):
    with open(fname, "w") as f:
        f.write("Type")
        for _ in range(max(n_isi_comps, n_auto_comps)):
            f.write(",Variance Ratio")
        f.write("\nISI_PCA")
        for val in isi_pca.explained_variance_ratio_:
            f.write("," + str(val))
        f.write("\nACH_PCA")
        for val in corr_pca.explained_variance_ratio_:
            f.write("," + str(val))
        f.write("\n")
        f.write("\n")
        o_count = 0
        for i in range(container.get_num_data()):
            str_info = container.get_index_info(i)
            for j in str_info["Units"]:
                f.write(os.path.basename(str_info["Spike"]) +
                        "_Unit_" + str(j) + ",")
            f.write("\n")
            for val in clust.labels_[o_count:o_count + len(str_info["Units"])]:
                f.write(str(val) + ",")
            o_count = o_count + len(str_info["Units"])
            f.write("\n")
        f.write("\n")


def pca_clustering(
        container, in_dir, n_isi_comps=3, n_auto_comps=2,
        opt_end="", s_color=False):
    """
    Wraps up other functions to do PCA clustering on a container.

    Computes PCA for ISI and AC and then clusters based on these.

    Params
    ------
    container - the input NDataContainer to consider
    in_dir - the directory to save information to
    n_isi_comps - the number of principal components for isi
    n_auto_comps - the number of principla components for auto_corr
    """
    print("Considering ISIH PCA")
    make_dir_if_not_exists(os.path.join(in_dir, "nc_plots", "dummy.txt"))
    isi_hist_matrix = calculate_isi_hist(
        container, in_dir, opt_end=opt_end, s_color=s_color)
    isi_after_pca, isi_pca = perform_pca(
        isi_hist_matrix, n_isi_comps, True)
    print("Considering ACH PCA")
    auto_corr_matrix = calculate_auto_corr(
        container, in_dir, opt_end=opt_end, s_color=s_color)
    corr_after_pca, corr_pca = perform_pca(
        auto_corr_matrix, n_auto_comps, True)
    joint_pca = np.empty(
        (len(container), n_isi_comps + n_auto_comps), dtype=float)
    joint_pca[:, :n_isi_comps] = isi_after_pca
    joint_pca[:, n_isi_comps:n_isi_comps + n_auto_comps] = corr_after_pca
    clust = ward_clustering(
        joint_pca, in_dir, 0, 3, opt_end=opt_end, s_color=s_color)
    fname = os.path.join(
        in_dir, "nc_results", "PCA_results" + opt_end + ".csv")
    save_pca_res(
        container, fname, n_isi_comps, n_auto_comps, isi_pca, corr_pca, clust)


def main(
        in_dir, tetrode_list, analysis_flags,
        re_filter=None, test_only=False, opt_end="",
        s_color=False):
    """Summarise all tetrodes in in_dir"""
    # Load files from dir in tetrodes x, y, z
    container = NDataContainer(load_on_fly=True)
    out_name = container.add_axona_files_from_dir(
        in_dir, tetrode_list=tetrode_list,
        recursive=True, re_filter=re_filter)
    container.setup()

    if test_only:
        exit(0)

    # Show summary of place
    if analysis_flags[0]:
        # place_cell_summary(
        #     container, dpi=200, out_dirname="nc_place_plots")
        place_cell_summary(
            container, dpi=200, out_dirname="nc_cell_plots", filter_place_cells=False, filter_low_freq=False,
            opt_end=opt_end)
        plt.close("all")

    # Do numerical analysis
    should_plot = analysis_flags[2]
    if analysis_flags[1]:
        import re
        out_name = remove_extension(out_name) + "csv"
        out_name = re.sub(r"file_list_", r"cell_stats_", out_name)
        cell_classification_stats(
            in_dir, container, out_name,
            should_plot=should_plot, opt_end=opt_end)

    # Do PCA based analysis
    if analysis_flags[3]:
        pca_clustering(container, in_dir, opt_end=opt_end, s_color=s_color)


def setup_logging(in_dir):
    fname = os.path.join(in_dir, 'nc_output.log')
    if os.path.isfile(fname):
        open(fname, 'w').close()
    logging.basicConfig(
        filename=fname, level=logging.DEBUG)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)


if __name__ == "__main__":
    # in_dir = r'C:\Users\smartin5\OneDrive - TCDUD.onmicrosoft.com\Bernstein'
    in_dir = r"C:\Users\smartin5\Recordings\11092017"
    setup_logging(in_dir)
    tetrode_list = [i for i in range(1, 17)]
    optional_end = "_Sean"

    # Use a Regex to filter out certain directories
    re_filter = None
    # re_filter = r"^LSR.*"

    # Analysis 0 - summary place cell plot
    # Analysis 1 - csv file of data to classify cells
    # Analysis 2 - more graphical output
    # Analysis 3 - PCA and Dendogram and agglomerative clustering
    analysis_flags = [False, True, False, True]
    main(
        in_dir, tetrode_list, analysis_flags,
        re_filter=re_filter, test_only=False,
        opt_end=optional_end, s_color=False)
