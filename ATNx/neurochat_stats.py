"""Burst analysis of cells."""
import csv
import os
import sys
from copy import copy
import logging
import configparser
import json
from pprint import pprint
import argparse

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib

# matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_containeranalysis import place_cell_summary
from neurochat.nc_utils import make_dir_if_not_exists, log_exception, oDict
from neurochat.nc_utils import remove_extension
import neurochat.nc_plot as nc_plot


def save_dicts_to_csv(filename, in_dicts):
    """
    Save a list of dictionaries to a csv.

    The headers are set as the maximal set of keys in in_dicts.
    It is assumed that all other dicts will have a subset of these keys.
    Each entry in the dict is saved to a row of the csv, so it is assumed that
    the values in the dict are mostly floats / ints / etc.
    """
    # first, find the dict with the most keys
    max_key = in_dicts[0].keys()
    for in_dict in in_dicts:
        names = in_dict.keys()
        if len(names) > len(max_key):
            max_key = names

    # Then append other keys if still missing keys
    for in_dict in in_dicts:
        names = in_dict.keys()
        for name in names:
            if not name in max_key:
                max_key.append(name)

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=max_key)
            writer.writeheader()
            for in_dict in in_dicts:
                writer.writerow(in_dict)

    except Exception as e:
        log_exception(e, "When {} saving to csv".format(filename))


def cell_classification_stats(
    in_dir,
    container,
    out_name,
    should_plot=False,
    opt_end="",
    output_spaces=True,
    good_cells=None,
):
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
    overall_count = 0
    for i in range(len(container)):
        try:
            data_idx, unit_idx = container._index_to_data_pos(i)
            name = spike_names[data_idx][0]
            parts = os.path.basename(name).split(".")
            o_name = os.path.join(
                os.path.dirname(name)[len(in_dir + os.sep) :], parts[0]
            )
            note_dict = oDict()
            # Setup up identifier information
            dir_t = os.path.dirname(name)
            note_dict["Index"] = i
            note_dict["FullDir"] = dir_t
            if dir_t != in_dir:
                note_dict["RelDir"] = os.path.dirname(name)[len(in_dir + os.sep) :]
            else:
                note_dict["RelDir"] = ""
            note_dict["Recording"] = parts[0]
            note_dict["Tetrode"] = int(parts[-1])
            if good_cells is not None:
                check = [
                    os.path.normpath(name[len(in_dir + os.sep) :]),
                    container.get_units(data_idx)[unit_idx],
                ]
                if check not in good_cells:
                    continue
            ndata = container[i]
            overall_count += 1
            print(
                "Working on unit {} of {}: {}, T{}, U{}".format(
                    i + 1, len(container), o_name, parts[-1], ndata.get_unit_no()
                )
            )

            note_dict["Unit"] = ndata.get_unit_no()
            ndata.update_results(note_dict)

            # Caculate cell properties
            ndata.wave_property()
            ndata.place()
            isi = ndata.isi()
            ndata.burst(burst_thresh=6)
            theta_index = ndata.theta_index()
            ndata._results["IsPyramidal"] = cell_type(ndata)
            result = copy(ndata.get_results(spaces_to_underscores=not output_spaces))
            _results.append(result)

        except Exception as e:
            to_out = note_dict.get("Unit", "NA")
            print(
                "WARNING: Failed to analyse {} unit {}".format(
                    os.path.basename(name), to_out
                )
            )
            log_exception(
                e, "Failed on {} unit {}".format(os.path.basename(name), to_out)
            )

    # Save the cell statistics
    make_dir_if_not_exists(out_name)
    print("Analysed {} cells in total".format(overall_count))
    save_dicts_to_csv(out_name, _results)
    _results.clear()


def is_pyramidal(wave_width, spike_rate, mean_autocorr):
    """The values should be in msec."""
    # From https://www.jneurosci.org/content/19/1/274
    # Pyramidal
    # Width 0.44 +- 0.005 ms
    # Mean autocorrelation 7.1 +- 0.12 ms
    # Spike rate 1.4 +- 0.01 Hz
    # Interneuron
    # Width 0.24 +- 0.01 ms
    # Mean autocorrelation 12.0 +- 0.17 ms
    # Spike rate 13.0 +- 1.62
    wave_is_inter = wave_width < 0.2
    rate_is_inter = spike_rate > 5
    autocorr_is_inter = mean_autocorr > 10

    if (int(wave_is_inter) + int(rate_is_inter) + int(autocorr_is_inter)) >= 2:
        return False
    else:
        return True


def cell_type(ndata):
    results = ndata.get_results()
    if "Mean Width" not in results:
        ndata.wave_property()
    mean_width = results["Mean width"] / 1000
    ndata._results["Mean width"] = mean_width
    spike_rate = results["Mean Spiking Freq"]
    isi_data = ndata.isi_corr(bound=[-20, 20])
    isi_corr = isi_data["isiCorr"] / results["Number of Spikes"]
    all_bins = isi_data["isiAllCorrBins"]
    centre = np.flatnonzero(all_bins == 0)[0]
    bin_centres = [
        (all_bins[i + 1] + all_bins[i]) / 2 for i in range(len(all_bins) - 1)
    ]
    autocorr_mean = np.sum(bin_centres[centre:] * isi_corr[centre:]) / np.sum(
        isi_corr[centre:]
    )
    ndata._results["AC mean"] = autocorr_mean

    return is_pyramidal(mean_width, spike_rate, autocorr_mean)


def main(args, config):
    # Unpack out the cfg file into easier names
    in_dir = config.get("Setup", "in_dir")
    cells_to_use = config.get("Setup", "cell_csv_location")
    regex_filter = config.get("Setup", "regex_filter")
    regex_filter = None if regex_filter == "None" else regex_filter
    analysis_flags = json.loads(config.get("Setup", "analysis_flags"))
    tetrode_list = json.loads(config.get("Setup", "tetrode_list"))
    should_filter = config.getboolean("Setup", "should_filter")
    seaborn_style = config.getboolean("Plot", "seaborn_style")
    plot_order = json.loads(config.get("Plot", "plot_order"))
    fixed_color = config.get("Plot", "path_color")
    fixed_color = None if fixed_color == "None" else fixed_color
    if len(fixed_color) > 1:
        fixed_color = json.loads(fixed_color)
    s_color = config.getboolean("Plot", "should_color")
    plot_outname = config.get("Plot", "output_dirname")
    dot_size = config.get("Plot", "dot_size")
    dot_size = None if dot_size == "None" else int(dot_size)
    summary_dpi = int(config.get("Plot", "summary_dpi"))
    hd_predict = config.getboolean("Plot", "hd_predict")
    output_format = config.get("Output", "output_format")
    save_bin_data = config.getboolean("Output", "save_bin_data")
    output_spaces = config.getboolean("Output", "output_spaces")
    opt_end = config.get("Output", "optional_end")
    max_units = int(config.get("Setup", "max_units"))
    isi_bound = int(config.get("Params", "isi_bound"))
    isi_bin_length = int(config.get("Params", "isi_bin_length"))

    setup_logging(in_dir)

    if output_format == "pdf":
        matplotlib.use("pdf")

    if seaborn_style:
        sns.set(palette="colorblind")
    else:
        sns.set_style("ticks", {"axes.spines.right": False, "axes.spines.top": False})

    # Automatic extraction of files from starting dir onwards
    container = NDataContainer(load_on_fly=True)
    out_name = container.add_axona_files_from_dir(
        in_dir,
        tetrode_list=tetrode_list,
        recursive=True,
        re_filter=regex_filter,
        verbose=False,
        unit_cutoff=(0, max_units),
    )
    container.setup()
    if len(container) == 0:
        print("Unable to find any files matching regex {}".format(regex_filter))
        exit(-1)

    # Show summary of place
    if analysis_flags[0]:
        place_cell_summary(
            container,
            dpi=summary_dpi,
            out_dirname=plot_outname,
            filter_place_cells=should_filter,
            filter_low_freq=should_filter,
            opt_end=opt_end,
            base_dir=in_dir,
            output_format=output_format,
            isi_bound=isi_bound,
            isi_bin_length=isi_bin_length,
            output=plot_order,
            save_data=save_bin_data,
            fixed_color=fixed_color,
            point_size=dot_size,
            color_isi=s_color,
            burst_thresh=6,
            hd_predict=hd_predict,
        )
        plt.close("all")

    # Do numerical analysis of bursting
    should_plot = analysis_flags[2]
    if analysis_flags[1]:
        import re

        if (cells_to_use is not None) and (cells_to_use != "None"):
            cell_list = []
            with open(cells_to_use, "r") as csvfile:
                reader = csv.reader(csvfile, delimiter=",")
                for row in reader:
                    cell_list.append(
                        [os.path.normpath(row[0].replace("\\", "/")), int(row[1])]
                    )
        else:
            cell_list = None
        out_name = remove_extension(out_name) + "csv"
        out_name = re.sub(r"file_list_", r"cell_stats_", out_name)
        print("Computing cell stats to save to {}".format(out_name))
        cell_classification_stats(
            in_dir,
            container,
            out_name,
            should_plot=should_plot,
            opt_end=opt_end,
            output_spaces=output_spaces,
            good_cells=cell_list,
        )


def setup_logging(in_dir):
    fname = os.path.join(in_dir, "nc_output.log")
    if os.path.isfile(fname):
        open(fname, "w").close()
    logging.basicConfig(filename=fname, level=logging.DEBUG)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)


def print_config(config, msg=""):
    if msg != "":
        print(msg)
    """Prints the contents of a config file"""
    config_dict = [{x: tuple(config.items(x))} for x in config.sections()]
    pprint(config_dict, width=120)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "ca1.cfg")
    config.read(config_path)

    parser = argparse.ArgumentParser(
        description="Process modifiable parameters from command line"
    )
    args, unparsed = parser.parse_known_args()

    if len(unparsed) != 0:
        print("Unrecognised command line argument passed")
        print(unparsed)
        exit(-1)

    print_config(config, "Program started with configuration")
    if len(sys.argv) > 1:
        print("Command line arguments", args)

    main(args, config)
