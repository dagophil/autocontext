"""
DESCRIPTION

How to use:
* Start a new ilastik project and add one or more datasets.
* Select some features.
* Create labels (only on one dataset).
* Save project and exit ilastik.
* Run this script (parameters: see __main__ part below) or use the autocontext function.
"""
from core.ilp import ILP
from core.labels import scatter_labels
from core.ilp import which
import colorama as col
import sys
import argparse
import os


def autocontext(ilastik_cmd, project, runs, label_data_nr, weights=None):
    """Trains and predicts the ilastik project using the autocontext method.

    The parameter weights can be used to take different amounts of the labels in each loop run.
    Example: runs = 3, weights = [3, 2, 1]
             The sum of the weights is 6, so in the first run, 1/2 (== 3/6) of the labels is used,
             then 1/3 (== 2/6), then 1/6.
    If weights is None, the labels are equally distributed over the loop runs.
    :param ilastik_cmd: path to run_ilastik.sh
    :param project: the ILP object of the project
    :param runs: number of runs of the autocontet loop
    :param label_data_nr: number of dataset that contains the labels
    :param weights: weights for the labels
    """
    assert isinstance(project, ILP)

    # Create weights if none were given.
    if weights is None:
        weights = [1]*runs
    if len(weights) < runs:
        raise Exception("The number of weights must not be smaller than the number of runs.")
    weights = weights[:runs]

    # Copy the raw data to the output folder and reshape it to txyzc.
    project.extend_data_tzyxc()

    # Get the number of datasets.
    data_count = project.data_count

    # Get the current number of channels in the datasets.
    # The data in those channels is left unchanged when the ilastik output is merged back.
    keep_channels = [project.get_channel_count(i) for i in range(data_count)]

    # Read the labels from the first block and split them into parts, so not all labels are used in each loop.
    blocks, block_slices = project.get_labels(label_data_nr)
    label_count = len(project.label_names)
    scattered_labels = scatter_labels(blocks, label_count, runs, weights)

    # Do the autocontext loop.
    for i in range(runs):
        print col.Fore.GREEN + "- Running autocontext loop %d of %d -" % (i+1, runs) + col.Fore.RESET

        # Insert the subset of the labels into the project.
        split_blocks = scattered_labels[i]
        project.replace_labels(label_data_nr, split_blocks, block_slices)

        # Retrain the project.
        print col.Fore.GREEN + "Retraining:" + col.Fore.RESET
        project.retrain(ilastik_cmd)

        # Predict all datasets.
        for k in range(data_count):
            print col.Fore.GREEN + "Predicting dataset %d of %d:" % (k+1, data_count) + col.Fore.RESET
            project.predict_dataset(ilastik_cmd, k)

        # Merge the probabilities back into the datasets.
        print col.Fore.GREEN + "Merging output back into datasets." + col.Fore.RESET
        for k in range(data_count):
            project.merge_output_into_dataset(k, keep_channels[k])

    # Insert the original labels back into the project.
    project.replace_labels(label_data_nr, blocks, block_slices)


def process_command_line():
    """Parse command line arguments.
    """
    # Add the command line arguments.
    parser = argparse.ArgumentParser(description="ilastik autocontext",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", type=str,
                        help="input file")
    parser.add_argument("-o", "--outfile", type=str, default="",
                        help="output file")
    parser.add_argument("-n", "--nloops", type=int, default=3,
                        help="number of autocontext loop iterations")
    parser.add_argument("-d", "--labeldataset", type=int, default=0,
                        help="id of dataset in the ilp file that contains the labels")
    parser.add_argument("-c", "--cache", type=str, default="cache",
                        help="name of the cache folder")
    parser.add_argument("--ilastik", type=str, required=True,
                        help="path to the file run_ilastik.sh")
    args = parser.parse_args()

    # Check arguments for validity.
    if not os.path.isfile(args.infile):
        raise Exception("%s is not a file" % args.infile)
    if len(args.outfile) == 0:
        file_path, file_ext = os.path.splitext(args.infile)
        args.outfile = file_path + "_out" + file_ext
    if not os.path.isfile(args.ilastik) or not os.access(args.ilastik, os.X_OK):
        raise Exception("%s is not an executable file" % args.ilastik)
    return args


def main():
    """
    """
    import shutil
    import random
    import os

    # Read command line arguments.
    args = process_command_line()

    # Check if h5repack is installed.
    if which("h5repack") is None:
        raise Exception("Currently, h5repack is needed to remove the memory holes created by vigra.writeHDF5.")

    # Initialize colorama and random seeds.
    random.seed(0)
    col.init()

    # Copy the project file.
    # TODO: If the file exists, ask the user if it shall be deleted.
    if os.path.isfile(args.outfile):
        os.remove(args.outfile)
    shutil.copyfile(args.infile, args.outfile)

    # Clear the cache folder.
    if os.path.isdir(args.cache):
        print "The cache folder", os.path.abspath(args.cache), "already exists."
        clear_cache = raw_input("Clear cache folder? [y|n] : ")
        if clear_cache in ["y", "Y"]:
            for f in os.listdir(args.cache):
                f_path = os.path.join(args.cache, f)
                try:
                    if os.path.isfile(f_path):
                        os.remove(f_path)
                    elif os.path.isdir(f_path):
                        shutil.rmtree(f_path)
                except Exception, e:
                    print e
            print "Cleared cache folder."
        else:
            print "Cache folder not cleared."

    # Create an ILP object for the project.
    proj = ILP(args.outfile, args.cache)

    # Do the autocontext loop.
    autocontext(args.ilastik, proj, args.nloops, args.labeldataset)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
