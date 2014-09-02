"""
DESCRIPTION

How to use:
* Start a new ilastik project and add one or more datasets.
* Select some features.
* Create labels (only on one dataset).
* Save project and exit ilastik.
* Set the desired parameters (see below).
* Run this script.
"""

from core.ilp import ILP
from core.labels import scatter_labels
import colorama as col


def autocontext(ilastik_cmd, project, runs, label_data_nr, weights=None):
    assert isinstance(project, ILP)

    # Create weights if none were given.
    if weights is None:
        weights = [1]*runs
    if len(weights) < runs:
        raise Exception("The number of weights must not be smaller than the number of runs.")
    weights = weights[:runs]

    # Copy the raw data and reshape it to txyzc.
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


if __name__ == "__main__":
    import shutil
    import random
    import os

    #  ========  Parameters  ========
    ilastik_sh = "/home/philip/inst/ilastik-1.1.1-Linux/run_ilastik.sh"
    project_name = "/home/philip/src/autocontext/data/test_50_100.ilp"
    output_project_name = "/home/philip/src/autocontext/data/test_50_100_output.ilp"
    cache_folder = "data/output_data"
    loop_runs = 3
    label_dataset = 0
    #  ==============================

    # Initialize colorama and random seeds.
    random.seed(0)
    col.init()

    # Copy the project file.
    if os.path.isfile(output_project_name):
        os.remove(output_project_name)
    shutil.copyfile(project_name, output_project_name)

    # Clear the cache folder.
    # TODO: Maybe don't do this...
    if os.path.isdir(cache_folder):
        shutil.rmtree(cache_folder)

    # Create an ILP object for the project.
    proj = ILP(output_project_name, cache_folder)

    # Do the autocontext loop.
    autocontext(ilastik_sh, proj, loop_runs, label_dataset)
