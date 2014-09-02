# autocontext.py
#
# How to use:
# Step 1: Start a new ilastik project with the raw data.
# Step 2: Select features.
# Step 3: Create labels.
# Step 4: Save project and exit ilastik.
# Step 5: Use this script.

import os
from shutil import copyfile
from core.ilp import ILP
from core.labels import scatter_labels
import random
import shutil
import colorama as col


#  ========  Params  ========
ilastik_cmd = "/home/philip/inst/ilastik-1.1.1-Linux/run_ilastik.sh"
project_name = "/home/philip/src/autocontext/data/test_50_100.ilp"
output_project_name = "/home/philip/src/autocontext/data/test_50_100_output.ilp"
cache_folder = "data/output_data"
loop_runs = 2
label_data_nr = 0
#  ==========================


# Initialize colorama and random seeds.
random.seed(0)
col.init()

# Copy the project file.
if os.path.isfile(output_project_name):
    os.remove(output_project_name)
copyfile(project_name, output_project_name)

# Clear the cache folder.
if os.path.isdir(cache_folder):
    shutil.rmtree(cache_folder)

# Create an ILP object for the project.
proj = ILP(output_project_name, cache_folder)

# Copy the raw data and reshape it to txyzc.
proj.extend_data_tzyxc()

# Get the current number of channels in the datasets.
# The data in those channels is left unchanged when the ilastik output is merged back.
keep_channels = [proj.get_channel_count(i) for i in range(proj.data_count)]

# Read the labels from the first block and split them into parts, so not all labels are used in each loop.
blocks, block_slices = proj.get_labels(label_data_nr)
label_count = len(proj.label_names)
split_blocks = scatter_labels(blocks, label_count, loop_runs)

# Get the number of datasets.
data_count = proj.data_count

# Do the autocontext loop.
for i in range(loop_runs):
    print col.Fore.GREEN + "Running loop %d of %d" % (i+1, loop_runs) + col.Fore.RESET

    # Insert the subset of the labels into the project.
    blocks = split_blocks[i]
    proj.replace_labels(label_data_nr, blocks, block_slices)

    # Retrain the project.
    print col.Fore.GREEN + "  Retraining:" + col.Fore.RESET
    proj.retrain(ilastik_cmd)

    # Predict all datasets.
    for k in range(data_count):
        print col.Fore.GREEN + "  Predicting dataset %d of %d:" % (k+1, data_count) + col.Fore.RESET
        proj.predict_dataset(ilastik_cmd, k)

    # Merge the probabilities back into the datasets.
    print col.Fore.GREEN + "  Merging output back into datasets." + col.Fore.RESET
    for k in range(data_count):
        proj.merge_output_into_dataset(k, keep_channels[k])
