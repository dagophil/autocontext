# How to use:
# Step 1: Start a new ilastik project with the raw data.
# Step 2: Select features.
# Step 3: Create labels.
# Step 4: Save project and exit ilastik.
# Step 5: Use this script.

import os
from shutil import copyfile
from core.ilp import ILP

#  ========  Params  ========
ilastik_cmd = "/home/philip/inst/ilastik-1.1.1-Linux/run_ilastik.sh"
project_name = "/home/philip/src/autocontext/data/test100.ilp"
output_project_name = "/home/philip/src/autocontext/data/test100_output.ilp"
probs_filename = "/home/philip/src/autocontext/data/test100_probs.h5"
loop_runs = 3
#  ==========================


# Copy the project file.
if os.path.isfile(output_project_name):
    os.remove(output_project_name)
copyfile(project_name, output_project_name)

# Create an ILP object for the project.
proj = ILP(ilastik_cmd, output_project_name)

# Copy the raw data and reshape it to xyzc.
proj.copy_raw_data_multichannel()

# Do the autocontext loop.
for i in range(loop_runs):
    proj.run_ilastik(probs_filename, delete_batch=True)
    proj.merge_probs_into_raw(probs_filename)

    # Show some output.
    print
    print "   ----- Finished step {} of {} -----".format(i+1, loop_runs)
    print
