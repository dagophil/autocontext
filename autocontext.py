# How to use:
# Step 1: Start a new ilastik project with the raw data.
# Step 2: Select features.
# Step 3: Create labels.
# Step 4: Save project and exit ilastik.
# Step 5: Use this script.

import os
import vigra
import numpy
from shutil import copyfile

# ----- Params -----
ilastik_cmd = "/home/philip/inst/ilastik-1.1.1-Linux/run_ilastik.sh"
project = "/home/philip/src/autocontext/data/testproject.ilp"
output_project = "/home/philip/src/autocontext/data/testproject_output.ilp"
temp_probs_file = "/home/philip/src/autocontext/data/_temp_probs.h5"
loop_runs = 4
# ------------------


# Retrain using ilastik.
def run_ilastik(ilastik_sh, project_name, raw_path_and_key, temp_filename, step=1, max_steps=1, delete_batch=False):
    # Run ilastik.
    if os.path.isfile(temp_filename):
        os.remove(temp_filename)
    cmd = '{} --headless --project {} --output_format hdf5 --output_filename_format {} {} --retrain'\
        .format(ilastik_sh, project_name, temp_filename, raw_path_and_key)
    os.system(cmd)

    # Remove batch entries from project file.
    if delete_batch:
        import h5py
        proj = h5py.File(output_project, "r+")
        del proj['Batch Inputs']
        del proj['Batch Prediction Output Locations']
        proj.close()

    # Show some output.
    print
    print "   ----- Finished step {} of {} -----".format(step, max_steps)
    print

# .ilp constants.
ilp_filePath = "Input Data/infos/lane0000/Raw Data/filePath"
ilp_axisorder = "Input Data/infos/lane0000/Raw Data/axisorder"
ilp_axistags = "Input Data/infos/lane0000/Raw Data/axistags"
ilp_xyzc_axistags = """{
  "axes": [
    {
      "key": "x",
      "typeFlags": 2,
      "resolution": 0,
      "description": ""
    },
    {
      "key": "y",
      "typeFlags": 2,
      "resolution": 0,
      "description": ""
    },
    {
      "key": "z",
      "typeFlags": 2,
      "resolution": 0,
      "description": ""
    },
    {
      "key": "c",
      "typeFlags": 1,
      "resolution": 0,
      "description": ""
    }
  ]
}"""

# Copy the ilastik project.
if os.path.isfile(output_project):
    os.remove(output_project)
copyfile(project, output_project)

# Read raw data.
raw_path = vigra.readHDF5(project, ilp_filePath)
raw_key = os.path.basename(raw_path)
project_dir = os.path.dirname(os.path.realpath(project))
raw_path = os.path.join(project_dir, raw_path[:-len(raw_key)-1])
raw = vigra.readHDF5(raw_path, raw_key)

# Reshape raw data and save a copy.
raw_axisorder = vigra.readHDF5(project, ilp_axisorder)
if raw_axisorder == "xyz":
    raw = numpy.reshape(raw, raw.shape+(1,))
    number_of_channels = 1

    # Update the output project.
    vigra.writeHDF5("xyzc", output_project, ilp_axisorder)
    vigra.writeHDF5(ilp_xyzc_axistags, output_project, ilp_axistags)
else:
    number_of_channels = raw.shape[-1]
raw_copy_path = raw_path[:-3]+"_with_probs.h5"
raw_copy_key = raw_key
if os.path.isfile(raw_copy_path):
    os.remove(raw_copy_path)
vigra.writeHDF5(raw, raw_copy_path, raw_copy_key, compression="lzf")
del raw

# Modify the project copy to use the copied raw data.
raw_copy_path_and_key = raw_copy_path+"/"+raw_copy_key
vigra.writeHDF5(raw_copy_path_and_key, output_project, ilp_filePath)

# In a loop: Run ilastik and merge the probabilities into the raw data file.
for i in range(loop_runs):
    # Run ilastik.
    run_ilastik(ilastik_cmd, output_project, raw_copy_path_and_key, temp_probs_file, i+1, loop_runs, delete_batch=True)

    # Read the probabilities.
    raw_copy = vigra.readHDF5(raw_copy_path, raw_copy_key)
    probs = vigra.readHDF5(temp_probs_file, "exported_data")

    # Merge raw and probability into one array.
    raw_copy_with_probs_shape = list(raw_copy.shape)
    raw_copy_with_probs_shape[-1] = number_of_channels + probs.shape[-1]
    raw_copy_with_probs_shape = tuple(raw_copy_with_probs_shape)
    raw_copy_with_probs = numpy.zeros(raw_copy_with_probs_shape)
    raw_copy_with_probs[:, :, :, 0:number_of_channels] = raw_copy[:, :, :, 0:number_of_channels]
    raw_copy_with_probs[:, :, :, number_of_channels:] = probs

    # Save the result.
    vigra.writeHDF5(raw_copy_with_probs, raw_copy_path, raw_copy_key, compression="lzf")
    del raw_copy
    del probs
    del raw_copy_with_probs

# # Run ilastik.
# run_ilastik(ilastik_cmd, output_project, raw_copy_path_and_key, temp_probs_file, loop_runs+1, loop_runs+1, delete_batch=False)

# if os.path.isfile(temp_probs_file):
#     os.remove(temp_probs_file)
