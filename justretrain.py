# Just retrain the given project.

import os
import vigra

# ----- Params -----
ilastik_cmd = "/home/philip/inst/ilastik-1.1.1-Linux/run_ilastik.sh"
project = "/home/philip/src/autocontext/data/testproject_output.ilp"
temp_file = "/home/philip/src/autocontext/data/_temp_probs.h5"
# ------------------

# .ilp constants.
ilp_filePath = "Input Data/infos/lane0000/Raw Data/filePath"

# Read raw data.
raw_path = vigra.readHDF5(project, ilp_filePath)
raw_key = os.path.basename(raw_path)
project_dir = os.path.dirname(os.path.realpath(project))
raw_path = os.path.join(project_dir, raw_path[:-len(raw_key)-1])
raw_path_and_key = raw_path+"/"+raw_key

cmd = '{} --headless --project {} --output_format hdf5 --output_filename_format {} {} --retrain'.format(ilastik_cmd, project, temp_file, raw_path_and_key)
os.system(cmd)
