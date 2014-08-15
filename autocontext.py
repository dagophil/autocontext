# Step 1: Start a new ilastik project with the raw data.
# Step 2: Select features.
# Step 3: Create labels.
# Step 4: Save project and exit ilastik.
# Step 5: Use this script.

import os
import vigra
import numpy


ilastik_cmd = '/home/philip/inst/ilastik-1.1.1-Linux/run_ilastik.sh'
project = '/home/philip/src/ilastik_stuff/testprojects/raw_1.ilp'
probs = '/home/philip/src/ilastik_stuff/output/raw_1.h5'


# 1: Create copy of project
# 2: Read raw_path
# 3: Reshape raw to multichannel (if necessary) and save under new file
# 4: Modify project copy to use new raw data
# Loop:
#   5: Run ilastik to get probabilities
#   6: Merge probs into raw


