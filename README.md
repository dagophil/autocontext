autocontext
===========

Contains a python script to apply the autocontext method to an ilastik project. Currently, this only works on hdf5 data.

Example usage
=============

Before using the script, you must create an ilastik project file:
* Create an ilastik pixel classification project and add one or more datasets.
* Select some features.
* Add some labels.
* Save the project and exit ilastik.

Now you can use the autocontext script. Open a terminal and run a command like the following:
* python autocontext.py --help
* python autocontext.py myproject.ilp --ilastik /usr/local/ilastik/run_ilastik.sh
* python autocontext.py infile.ilp -o outfile.ilp --ilastik /usr/local/ilastik/run_ilastik.sh

Prevent OSError in autocontext iteration
========================================
If possible, replace your ilastik.py by autocontxt/ilastik_mods/ilastik-1.1.X/ilastik.py and start autocontext with the --predict_file flag. This prevents the OSError "Argument list too long" in the prediction step of the autocontext iteration.

Dependencies
============

Python packages (all included in the python installation from ilastik):
* vigra
* numpy
* colorama
* h5py

Other:
* ilastik

