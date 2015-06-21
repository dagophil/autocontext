autocontext
===========

Contains a python script to apply the autocontext method to an ilastik project.

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

Dependencies
============

Python packages (all included in the python installation from ilastik):
* vigra
* numpy
* colorama
* h5py

Other:
* ilastik

