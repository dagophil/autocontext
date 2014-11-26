autocontext
===========

Contains a python script to apply the autocontext method to an ilastik project.
Main part of the code is the python class ILP that can be used to open and modify ilastik ilp files.

Example usage
=============

Open a terminal and run a command like the following:
* python autocontext.py --help
* python autocontext.py myproject.ilp --ilastik /usr/local/ilastik/run_ilastik.sh
* python autocontext.py infile.ilp -o outfile.ilp --ilastik /usr/local/ilastik/run_ilastik.sh

Dependencies
============

Python packages:
* vigra
* numpy
* colorama
* h5py

Other:
* ilastik
* h5repack: Currently, you need the to have the system command h5repack available.
