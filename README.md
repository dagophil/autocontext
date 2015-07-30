# autocontext

Contains a python script to apply the autocontext method to an ilastik project. Currently, this only works on hdf5 data.


## Example usage (training)

Before using the script, you must create an ilastik project file:

* Create an ilastik pixel classification project and add one or more datasets.
* Select some features.
* Add some labels.
* Save the project and exit ilastik.

Now you can use the autocontext script. Open a terminal and run a command like the following:

* `python autocontext.py --help`
* `python autocontext.py --train myproject.ilp --ilastik /usr/local/ilastik/run_ilastik.sh`
* `python autocontext.py --train infile.ilp -o outfile.ilp --ilastik /usr/local/ilastik/run_ilastik.sh`


## Example usage (batch prediction)

You can use the autocontext in combination with the ilastik batch prediction. Lets say you want to use batch prediction
on the files `to_predict0.h5/raw` and `to_predict1.h5/raw`. First, you have to train the autocontext (see above). The trained 
autocontext is saved in the cache folder, say this is the folder `training/cache`. Now you can call the batch prediction:

* `python autocontext.py --batch_predict training/cache --ilastik /usr/local/ilastik/run_ilastik.sh --files to_predict0.h5/raw to_predict1.h5/raw`

Please keep in mind, that you need a cache folder for the batch prediction, too. It may be a good idea to use different
cache folders for training and batch prediction.


#### Coming soon:

You can also use the placeholder * to predict a whole folder of files. In order to prevent the automatic command line
expansion, you may have to enclose the filename in quotes:
you have to enclose the filename in quotes:

* `python autocontext.py --batch_predict training/cache --ilastik /usr/local/ilastik/run_ilastik.sh --files "*.h5/raw"`


## Prevent OSError in autocontext iteration

If possible, replace your `ilastik.py` by `autocontxt/ilastik_mods/ilastik-1.1.X/ilastik.py` and start autocontext with
the `--predict_file` flag. This prevents the OSError "Argument list too long" in the prediction step of the autocontext
iteration.


## Dependencies

Python packages (all included in the python installation from ilastik):

* vigra
* numpy
* colorama
* h5py

Other:

* ilastik


## TODO

* Support ilastik projects with datasets other than hdf5.

