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
* `python autocontext.py --train myproject.ilp --ilastik /usr/local/ilastik/run_ilastik.sh --cache training/cache`
* `python autocontext.py --train infile.ilp -o outfile.ilp --ilastik /usr/local/ilastik/run_ilastik.sh`


## Example usage (batch prediction)

You can use the autocontext in combination with the ilastik batch prediction. Lets say you want to use batch prediction
on the files `to_predict0.h5/raw` and `to_predict1.h5/raw`. First, you have to train the autocontext (see above). The
trained autocontext is saved in the cache folder, say this is the folder `training/cache`. Now you can call the batch
prediction:

* `python autocontext.py --batch_predict training/cache --ilastik /usr/local/ilastik/run_ilastik.sh --cache prediction/cache --files to_predict0.h5/raw to_predict1.h5/raw`

Please keep in mind, that you need a cache folder for the batch prediction, too. It may be a good idea to use different
cache folders for training and batch prediction.

#### Forwarding arguments to ilastik

All command line arguments that are not used by autocontext are forwarded to ilastik. See
[http://ilastik.org/documentation/pixelclassification/headless.html]
(http://ilastik.org/documentation/pixelclassification/headless.html)
for a full list of ilastik options. Example:

* `python autocontext.py --batch_predict training/cache --ilastik /usr/local/ilastik/run_ilastik.sh --cache prediction/cache --files to_predict0.h5/raw to_predict1.h5/raw --output_filename_format {nickname}_Probabilities.h5 --output_internal_path my_personal_export_key`

There are a few exceptions:

* The options `--headless` and `--project` are ignored, since they are predefined by the autocontext.
* Since you only need the ilastik results from the last autocontext iteration, the options `--output_format`,
  `--output_filename_format`, `--output_internal_path` are only taken into account in the last iteration.

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

