import os
import vigra
import numpy


# The ILP class can be used for basic interactions with ilp files.
class ILP:

    #TODO: Change arguments so the static functions can be used with multiple lanes.

    @staticmethod
    def filepath():
        return "Input Data/infos/lane0000/Raw Data/filePath"

    @staticmethod
    def axisorder():
        return "Input Data/infos/lane0000/Raw Data/axisorder"

    @staticmethod
    def axistags():
        return "Input Data/infos/lane0000/Raw Data/axistags"

    @staticmethod
    def xyzc_axistags():
        return """{
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
    export_key = "exported_data"

    def __init__(self, ilastik_cmd, project_name):
        self.ilastik_cmd = ilastik_cmd
        self.project_name = project_name

        # Read data from project file.
        raw_path = vigra.readHDF5(self.project_name, ILP.filepath())
        raw_key = os.path.basename(raw_path)
        project_dir = os.path.dirname(os.path.realpath(self.project_name))
        raw_path = os.path.join(project_dir, raw_path[:-len(raw_key)-1])
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw_axisorder = vigra.readHDF5(self.project_name, ILP.axisorder())

        # Read number of channels from raw data.
        self.number_of_channels = 1
        self.number_of_probability_channels = 0
        if self.raw_axisorder != "xyz":
            raw = vigra.readHDF5(self.raw_path, self.raw_key)
            self.number_of_channels = raw.shape[-1]

    # The string raw_path/raw_key is used in ilp files as path to the raw data.
    def raw_path_key(self):
        return self.raw_path + "/" + self.raw_key

    # Copy the raw data and reshape it to a multichannel dataset.
    def copy_raw_data_multichannel(self, file_suffix="_copy"):
        raw = vigra.readHDF5(self.raw_path, self.raw_key)

        # Reshape raw data if necessary.
        if self.raw_axisorder == "xyz":
            raw = numpy.reshape(raw, raw.shape+(1,))

            # Update the project file.
            vigra.writeHDF5("xyzc", self.project_name, ILP.axisorder())
            vigra.writeHDF5(ILP.xyzc_axistags(), self.project_name, ILP.axistags())

        # Copy the data.
        self.raw_path = self.raw_path[:-3] + file_suffix + ".h5"
        if os.path.isfile(self.raw_path):
            os.remove(self.raw_path)
        vigra.writeHDF5(raw, self.raw_path, self.raw_key, compression="lzf")

        # Update the project file.
        vigra.writeHDF5(self.raw_path_key(), self.project_name, ILP.filepath())

    # Retrain using ilastik.
    def run_ilastik(self, probs_filename, delete_batch=False):
        # Run ilastik.
        if os.path.isfile(probs_filename):
            os.remove(probs_filename)
        cmd = '{} --headless --project {} --output_format hdf5 --output_filename_format {} {} --retrain'\
            .format(self.ilastik_cmd, self.project_name, probs_filename, self.raw_path_key())
        print cmd
        os.system(cmd)

        # Remove batch entries from project file.
        if delete_batch:
            import h5py
            proj = h5py.File(self.project_name, "r+")
            del proj['Batch Inputs']
            del proj['Batch Prediction Output Locations']
            proj.close()

            #TODO
            # Remove the created memory holes in the h5 file
            # (see "Deleting a dataset doesn't always reduce the file size" on
            # https://github.com/h5py/h5py/wiki/Common-Problems).
            # os.system("h5repack -i projectfile.h5 -o tempfile.h5")
            # os.remove("projectfile.h5")
            # os.rename("tempfile.h5", "projectfile.h5")

    # Merge probabilities into the raw data.
    def merge_probs_into_raw(self, probs_filename):
        # Read raw and probability data.
        raw = vigra.readHDF5(self.raw_path, self.raw_key)
        probs = vigra.readHDF5(probs_filename, ILP.export_key)
        self.number_of_probability_channels = probs.shape[-1]

        # Merge raw and probabilitiy data into one array.
        raw_probs_shape = list(raw.shape)
        raw_probs_shape[-1] = self.number_of_channels + self.number_of_probability_channels
        raw_probs_shape = tuple(raw_probs_shape)
        raw_probs = numpy.zeros(raw_probs_shape)
        raw_probs[:, :, :, :self.number_of_channels] = raw[:, :, :, :self.number_of_channels]
        raw_probs[:, :, :, self.number_of_channels:] = probs

        # Save the result.
        vigra.writeHDF5(raw_probs, self.raw_path, self.raw_key, compression="lzf")
