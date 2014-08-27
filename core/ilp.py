# ilp.py
#
# Provides basic interactions with ilp files.
import os
import vigra
import numpy


# Recursively applies the keys in key_list into the dict.
# Example: key_list = ["key1", "key2"] --> return d["key1"]["key2"]
def move_into_dict(d, key_list):
    val = d
    for key in key_list:
        val = val[key]
    return val


# The ILP class can be used for basic interactions with ilp files.
class ILP(object):

    @staticmethod
    def filepath(lane_number):
        return "/".join(ILP.filepath_list(lane_number))

    @staticmethod
    def filepath_list(lane_number):
        lane_number = str(lane_number).zfill(4)
        return ["Input Data", "infos", "lane" + lane_number, "Raw Data", "filePath"]

    @staticmethod
    def axisorder(lane_number):
        return "/".join(ILP.axisorder_list(lane_number))

    @staticmethod
    def axisorder_list(lane_number):
        lane_number = str(lane_number).zfill(4)
        return ["Input Data", "infos", "lane" + lane_number, "Raw Data", "axisorder"]

    @staticmethod
    def axistags(lane_number):
        return "/".join(ILP.axistags_list(lane_number))

    @staticmethod
    def axistags_list(lane_number):
        lane_number = str(lane_number).zfill(4)
        return ["Input Data", "infos", "lane" + lane_number, "Raw Data", "axistags"]

    @staticmethod
    def label_names_path():
        return "/".join(ILP.label_names_path_list())

    @staticmethod
    def label_names_path_list():
        return ["PixelClassification", "LabelNames"]

    @staticmethod
    def label_path(lane_number):
        return "/".join(ILP.label_path_list(lane_number))

    @staticmethod
    def label_path_list(lane_number):
        lane_number = str(lane_number).zfill(3)
        return ["PixelClassification", "LabelSets", "labels" + lane_number]

    @staticmethod
    def label_block_path(lane_number, block_number):
        return "/".join(ILP.label_block_path_list(lane_number, block_number))

    @staticmethod
    def label_block_path_list(lane_number, block_number):
        lane_number = str(lane_number).zfill(3)
        block_number = str(block_number).zfill(4)
        return ["PixelClassification", "LabelSets", "labels" + lane_number, "block" + block_number]

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

    @staticmethod
    def txyzc_axistags():
        return """{
  "axes": [
    {
      "key": "t",
      "typeFlags": 8,
      "resolution": 0,
      "description": ""
    },
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

    # The constructor extracts the basic information from the ilp file.
    def __init__(self, ilastik_cmd, project_name, lane_number=0):
        self.ilastik_cmd = ilastik_cmd
        self._project_name = project_name
        self._lane_number = lane_number

        # Read data from project file.
        raw_path = vigra.readHDF5(project_name, ILP.filepath(lane_number))
        raw_key = os.path.basename(raw_path)
        project_dir = os.path.dirname(os.path.realpath(project_name))
        raw_path = os.path.join(project_dir, raw_path[:-len(raw_key)-1])
        self._raw_path = raw_path
        self._raw_key = raw_key
        self._raw_axisorder = vigra.readHDF5(project_name, ILP.axisorder(lane_number))

        # Get the labels and the number of label blocks.
        self._label_names = vigra.readHDF5(project_name, ILP.label_names_path())
        from h5py import File
        proj = File(project_name, "r")
        block_count = len(move_into_dict(proj, ILP.label_path_list(lane_number)).keys())
        proj.close()
        self._number_label_blocks = block_count

        # Read number of channels from raw data.
        self._number_channels = 1
        self._number_probability_channels = 0
        if self._raw_axisorder != "xyz":
            raw = vigra.readHDF5(raw_path, raw_key)
            self._number_of_channels = raw.shape[-1]

    # Getter for project name.
    @property
    def project_name(self):
        return self._project_name

    # Getter for lane number.
    @property
    def lane_number(self):
        return self._lane_number

    # Getter for raw path.
    @property
    def raw_path(self):
        return self._raw_path

    # Getter for raw key.
    @property
    def raw_key(self):
        return self._raw_key

    # Getter for raw axisorder.
    @property
    def raw_axisorder(self):
        return self._raw_axisorder

    # Getter for labels.
    @property
    def label_names(self):
        return self._label_names

    # Getter for number label blocks.
    @property
    def number_label_blocks(self):
        return self._number_label_blocks

    # Getter for number channels.
    @property
    def number_channels(self):
        return self._number_channels

    # Getter for number probability channels.
    @property
    def number_probability_channels(self):
        return self._number_probability_channels

    # The string raw_path/raw_key is used in ilp files as path to the raw data.
    @property
    def raw_path_key(self):
        return self._raw_path + "/" + self._raw_key

    # Reads the block slice of the given block.
    def _get_block_slice(self, block_number):
        from h5py import File
        proj = File(self.project_name, "r")
        h5block = self._get_h5_label_block(proj, block_number)
        block_slice = h5block.attrs['blockSlice']
        proj.close()
        return block_slice

    # Sets the block slice of the given block.
    def _set_block_slice(self, block_number, block_slice):
        from h5py import File
        proj = File(self.project_name, "r+")
        h5block = self._get_h5_label_block(proj, block_number)
        h5block.attrs['blockSlice'] = block_slice
        proj.close()

    # Returns the h5py label block of the given number.
    def _get_h5_label_block(self, proj, block_number):
        from h5py import File
        if not isinstance(proj, File):
            raise Exception("proj is not a h5py File.")
        return move_into_dict(proj, ILP.label_block_path_list(self.lane_number, block_number))

    # Writes default txyzc axistags into the project file.
    def _write_txyzc_into_project(self):
        if self.raw_axisorder == "xyz" or self.raw_axisorder == "xyzc":
            # Update axisorder and axistags.
            vigra.writeHDF5("txyzc", self.project_name, ILP.axisorder(self.lane_number))
            vigra.writeHDF5(ILP.txyzc_axistags(), self.project_name, ILP.axistags(self.lane_number))
            self._raw_axisorder = "txyzc"

            # Update the labels
            for i in range(self.number_label_blocks):
                block = vigra.readHDF5(self.project_name, ILP.label_block_path(self.lane_number, i))
                if len(block.shape) == 4:
                    block = numpy.reshape(block, (1,)+block.shape)
                    block_slice = self._get_block_slice(i)
                    block_slice = "[0:1," + block_slice[1:]
                    vigra.writeHDF5(block, self.project_name, ILP.label_block_path(self.lane_number, i))
                    self._set_block_slice(i, block_slice)
                elif len(block.shape) != 5:
                    raise Exception("The labels in the project file have the wrong shape.")
        else:
            if self.raw_axisorder != "txyzc":
                raise Exception("Wrong axisorder in project file.")

    # Copy the raw data and reshape it to a multichannel dataset.
    def copy_raw_data_txyzc(self, file_suffix="_copy"):
        raw = vigra.readHDF5(self.raw_path, self.raw_key)

        # Reshape raw data if necessary.
        if self.raw_axisorder == "xyz":
            raw = numpy.reshape(raw, (1,)+raw.shape+(1,))
        elif self.raw_axisorder == "xyzc":
            raw = numpy.reshape(raw, (1,)+raw.shape)
        self._write_txyzc_into_project()

        # Copy the data.
        self._raw_path = self.raw_path[:-3] + file_suffix + ".h5"
        if os.path.isfile(self.raw_path):
            os.remove(self.raw_path)
        vigra.writeHDF5(raw, self.raw_path, self.raw_key, compression="lzf")
        # TODO:
        # Add axistags attribute to raw dataset.

        # Update the project file.
        vigra.writeHDF5(self.raw_path_key, self.project_name, ILP.filepath(self.lane_number))

    # Returns a list with all label blocks. The list contains 2-tuples (block, blockSlice).
    def extract_label_blocks(self):
        # Read the label blocks.
        blocks = [vigra.readHDF5(self.project_name, ILP.label_block_path(self.lane_number, i))
                  for i in range(self.number_label_blocks)]

        # Read the block slices.
        from h5py import File
        proj = File(self.project_name, "r")
        block_slices = [move_into_dict(proj, ILP.label_block_path_list(self.lane_number, i)).attrs['blockSlice']
                        for i in range(self.number_label_blocks)]
        proj.close()
        return blocks, block_slices

    # Replace the label blocks with the given ones.
    def replace_label_blocks(self, blocks, block_slices):
        if len(blocks) != self.number_label_blocks:
            raise Exception("Wrong number of blocks to be inserted.")

        from h5py import File
        proj = File(self.project_name, "r+")
        for i in range(self.number_label_blocks):
            vigra.writeHDF5(blocks[i], self.project_name, ILP.label_block_path(self.lane_number, i))
            move_into_dict(proj, ILP.label_block_path_list(self.lane_number, i)).attrs['blockSlice'] = block_slices[i]
        proj.close()

    # Retrain the project using ilastik.
    def run_ilastik(self, probs_filename, delete_batch=False):
        # Run ilastik.
        if os.path.isfile(probs_filename):
            os.remove(probs_filename)
        cmd = '{} --headless --project {} --output_format hdf5 --output_filename_format {} {} --retrain'\
            .format(self.ilastik_cmd, self.project_name, probs_filename, self.raw_path_key)
        os.system(cmd)

        # Remove batch entries from project file.
        if delete_batch:
            from h5py import File
            proj = File(self.project_name, "r+")
            del proj['Batch Inputs']
            del proj['Batch Prediction Output Locations']
            proj.close()

            # TODO:
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
        from h5py import File
        proj = File(probs_filename, "r")
        probs = numpy.array(proj[ILP.export_key])
        proj.close()
        if len(raw.shape) != 5:
            raise Exception("Raw data has wrong dimension.")
        if len(probs.shape) != 5:
            raise Exception("Probability data has wrong dimension.")
        self._number_probability_channels = probs.shape[-1]

        # Merge raw and probabilitiy data into one array.
        raw_probs_shape = list(raw.shape)
        raw_probs_shape[-1] = self.number_channels + self.number_probability_channels
        raw_probs_shape = tuple(raw_probs_shape)
        raw_probs = numpy.zeros(raw_probs_shape)
        raw_probs[:, :, :, :, :self.number_channels] = raw[:, :, :, :, :self.number_channels]
        raw_probs[:, :, :, :, self.number_channels:] = probs

        # Save the result.
        vigra.writeHDF5(raw_probs, self.raw_path, self.raw_key, compression="lzf")
