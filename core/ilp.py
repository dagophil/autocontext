import vigra
import numpy
import os
import h5py
import ilp_constants as const


def eval_h5(proj, key_list):
    """Recursively apply the keys in key_list to proj.

    Example: If key_list is ["key1", "key2"], then proj["key1"]["key2"] is returned.
    :param proj: h5py File object
    :param key_list: list of keys to be applied to proj
    :return: entry of the last dict after all keys have been applied
    """
    if not isinstance(proj, h5py.File):
        raise Exception("A valid h5py File object must be given.")
    val = proj
    for key in key_list:
        val = val[key]
    return val


def reshape_tzyxc(data):
    """Reshape data to tzyxc axisorder and set proper axistags.

    :param data: dataset to be reshaped
    :type data: vigra or numpy array
    :return: reshaped dataset
    """
    if not hasattr(data, "axistags"):
        axistags = vigra.defaultAxistags(len(data.shape))
    else:
        axistags = data.axistags

    # Get the axes that have to be added to the dataset.
    axes = {"t": vigra.AxisInfo.t,
            "z": vigra.AxisInfo.z,
            "y": vigra.AxisInfo.y,
            "x": vigra.AxisInfo.x,
            "c": vigra.AxisInfo.c}
    for axis in axistags:
        assert isinstance(axis, vigra.AxisInfo)
        axes.pop(axis.key, None)

    # Add the axes and create the new shape.
    data_shape = list(data.shape)
    for axis in axes.values():
        axistags.append(axis)
        data_shape.append(1)
    data_shape = tuple(data_shape)

    # Reshape the old data and apply the new axistags.
    return data.reshape(data_shape, axistags=axistags)


def which(program):
    """Mimic the behavior of the UNIX 'which' command.

    :param program: program name
    :return: full path to the program or None if program not found
    """
    def is_exe(p):
        return os.path.isfile(p) and os.access(p, os.X_OK)

    file_path, file_name = os.path.split(program)
    if file_path:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


class ILP(object):
    """Provides basic interactions with ilp files.
    """
    # TODO:
    # Use joblib to cache the accesses of the h5 project file.
    # https://pythonhosted.org/joblib/memory.html

    def __init__(self, project_filename, output_folder):
        self._project_filename = project_filename
        self._cache_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # TODO:
        # Maybe check if the project exists and can be opened.

    @property
    def project_filename(self):
        """Returns filename of the ilp project.

        :return: filename of the ilp project
        :rtype: str
        """
        return self._project_filename

    @property
    def project_dir(self):
        """Returns directory path of the project file.

        :return: directory path of project file
        :rtype: str
        """
        return os.path.dirname(os.path.realpath(self.project_filename))

    @property
    def cache_folder(self):
        """Returns path of the cache folder.

        :return: path of the cache folder
        :rtype: str
        """
        return self._cache_folder

    @property
    def data_count(self):
        """Returns the number of datasets inside the project file.

        :return: number of datasets
        :rtype: int
        """
        proj = h5py.File(self.project_filename, "r")
        input_infos = eval_h5(proj, const.input_infos_list())
        data_count = len(input_infos.keys())
        proj.close()
        return data_count

    def get_data_path(self, data_nr):
        """Returns the file path of the dataset.

        :param data_nr: number of dataset
        :return: file path of the dataset
        :rtype: str
        """
        h5_key = const.filepath(data_nr)
        data_path = vigra.readHDF5(self.project_filename, h5_key)
        data_key = os.path.basename(data_path)
        data_path = os.path.join(self.project_dir, data_path[:-len(data_key)-1])
        return data_path

    def get_data_key(self, data_nr):
        """Returns the h5 key of the dataset.

        :param data_nr: number of dataset
        :return: key of the dataset inside its h5 file
        :rtype: str
        """
        h5_key = const.filepath(data_nr)
        data_path = vigra.readHDF5(self.project_filename, h5_key)
        data_key = os.path.basename(data_path)
        return data_key

    def get_data_path_key(self, data_nr):
        """Returns the h5 path of the dataset (e. g. data/raw.h5/raw).

        :param data_nr: number of dataset
        :return: h5 path of dataset
        :rtype: str
        """
        data_path = self.get_data_path(data_nr)
        data_key = self.get_data_key(data_nr)
        return data_path + "/" + data_key

    def set_data_path_key(self, data_nr, new_path, new_key):
        """Sets file path and h5 key of the dataset.

        :param data_nr: number of dataset
        :param new_path: new file path
        :param new_key: new h5 key
        """
        rel_path = os.path.relpath(os.path.abspath(new_path), self.project_dir) + "/" + new_key
        h5_key = const.filepath(data_nr)
        vigra.writeHDF5(rel_path, self.project_filename, h5_key)

    def get_data(self, data_nr):
        """Returns the dataset.

        :param data_nr: number of dataset
        :return: the dataset
        """
        return vigra.readHDF5(self.get_data_path(data_nr), self.get_data_key(data_nr))

    def get_output_data(self, data_nr):
        """Returns the dataset that was produced by ilastik.

        :param data_nr: number of dataset
        :return: output dataset of ilastik
        """
        return vigra.readHDF5(self._get_output_data_path(data_nr), const.default_export_key())

    def get_cache_data_path(self, data_nr):
        """Returns the file path to the dataset copy in the cache folder.

        :param data_nr: number of dataset
        :return: file path to dataset in the cache folder
        :rtype: str
        """
        data_path = os.path.basename(self.get_data_path(data_nr))
        return os.path.join(self.cache_folder, data_path)

    def _get_output_data_path(self, data_nr):
        """Returns the file path to the output file produced by ilastik.

        :param data_nr: number of dataset
        :return: file path to output file from ilastik
        :rtype: str
        """
        cache_path = self.get_cache_data_path(data_nr)
        path, ext = os.path.splitext(cache_path)
        return path + "_probs" + ext

    def get_channel_count(self, data_nr):
        """Returns the number of channels of the dataset.

        :param data_nr: number of dataset
        :return: number of channels of dataset
        :rtype: int
        """
        # Find the channel axis.
        channel_dim = -1
        axisorder = self.get_axisorder(data_nr)
        if "c" in axisorder:
            channel_dim = axisorder.find("c")

        # Find the channel count.
        data_path = self.get_data_path(data_nr)
        data_key = self.get_data_key(data_nr)
        data = h5py.File(data_path, "r")
        channel_count = data[data_key].shape[channel_dim]
        data.close()
        return channel_count

    def get_axisorder(self, data_nr):
        """Returns the axisorder of the dataset.

        :param data_nr: number of dataset
        :return: axisorder of dataset
        :rtype: str
        """
        return vigra.readHDF5(self.project_filename, const.axisorder(data_nr))

    def set_axisorder(self, data_nr, new_axisorder):
        """Sets the axisorder of the dataset.

        :param data_nr: number of dataset
        :param new_axisorder: new axisorder of dataset
        """
        h5_key = const.axisorder(data_nr)
        vigra.writeHDF5(new_axisorder, self.project_filename, h5_key)

    def get_axistags(self, data_nr):
        """Returns the axistags of the dataset as they are in the project file.

        :param data_nr: number of dataset
        :return: axistags of dataset
        :rtype: str
        """
        return vigra.readHDF5(self.project_filename, const.axistags(data_nr))

    def set_axistags(self, data_nr, new_axistags):
        """Sets the axistags of the dataset (only in the project file, not in the dataset itself).

        :param data_nr: number of dataset
        :param new_axistags: new axistags of dataset
        """
        h5_key = const.axistags(data_nr)
        vigra.writeHDF5(new_axistags, self.project_filename, h5_key)

    def _get_axistags_from_data(self, data_nr):
        """Returns the axistags of the dataset.

        :param data_nr: number of dataset
        :return: axistags
        :rtype: str
        """
        data_path = self.get_data_path(data_nr)
        data_key = self.get_data_key(data_nr)
        data = h5py.File(data_path)
        tags = data[data_key].attrs['axistags']
        data.close()
        return tags

    def _set_axistags_from_data(self, data_nr):
        """Reads the axistags from the raw data and writes them into the project file.

        :param data_nr: number of dataset
        """
        data_axistags = self._get_axistags_from_data(data_nr)
        self.set_axistags(data_nr, data_axistags)

    @staticmethod
    def _h5_labels(proj, data_nr):
        """Returns the h5py object that holds the label blocks.

        :param proj: the ilp project file
        :type proj: h5py.File
        :param data_nr: number of dataset
        :return: labels of the dataset
        :rtype: h5py.Group
        """
        if not isinstance(proj, h5py.File):
            raise Exception("A valid h5py File object must be given.")
        data_nr = str(data_nr).zfill(3)
        h5_key = const.labels_list(data_nr)
        return eval_h5(proj, h5_key)

    def _label_block_count(self, data_nr):
        """Returns the number of label blocks of the dataset.

        :param data_nr: number of dataset
        :return: number of label blocks of the dataset
        :rtype: int
        """
        proj = h5py.File(self.project_filename, "r")
        labels = ILP._h5_labels(proj, data_nr)
        block_count = len(labels.keys())
        proj.close()
        return block_count

    def get_labels(self, data_nr):
        """Returns the labels and their block slices of the dataset.

        :param data_nr: number of dataset
        :return: labels and blockslices of the dataset
        :rtype: tuple
        """
        # Read the label blocks.
        block_count = self._label_block_count(data_nr)
        blocks = [vigra.readHDF5(self.project_filename, const.label_blocks(data_nr, i))
                  for i in range(block_count)]

        # Read the block slices.
        proj = h5py.File(self.project_filename, "r")
        block_slices = [eval_h5(proj, const.label_blocks_list(data_nr, i)).attrs['blockSlice']
                        for i in range(block_count)]
        proj.close()
        return blocks, block_slices

    @property
    def label_names(self):
        """Returns the names of the labels of the dataset.

        :return: names of the labels of the dataset
        :rtype: numpy.ndarray
        """
        return vigra.readHDF5(self.project_filename, const.label_names())

    def replace_labels(self, data_nr, blocks, block_slices):
        """Replaces the labels and their block slices of the dataset.

        :param data_nr: number of dataset
        :param blocks: label blocks
        :param block_slices: block slices
        """
        if len(blocks) != self._label_block_count(data_nr):
            raise Exception("Wrong number of label blocks to be inserted.")
        if len(block_slices) != self._label_block_count(data_nr):
            raise Exception("Wrong number of label block slices to be inserted.")

        proj = h5py.File(self.project_filename, "r+")
        for i in range(self._label_block_count(data_nr)):
            vigra.writeHDF5(blocks[i], self.project_filename, const.label_blocks(data_nr, i))
            h5_blocks = eval_h5(proj, const.label_blocks_list(data_nr, i))
            h5_blocks.attrs['blockSlice'] = block_slices[i]
        proj.close()

    def _reshape_labels(self, data_nr, old_axisorder, new_axisorder):
        """Reshapes the label blocks and their slices.

        :param data_nr: number of dataset
        :param old_axisorder: old axisorder of dataset
        :param new_axisorder: new axisorder of dataset
        """
        # NOTE:
        # When creating labels in ilastik, the axisorder of the labels is the same as
        # in the dataset, except that the c-axis is added,  the t-axis is moved to the
        # left and the c-axis is moved to the right.

        # Get blocks and block slices.
        label_blocks, block_slices = self.get_labels(data_nr)

        # Make lists of old_axisorder and block_slices, so it is easier to insert and swap values.
        old_axisorder = list(old_axisorder)
        block_slices = [sl[1:-1].split(",") for sl in block_slices]

        # Check if it is possible to sort the axes.
        if len(old_axisorder) != len(label_blocks[0].shape):
            if not "c" in old_axisorder:
                old_axisorder.append("c")
            else:
                raise Exception("The labels have the wrong shape or the axisorder is wrong.")
        if not len(label_blocks[0].shape) == len(old_axisorder):
            raise Exception("The labels have the wrong shape or the axisorder is wrong.")
        for axis in old_axisorder:
            if not axis in new_axisorder:
                raise Exception("The axisorder is wrong.")

        # Sort the axes.
        for i, axis in enumerate(new_axisorder):
            # If the new axis is not found, insert it at the current position.
            if not axis in old_axisorder:
                old_axisorder.insert(i, axis)
                label_blocks = [numpy.expand_dims(block, i) for block in label_blocks]
                for sl in block_slices:
                    sl.insert(i, "0:1")
                continue

            # If the axis is at the wrong position, swap it to the correct position.
            old_index = old_axisorder.index(axis)
            if old_index != i:
                old_axisorder[i], old_axisorder[old_index] = old_axisorder[old_index], old_axisorder[i]
                label_blocks = [numpy.swapaxes(block, i, old_index) for block in label_blocks]
                for sl in block_slices:
                    sl[i], sl[old_index] = sl[old_index], sl[i]
                continue

        # Write the reshaped labels into the project file.
        block_slices = ["[" + ",".join(sl) + "]" for sl in block_slices]
        self.replace_labels(data_nr, label_blocks, block_slices)

    def extend_data_tzyxc(self, data_nr=None):
        """Extends the dimension of the dataset and its labels to tzyxc.

        If data_nr is None, all datasets are extended.
        :param data_nr: number of dataset
        """
        if data_nr is None:
            for i in range(self.data_count):
                self.extend_data_tzyxc(i)
        else:
            # Reshape the data with the correct axistags.
            data = self.get_data(data_nr)
            axisorder = self.get_axisorder(data_nr)
            if not hasattr(data, "axistags"):
                data = vigra.VigraArray(data, axistags=vigra.defaultAxistags(axisorder), dtype=data.dtype)
            new_data = reshape_tzyxc(data)

            # Save the reshaped dataset.
            output_path = self.get_cache_data_path(data_nr)
            output_key = self.get_data_key(data_nr)
            vigra.writeHDF5(new_data, output_path, output_key, compression="lzf")

            # Update the project file.
            self.set_data_path_key(data_nr, output_path, output_key)
            self.set_axisorder(data_nr, "tzyxc")
            self._set_axistags_from_data(data_nr)

            # If the dataset has labels, reshape them.
            if self._label_block_count(data_nr) > 0:
                self._reshape_labels(data_nr, axisorder, "tzyxc")

    def retrain(self, ilastik_cmd):
        """Retrains the project using ilastik.

        :param ilastik_cmd: path to the file run_ilastik.sh
        """
        cmd = '{} --headless --project {} --retrain'.format(ilastik_cmd, self.project_filename)
        os.system(cmd)

    def predict_all_datasets(self, ilastik_cmd):
        """Calls predict_dataset for each dataset in the project.

        :param ilastik_cmd: path to the file run_ilastik.sh
        """
        for i in range(self.data_count):
            self.predict_dataset(ilastik_cmd, i)

    def predict_dataset(self, ilastik_cmd, data_nr):
        """Uses ilastik to predict the probabilities of the dataset.

        If data_nr is None, all datasets are predicted.
        :param ilastik_cmd: path to the file run_ilastik.sh
        :param data_nr: number of dataset
        """
        output_filename = self._get_output_data_path(data_nr)
        data_path_key = self.get_data_path_key(data_nr)
        cmd = '{} --headless --project {} --output_format hdf5 --output_filename_format {} {}'\
            .format(ilastik_cmd, self.project_filename, output_filename, data_path_key)
        os.system(cmd)

    def predict(self, ilastik_cmd, input_filename, output_filename):
        """Uses ilastik to predict the probabilities of the given file.

        :param ilastik_cmd: path to the file run_ilastik.sh
        :param input_filename: h5 path with key to input data (e. g. data/raw.h5/raw)
        :param output_filename: output filename
        """
        cmd = '{} --headless --project {} --output_format hdf5 --output_filename_format {} {}'\
            .format(ilastik_cmd, self.project_filename, output_filename, input_filename)
        os.system(cmd)

    def merge_output_into_dataset(self, data_nr, n=0):
        """Merges the ilastik output in the dataset. The first n channels of the dataset are left unchanged.

        It is assumed, that extend_data_tzyxc() has been called, so the channels are in the last dimension.
        :param data_nr: number of dataset
        :param n: number of channels that are left unchanged
        """
        # Read the data.
        data = self.get_data(data_nr)
        output_data = self.get_output_data(data_nr)

        # Check if the last dimension is used for the channels.
        if not hasattr(data, "axistags") or not hasattr(output_data, "axistags"):
            raise Exception("Dataset has no axistags.")
        if data.axistags[-1].key != "c" or output_data.axistags[-1].key != "c":
            raise Exception("Dataset has wrong axistags.")

        # Check that both datasets have the same shape, except for the number of channels.
        if data.shape[:-1] != output_data.shape[:-1] or len(data.shape) != len(output_data.shape):
            raise Exception("Both datasets must have the same shape, except for the number of channels.")

        # Delete the all channels of data except for the first n ones.
        data = data[..., 0:n]

        # Merge the datasets together and preserve the axistags.
        data = vigra.VigraArray(numpy.concatenate([data, output_data], -1), axistags=data.axistags)

        # Overwrite the old data.
        vigra.writeHDF5(data, self.get_data_path(data_nr), self.get_data_key(data_nr), compression="lzf")

        # Use h5repack to remove the memory holes created by vigra.writeHDF5.
        if which("h5repack") is None:
            raise Exception("Currently, h5repack is needed to remove the memory holes created by vigra.writeHDF5.")
        # TODO: Implement a way that does not use h5repack.
        filedir, filename = os.path.split(self.get_data_path(data_nr))
        temp_filepath = os.path.join(filedir, "_TODELETE_" + filename)
        os.rename(self.get_data_path(data_nr), temp_filepath)
        os.system("h5repack -i {} -o {}".format(temp_filepath, self.get_data_path(data_nr)))
        os.remove(temp_filepath)
