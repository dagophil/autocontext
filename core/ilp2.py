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


# TODO:
# Use joblib to cache the accesses of the h5 project file.
# https://pythonhosted.org/joblib/memory.html
class ILP(object):
    """Provides basic interactions with ilp files.
    """

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
        """
        return self._project_filename

    @property
    def project_dir(self):
        """Returns directory path of the project file.

        :return: directory path of project file
        """
        return os.path.dirname(os.path.realpath(self.project_filename))

    @property
    def cache_folder(self):
        """Returns path of the cache folder.

        :return: path of the cache folder
        """
        return self._cache_folder

    @property
    def data_count(self):
        """Returns the number of datasets inside the project file.

        :return: number of datasets
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
        """
        h5_key = const.filepath(data_nr)
        data_path = vigra.readHDF5(self.project_filename, h5_key)
        data_key = os.path.basename(data_path)
        return data_key

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

    def get_output_data_path(self, data_nr):
        """Returns the file path to the probability data of the dataset.

        :param data_nr: number of dataset
        :return: file path to probability data of the dataset
        """
        data_path = self.get_data_path(data_nr)
        filename, ext = os.path.splitext(os.path.basename(data_path))
        return os.path.join(self.cache_folder, filename + "_probs" + ext)

    def get_axisorder(self, data_nr):
        """Returns the axisorder of the dataset.

        :param data_nr: number of dataset
        :return: axisorder of dataset
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
        :return: labels of the dataset as h5py object
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
        """Extend the dimension of some dataset and its labels to tzyxc.

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
            output_path = self.get_output_data_path(data_nr)
            output_key = self.get_data_key(data_nr)
            vigra.writeHDF5(new_data, output_path, output_key)

            # Update the project file.
            self.set_data_path_key(data_nr, output_path, output_key)
            self.set_axisorder(data_nr, "tzyxc")
            self._set_axistags_from_data(data_nr)

            # Reshape the labels.
            self._reshape_labels(data_nr, axisorder, "tzyxc")

    # TODO: Implement this function.
    def retrain(self, ilastik_cmd):
        """Retrain the project using ilastik.

        :param ilastik_cmd: path to the file run_ilastik.sh
        :return:
        """
        raise NotImplementedError

    # TODO: Implement this function.
    def merge_probs_into_raw(self, data_nr, probs_filename=None):
        """Merge probabilities into raw data.

        If probs_filename is None, the default output of retrain is taken.
        :param data_nr:
        :param probs_filename:
        :return:
        """
        raise NotImplementedError
