import vigra
import numpy
import os
import h5py
import ilp_constants as const
import block_yielder
import sys
import subprocess


def eval_h5(proj, key_list):
    """Recursively apply the keys in key_list to proj.

    Example: If key_list is ["key1", "key2"], then proj["key1"]["key2"] is returned.
    :param proj: h5py File object
    :param key_list: list of keys to be applied to proj
    :return: entry of the last dict after all keys have been applied
    """
    if not isinstance(proj, h5py.File):
        raise Exception("A valid h5py File object must be given.")
    # TODO: Maybe add exception if len(key_list) == 0.
    val = proj
    for key in key_list:
        val = val[key]
    return val


def del_from_h5(proj, key_list):
    """Recursively apply the keys in key_list to proj and delete the end result.

    Example: If key_list is ["key1", "key"], then proj["key1"]["key2"] will be deleted.
    :param proj: h5py File object
    :param key_list: list of keys to be applied to proj
    """
    if not isinstance(proj, h5py.File):
        raise Exception("A valid h5py File object must be given.")
    if len(key_list) == 0:
        raise Exception("No keys were given.")
    val = proj
    for key in key_list[:-1]:
        val = val[key]
    del val[key_list[-1]]


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


class ILP(object):
    """Provides basic interactions with ilp files.
    """
    # TODO:
    # Use joblib to cache the accesses of the h5 project file.
    # https://pythonhosted.org/joblib/memory.html

    def __init__(self, project_filename, output_folder, compression="lzf"):
        self._project_filename = project_filename
        self._cache_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self._compression = compression
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

    @property
    def labelsets_count(self):
        """Returns the number of label sets.

        :return: number of label sets
        :rtype: int
        """
        proj = h5py.File(self.project_filename, "r")
        label_sets = eval_h5(proj, const.label_sets_list())
        count = len(label_sets.keys())
        proj.close()
        return count

    def get_data_path(self, data_nr):
        """Returns the file path of the dataset.

        :param data_nr: number of dataset
        :return: file path of the dataset
        :rtype: str
        """
        if self.is_internal(data_nr):
            return self.project_filename
        else:
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
        if self.is_internal(data_nr):
            dataset_id = self.get_dataset_id(data_nr)
            data_key = const.localdata(dataset_id)
        else:
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

    def get_data_location(self, data_nr):
        """Returns the data location (either "ProjectInternal" or "FileSystem").

        :param data_nr: number of dataset
        :return: data location
        :rtype: str
        """
        h5_key = const.datalocation(data_nr)
        data_location = vigra.readHDF5(self.project_filename, h5_key)
        return data_location

    def get_dataset_id(self, data_nr):
        """Returns the ilp dataset id.

        :param data_nr: number of dataset
        :return: dataset id
        :rtype: str
        """
        h5_key = const.datasetid(data_nr)
        dataset_id = vigra.readHDF5(self.project_filename, h5_key)
        return dataset_id

    def get_localdata_key(self, data_nr):
        """Returns the h5 key of the data that is stored inside the ilp file.

        :param data_nr: number of dataset
        :return: key of the data stored inside the ilp file
        :rtype: str
        """
        dataset_id = self.get_dataset_id(data_nr)
        h5_key = const.localdata(dataset_id)
        return h5_key

    def is_internal(self, data_nr):
        """Returns true if the dataset is stored inside the ilp file and false if it is stored on the file system.

        :param data_nr: number of dataset
        :return: whether the dataset is stored inside the ilp file or note
        :rtype: bool
        """
        location = self.get_data_location(data_nr)
        return location == "ProjectInternal"

    def _set_internal(self, data_nr, val):
        """Sets the ilp flag of the given dataset to "ProjectInternal" if val is True, else to "FileSystem".

        :param data_nr: number of dataset
        :param val: whether to set the flag to "ProjectInternal" or "FileSystem"
        """
        h5_key = const.datalocation(data_nr)
        if val:
            vigra.writeHDF5("ProjectInternal", self.project_filename, h5_key)
        else:
            vigra.writeHDF5("FileSystem", self.project_filename, h5_key)

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
        if self.is_internal(data_nr):
            return os.path.join(self.cache_folder, self.get_dataset_id(data_nr) + ".h5")
        else:
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

    def replace_labels(self, data_nr, blocks, block_slices, delete_old_blocks=True):
        """Replaces the labels and their block slices of the dataset.

        :param data_nr: number of dataset
        :param blocks: label blocks
        :param block_slices: block slices
        :param delete_old_blocks: whether the old blocks in the project file shall be deleted
        """
        if len(blocks) != len(block_slices):
            raise Exception("The number of blocks and block slices must be the same.")
        if not delete_old_blocks:
            if len(blocks) != self._label_block_count(data_nr):
                raise Exception("Wrong number of label blocks to be inserted.")

        proj = h5py.File(self.project_filename, "r+")
        if delete_old_blocks:
            for i in range(self._label_block_count(data_nr)):
                del_from_h5(proj, const.label_blocks_list(data_nr, i))

        for i in range(len(blocks)):
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
            if "c" not in old_axisorder:
                old_axisorder.append("c")
            else:
                raise Exception("The labels have the wrong shape or the axisorder is wrong.")
        if not len(label_blocks[0].shape) == len(old_axisorder):
            raise Exception("The labels have the wrong shape or the axisorder is wrong.")
        for axis in old_axisorder:
            if axis not in new_axisorder:
                raise Exception("The axisorder is wrong.")

        # Sort the axes.
        for i, axis in enumerate(new_axisorder):
            # If the new axis is not found, insert it at the current position.
            if axis not in old_axisorder:
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
            output_folder, output_filename = os.path.split(self.get_cache_data_path(data_nr))
            output_path = os.path.join(output_folder, str(data_nr).zfill(4) + "_" + output_filename)
            if self.is_internal(data_nr):
                output_key = self.get_dataset_id(data_nr)
            else:
                output_key = self.get_data_key(data_nr)
            vigra.writeHDF5(new_data, output_path, output_key, compression=self._compression)

            # Update the project file.
            self.set_data_path_key(data_nr, output_path, output_key)
            self._set_internal(data_nr, False)
            self.set_axisorder(data_nr, "tzyxc")
            self._set_axistags_from_data(data_nr)

            # If the dataset has labels, reshape them.
            if self._label_block_count(data_nr) > 0:
                self._reshape_labels(data_nr, axisorder, "tzyxc")

    def retrain(self, ilastik_cmd):
        """Retrains the project using ilastik.

        :param ilastik_cmd: path to the file run_ilastik.sh
        """
        cmd = [ilastik_cmd, "--headless", "--project=%s" % self.project_filename, "--retrain"]
        subprocess.call(cmd, stdout=sys.stdout)

    def predict_all_datasets(self, ilastik_cmd, predict_file=False):
        """Predicts the probabilities of all datasets in the project.

        :param ilastik_cmd: path to the file run_ilastik.sh
        :param predict_file: if this is True, the --predict_file option of ilastik is used
        """
        output_filename = os.path.join(self.cache_folder, "{nickname}_probs.h5")
        cmd = [ilastik_cmd, "--headless", "--project=%s" % self.project_filename, "--output_format=hdf5",
               "--output_filename_format=%s" % output_filename]
        if predict_file:
            pfile = os.path.join(self.cache_folder, "predict_file.txt")
            with open(pfile, "w") as f:
                for i in range(self.data_count):
                    f.write(self.get_data_path_key(i) + "\n")
            cmd.append("--predict_file=%s" % pfile)
        else:
            for i in range(self.data_count):
                cmd.append(self.get_data_path_key(i))
        subprocess.call(cmd, stdout=sys.stdout)

    def predict_dataset(self, ilastik_cmd, data_nr):
        """Uses ilastik to predict the probabilities of the dataset.

        If data_nr is None, all datasets are predicted.
        :param ilastik_cmd: path to the file run_ilastik.sh
        :param data_nr: number of dataset
        """
        output_filename = os.path.join(self.cache_folder, "{nickname}_probs.h5")
        data_path_key = self.get_data_path_key(data_nr)
        cmd = [ilastik_cmd, "--headless", "--project=%s" % self.project_filename, "--output_format=hdf5",
               "--output_filename_format=%s" % output_filename, data_path_key]
        subprocess.call(cmd, stdout=sys.stdout)

    def predict(self, ilastik_cmd, input_filename, output_filename):
        """Uses ilastik to predict the probabilities of the given file.

        :param ilastik_cmd: path to the file run_ilastik.sh
        :param input_filename: h5 path with key to input data (e. g. data/raw.h5/raw)
        :param output_filename: output filename
        """
        cmd = [ilastik_cmd, "--headless", "--project=%s" % self.project_filename, "--output_format=hdf5",
               "--output_filename_format=%s" % output_filename, input_filename]
        subprocess.call(cmd, stdout=sys.stdout)

    def merge_output_into_dataset(self, data_nr, n=0):
        """Merges the ilastik output in the dataset. The first n channels of the dataset are left unchanged.

        It is assumed, that extend_data_tzyxc() has been called, so the channels are in the last dimension.
        :param data_nr: number of dataset
        :param n: number of channels that are left unchanged
        """
        # Get the data.
        filepath = self.get_data_path(data_nr)
        h5key = self.get_data_key(data_nr)
        h5_data_file = h5py.File(filepath, "r")
        h5_data = h5_data_file[h5key]
        h5_output_data_file = h5py.File(self._get_output_data_path(data_nr), "r")
        h5_output_data = h5_output_data_file[const.default_export_key()]

        # Check if the last dimension is used for the channels.
        if "axistags" not in h5_data.attrs or "axistags" not in h5_output_data.attrs:
            raise Exception("Dataset has no axistags.")
        data_axistags = vigra.AxisTags.fromJSON(h5_data.attrs["axistags"])
        output_data_axistags = vigra.AxisTags.fromJSON(h5_output_data.attrs["axistags"])
        if data_axistags != output_data_axistags:
            raise Exception("The merge datasets must have the same axistags.")
        if data_axistags[-1].key != "c":
            raise Exception("Dataset has wrong axistags.")

        # Check that both datasets have the same shape, except for the number of channels.
        if h5_data.shape[:-1] != h5_output_data.shape[:-1] or len(h5_data.shape) != len(h5_output_data.shape):
            raise Exception("Both datasets must have the same shape, except for the number of channels.")

        # Create the h5 file for the merged dataset.
        merge_shape = h5_data.shape[:-1] + (n+h5_output_data.shape[-1],)
        max_chunk_shape = (1, 100, 100, 100, 1)
        chunk_shape = tuple(min(a, b) for a, b in zip(merge_shape, max_chunk_shape))
        temp_filepath = filepath + "_TMP_"
        h5_merged_file = h5py.File(temp_filepath, "w")
        h5_merged_file.create_dataset(h5key, shape=merge_shape, chunks=chunk_shape,
                                      compression=self._compression, dtype=h5_data.dtype)
        h5_merged = h5_merged_file[h5key]
        h5_merged.attrs["axistags"] = h5_data.attrs["axistags"]

        # Copy the raw data to the merge dataset.
        data_merge_shape = h5_data.shape[:-1] + (n,)
        data_blocking = block_yielder.Blocking(data_merge_shape, chunk_shape)
        for block in data_blocking.yieldBlocks():
            slicing = tuple(block.slicing)
            h5_merged[slicing] = h5_data[slicing]

        # Copy the output data to the merge dataset.
        round_probs = h5_data.dtype.kind in "ui"  # round the probabilities if the raw data is of integer type
        output_data_blocking = block_yielder.Blocking(h5_output_data.shape, chunk_shape)
        for block in output_data_blocking.yieldBlocks():
            slicing = tuple(block.slicing)
            tmp_s = slicing[-1]
            s = slice(tmp_s.start + n, tmp_s.stop + n, tmp_s.step)
            merge_slicing = slicing[:-1] + (s,)
            if round_probs:
                h5_merged[merge_slicing] = h5_output_data[slicing] * numpy.iinfo(h5_data.dtype).max
            else:
                h5_merged[merge_slicing] = h5_output_data[slicing]

        # Close the files and rename them.
        h5_merged_file.close()
        h5_data_file.close()
        h5_output_data_file.close()
        os.remove(filepath)
        os.rename(temp_filepath, filepath)
