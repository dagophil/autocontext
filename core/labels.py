# labels.py
#
# Functions to modify labels.
import numpy
import math
import random


# Randomly scatter the labels of a single label block into several layers.
def scatter_labels_single_block(label_block, label_count, n):
    # Find the label coordinates inside the block.
    wh_list = []
    available_list = []
    for i in range(label_count):
        wh = numpy.where(label_block == i+1)
        wh_list.append(wh)
        available_list.append(range(len(wh[0])))

    scatter_blocks = []
    for k in range(n):
        block = numpy.zeros(label_block.shape, dtype=label_block.dtype)
        for i in range(label_count):
            # Choose a random sample of the available indices.
            available_indices = available_list[i]
            available_count = len(available_indices)
            num_samples = int(math.ceil(float(available_count)/(n-k)))
            chosen_indices = random.sample(available_indices, num_samples)

            # Remove the chosen indices from the available indices.
            available_list[i] = [x for x in available_indices if x not in chosen_indices]

            # Take the labels at the chosen indices and put them into the new block.
            wh = wh_list[i]
            dim = len(wh)
            wh = tuple(wh[d][chosen_indices] for d in range(dim))
            block[wh] = i+1
        scatter_blocks.append(block)
    return scatter_blocks


# Randomly scatter the labels of label blocks into several layers.
# label_blocks: List of label blocks.
# n: Number of splits.
#
# Returns:
# List, where each item is of the same format as label_blocks,
# containing only an n-th of the original labels.
def scatter_labels(label_blocks, label_count, n):
    # Create the output structure.
    return_list = []
    for k in range(n):
        return_list.append([])

    # Fill the output structure.
    for block in label_blocks:
        scatter_blocks = scatter_labels_single_block(block, label_count, n)
        for i, b in enumerate(scatter_blocks):
            return_list[i].append(b)
    return return_list
