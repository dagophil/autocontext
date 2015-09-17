import numpy
import math
import random


def scatter_labels_single_block(label_block, label_count, n, weights=None):
    """Creates n blocks of the same size as label_block, where each block contains a subset of the original labels.

    The new blocks are a decomposition of the original block, meaning that they are pairwise disjoint and form the
    original block when merged.
    :param label_block: label block
    :param label_count: number of labels inside the block
    :param n: number of blocks
    :param weights: weights how the labels are spread in the layers
    :return: list label blocks, where each block contains a subset of the original labels
    """
    # Create weights if none were given.
    if weights is None:
        weights = [1]*n
    if len(weights) < n:
        raise Exception("The number of weights is smaller than the number of layers.")
    weights = weights[:n]

    # Find the label coordinates inside the block.
    wh_list = [numpy.where(label_block == i+1) for i in range(label_count)]
    available_list = [range(len(wh[0])) for wh in wh_list]

    # Spread the labels.
    scatter_blocks = []
    for k in range(n):
        # Create a new block to save the labels.
        block = numpy.zeros(label_block.shape, dtype=label_block.dtype)

        # Get the weight factor for the current run.
        weight_factor = weights[0] / float(sum(weights))
        weights.pop(0)

        for i in range(label_count):
            # Choose a random sample of the available indices.
            available_indices = available_list[i]
            num_samples = int(math.ceil(len(available_indices)*weight_factor))
            chosen_indices = random.sample(available_indices, num_samples)

            # Remove the chosen indices from the available indices.
            available_list[i] = [x for x in available_indices if x not in chosen_indices]

            # Take the coordinates of the chosen indices in the block and put the current label there.
            wh = tuple(w[chosen_indices] for w in wh_list[i])
            block[wh] = i+1
        scatter_blocks.append(block)
    return scatter_blocks


def scatter_labels(label_blocks, label_count, n, weights=None):
    """Spread the labels of the given label blocks into n layers.

    If weights is None, the labels are equally spread.
    :param label_blocks: list of label blocks
    :param label_count: number of labels
    :param n: number of layers
    :param weights: weights how the labels are spread in the layers
    :return: list of length n, where each item is a list of label blocks of the original size,
             but containing only a subset of the original labels
    """
    # Create weights if none were given.
    if weights is None:
        weights = [1]*n
    if len(weights) < n:
        raise Exception("The number of weights is smaller than the number of layers.")

    # Scatter the labels in each block.
    return_list = [[] for _ in range(n)]
    for block in label_blocks:
        scatter_blocks = scatter_labels_single_block(block, label_count, n, weights)
        for i, b in enumerate(scatter_blocks):
            return_list[i].append(b)
    return return_list
