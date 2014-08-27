

def filepath_list(lane_number):
    lane_number = str(lane_number).zfill(4)
    return ["Input Data", "infos", "lane" + lane_number, "Raw Data", "filePath"]


def filepath(lane_number):
    return "/".join(filepath_list(lane_number))


def axisorder_list(lane_number):
    lane_number = str(lane_number).zfill(4)
    return ["Input Data", "infos", "lane" + lane_number, "Raw Data", "axisorder"]


def axisorder(lane_number):
    return "/".join(axisorder_list(lane_number))


def axistags_list(lane_number):
    lane_number = str(lane_number).zfill(4)
    return ["Input Data", "infos", "lane" + lane_number, "Raw Data", "axistags"]


def axistags(lane_number):
    return "/".join(axistags_list(lane_number))


def input_infos_list():
    return ["Input Data", "infos"]


def input_infos():
    return "/".join(input_infos_list())


def label_names_list():
    return ["PixelClassification", "LabelNames"]


def label_names():
    return "/".join(label_names_list())


def labels_list(lane_number):
    lane_number = str(lane_number).zfill(3)
    return ["PixelClassification", "LabelSets", "labels" + lane_number]


def labels(lane_number):
    return "/".join(labels_list(lane_number))


def label_blocks_list(lane_number, block_number):
    lane_number = str(lane_number).zfill(3)
    block_number = str(block_number).zfill(4)
    return ["PixelClassification", "LabelSets", "labels" + lane_number, "block" + block_number]


def label_blocks(lane_number, block_number):
    return "/".join(label_blocks_list(lane_number, block_number))
