import json
import numpy as np
import torch

def convert_to_braille_unicode(str_input: str, path: str = "utils/braille_map.json") -> str:
    with open(path, "r", encoding='utf-8') as fl:
        data = json.load(fl)

    if str_input in data.keys():
        str_output = data[str_input]
    return str_output


def map_characters(input_string, mapping_file_path):
    try:
        with open(mapping_file_path, encoding='utf-8') as f:
            mapping = json.load(f)  # Load the JSON mapping into a dictionary
    except FileNotFoundError:
        print(f"Mapping file not found: {mapping_file_path}")
        return input_string
    except json.JSONDecodeError:
        print(f"Invalid JSON format in mapping file: {mapping_file_path}")
        return input_string

    result_string = ""
    for char in input_string:
        if char in mapping:
            result_string += mapping[char]
        else:
            result_string += char

    return result_string


def parse_xywh_and_class(boxes: torch.Tensor) -> list:
    """
    boxes input tensor
        boxes (torch.Tensor) or (numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6).
        orig_shape (torch.Tensor) or (numpy.ndarray): Original image size, in the format (height, width).
    Properties:
        xyxy (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format.
        conf (torch.Tensor) or (numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor) or (numpy.ndarray): The class values of the boxes.
        xywh (torch.Tensor) or (numpy.ndarray): The boxes in xywh format.
        xyxyn (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format normalized by original image size.
        xywhn (torch.Tensor) or (numpy.ndarray): The boxes in xywh format normalized by original image size.
    """

    # copy values from troublesome "boxes" object to numpy array
    new_boxes = np.zeros(boxes.shape)
    new_boxes[:, :4] = boxes.xywh.numpy()  # first 4 channels are xywh
    new_boxes[:, 4] = boxes.conf.numpy()  # 5th channel is confidence
    new_boxes[:, 5] = boxes.cls.numpy()  # 6th channel is class which is last channel

    # sort according to y coordinate
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]

    # find threshold index to break the line
    y_threshold = np.mean(new_boxes[:, 3]) // 2
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]

    # cluster according to threshold_index
    boxes_clustered = np.split(new_boxes, threshold_index + 1)
    boxes_return = []
    for cluster in boxes_clustered:
        # sort according to x coordinate
        cluster = cluster[cluster[:, 0].argsort()]
        boxes_return.append(cluster)

    return boxes_return