# Forked from https://github.com/Julian-Wyatt/AnoDDPM/blob/3052f0441a472af55d6e8b1028f5d3156f3d6ed3/helpers.py

import json
from collections import defaultdict

import torch
import torchvision.utils


def gridify_output(images, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    scaled_images = scale_img(images)

    # images = [(img - img.min()) / (img.max() - img.min()) * 255 for img in images]  # scale each image to [0, 255]
    # scaled_images = torch.stack(images).to(torch.uint8)

    return (torchvision.utils.make_grid(scaled_images, nrow=row_size, pad_value=-1).cpu().data.
            permute(0, 2, 1).contiguous().permute(2, 1, 0))


def defaultdict_from_json(json_dict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(json_dict)
    return dd


def get_args_from_json(json_file_name, server):
    """
    Load the arguments from a json file

    :param json_file_name: JSON file name
    :param server: server name
    :return: loaded arguments
    """

    if server == 'IFL':
        with open(f'/home/polyaxon-data/lucie_huang/json_args/{json_file_name}.json', 'r') as f:
            args = json.load(f)
        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            args['data_path'] = "/home/polyaxon-data/data1/Lucie"
            args['output_path'] = "/home/polyaxon-data/outputs1/lucie_huang/gd_brats23"
        elif args['dataset'].lower() == "lits":
            args['data_path'] = "/home/polyaxon-data/data1/LiTS"
            args['output_path'] = "/home/polyaxon-data/outputs1/lucie_huang/lits"
        elif args['dataset'].lower() == "ultrasound":
            args['data_path'] = "/home/polyaxon-data/data1/Lucie/ultrasound/carotid_plaques"
            args['output_path'] = "/home/polyaxon-data/outputs1/lucie_huang/ultrasound"
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")

    elif server == 'TranslaTUM':
        with open(f'/home/data/lucie_huang/json_args/{json_file_name}.json', 'r') as f:
            args = json.load(f)
        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            args['output_path'] = "/home/data/lucie_huang/brats23"
        elif args['dataset'].lower() == "ultrasound":
            args['output_path'] = "/home/data/lucie_huang/ultrasound"
        elif args['dataset'].lower() == "lits":
            args['output_path'] = "/home/data/lucie_huang/lits"
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")

    else:
        # --- LOCAL MACHINE PATHS ---
        # NOTE: You MUST configure these paths to match your local file system.
        LOCAL_RAW_DATA_PATH = "../data"      # Root folder where you keep BraTS2023, LiTS, US folders
        LOCAL_OUTPUT_PATH = "../output"    # Root folder where preprocessed data and models will be saved
        # ---------------------------

        with open(f'json_args/{json_file_name}.json', 'r') as f:
            args = json.load(f)
        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            # Assumes raw data is in ../data/BraTS2023
            args['data_path'] = f"{LOCAL_RAW_DATA_PATH}/BraTS2023"
            args['output_path'] = f"{LOCAL_OUTPUT_PATH}/BraTS"
        elif args['dataset'].lower() == "ultrasound":
            # Assumes raw data is in ../data/US
            args['data_path'] = f"{LOCAL_RAW_DATA_PATH}/US"
            args['output_path'] = f"{LOCAL_OUTPUT_PATH}/US"
        elif args['dataset'].lower() == "lits":
            # Assumes raw data is in ../data/LiTS
            args['data_path'] = f"{LOCAL_RAW_DATA_PATH}/LiTS"
            args['output_path'] = f"{LOCAL_OUTPUT_PATH}/LiTS"
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")

    return args