import json
from collections import defaultdict

import torch
import torchvision.utils

# *** CHEMIN ABSOLU POUR LI.TS (MIS À JOUR AVEC LE CHEMIN KAGGLEHUB) ***
# Mettez ici le chemin ABSOLU du dossier RACINE contenant les dossiers 'volumes' et 'segmentations' de LiTS.
LITS_RAW_DATA_ABSOLUTE_PATH = r"C:\Users\danku\Documents\Уроки\ENS\3A\MVA\S1\Medical imaging\Code_medical_image_analysis\Training Batch 1"
# ************************************************************************

# Chemin Absolu pour l'Ultrasound (basé sur nos échanges précédents)
US_RAW_DATA_ABSOLUTE_PATH = "C:/Users/danku/Documents/Уроки/ENS/3A/MVA/S1/Medical imaging/train_images"


def gridify_output(images, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    scaled_images = scale_img(images)

    return (torchvision.utils.make_grid(scaled_images, nrow=row_size, pad_value=-1).cpu().data.
            permute(0, 2, 1).contiguous().permute(2, 1, 0))


def defaultdict_from_json(json_dict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(json_dict)
    return dd


def get_args_from_json(json_file_name, server):
    """
    Charge les arguments depuis un fichier json.

    :param json_file_name: Nom du fichier JSON
    :param server: Nom du serveur
    :return: arguments chargés
    """

    # ... (Code des serveurs IFL et TranslaTUM non modifié) ...
    if server == 'IFL':
        with open(f'/mnt/ceph/sharedscratch/lucie/json_args/{json_file_name}.json', 'r') as f:
            args = json.load(f)
        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            args['data_path'] = "/mnt/ceph/sharedscratch/lucie/BraTS2023_2017_GLI_Challenge_TrainingData"
            args['output_path'] = "/mnt/ceph/sharedscratch/lucie/gd_brats23"
        elif args['dataset'].lower() == "ultrasound":
            args['data_path'] = "/mnt/ceph/sharedscratch/lucie/ultrasound"
            args['output_path'] = "/mnt/ceph/sharedscratch/lucie/ultrasound"
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
        # Mode local (Votre configuration)
        try:
            with open(f'json_args/{json_file_name}.json', 'r') as f:
                args = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration JSON 'json_args/{json_file_name}.json' non trouvée.")

        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            args['data_path'] = "../data/BraTS2023"
            args['output_path'] = "../output/BraTS"
        elif args['dataset'].lower() == "ultrasound":
            # Utiliser le chemin absolu défini
            args['data_path'] = US_RAW_DATA_ABSOLUTE_PATH
            args['output_path'] = "../output/US"
        elif args['dataset'].lower() == "lits":
            # Utiliser le chemin absolu défini
            args['data_path'] = LITS_RAW_DATA_ABSOLUTE_PATH
            args['output_path'] = "../output/LiTS"
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")

    # Mettre à jour l'image size
    if 'img_size' in args and args['dataset'].lower() == "lits":
        args['img_size'] = 128
    
    return args