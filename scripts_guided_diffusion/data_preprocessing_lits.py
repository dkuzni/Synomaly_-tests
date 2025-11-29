import nibabel as nib
import numpy as np
import glob
import pickle
import os
import random

from matplotlib import pyplot as plt
from helpers import get_args_from_json, LITS_RAW_DATA_ABSOLUTE_PATH

random.seed(314)
np.random.seed(314)


def generate_datasets(volume_files, segmentation_files):
    """
    Generate healthy and anomalous datasets from the LiTS volumes.

    :param volume_files: list of volume file paths
    :param segmentation_files: list of segmentation file paths
    :return: healthy train and test datasets with liver masks, anomalous test dataset with tumor masks
    """

    healthy_abdomen_images = np.empty((0, 512, 512), dtype=np.float32)
    healthy_liver_masks = np.empty((0, 512, 512), dtype=np.uint8)
    anomalous_abdomen_images = np.empty((0, 512, 512), dtype=np.float32)
    tumor_masks = np.empty((0, 512, 512), dtype=np.uint8)

    for i in range(len(volume_files)):
        # print(f"Processing volume {i+1}/{len(volume_files)}")

        # Charger les données NIfTI
        try:
            volume_data = nib.load(volume_files[i]).get_fdata().astype(np.float32)
            segmentation_data = nib.load(segmentation_files[i]).get_fdata().astype(np.uint8)
        except Exception as e:
            print(f"Erreur lors du chargement des fichiers {volume_files[i]} ou {segmentation_files[i]}: {e}")
            continue

        # Transposer et normaliser pour avoir (Z, Y, X)
        volume_data = np.transpose(volume_data, (1, 0, 2))
        segmentation_data = np.transpose(segmentation_data, (1, 0, 2))

        # La segmentation LiTS est:
        # 0: arrière-plan
        # 1: Foie (Liver)
        # 2: Tumeur (Tumor)

        # Créer les masques séparés
        liver_mask = (segmentation_data == 1) | (segmentation_data == 2)
        tumor_mask = segmentation_data == 2

        # Déterminer les tranches saines (healthy) et anormales (anomalous)
        # Une tranche est "anomale" si elle contient des tumeurs.
        # Une tranche est "saine" si elle contient du foie mais pas de tumeur.

        # Trouver les indices de tranche où il y a du foie
        liver_slices_indices = np.where(np.any(liver_mask, axis=(0, 1)))[0]

        # Slices avec tumeur (anormales)
        anomalous_mask_slices = np.where(np.any(tumor_mask, axis=(0, 1)))[0]

        # Slices sans tumeur mais avec foie (saines)
        healthy_mask_slices = np.array([idx for idx in liver_slices_indices if idx not in anomalous_mask_slices])

        # Convertir en masques booléens 3D
        anomalous_mask = np.zeros_like(volume_data, dtype=bool)
        anomalous_mask[:, :, anomalous_mask_slices] = True

        healthy_mask = np.zeros_like(volume_data, dtype=bool)
        healthy_mask[:, :, healthy_mask_slices] = True

        # S'assurer que les masques sont cohérents
        # Le script original sélectionne toutes les tranches où la condition est VRAIE, ce qui est fait ici.

        # Normalisation Hounsfield Units (HU)
        # LiTS CT scans ont généralement une plage de -1000 à 400.
        # Recadrage pour se concentrer sur les tissus mous (Foie)
        # Le script original utilisait une plage de [-200, 250], ce qui est raisonnable pour le foie.
        HU_min = -200
        HU_max = 250
        volume_data = np.clip(volume_data, HU_min, HU_max)

        # Normalisation Min-Max simple pour [0, 1]
        volume_data = (volume_data - HU_min) / (HU_max - HU_min)

        # Le code original avait une étape de clip 0, 150 non normalisée, que je remplace par la normalisation HU standard.

        # Séparer et ajouter les tranches
        # Slices saines (Healthy)
        if healthy_mask.sum() > 0:
            current_healthy_images = np.moveaxis(volume_data[:, :, healthy_mask_slices], 2, 0)
            current_healthy_masks = np.moveaxis(liver_mask[:, :, healthy_mask_slices], 2, 0)

            # Application du masque hépatique aux images saines (pour ne garder que le foie)
            # Le foie est à 1 (True), le reste est 0. Multiplier par le masque assure que tout hors-foie est 0.
            current_healthy_images[~current_healthy_masks] = 0.0

            healthy_abdomen_images = np.append(healthy_abdomen_images, current_healthy_images.copy(), axis=0)
            healthy_liver_masks = np.append(healthy_liver_masks, current_healthy_masks.copy(), axis=0)

        # Slices anormales (Anomalous/Tumor)
        if anomalous_mask.sum() > 0:
            current_anomalous_images = np.moveaxis(volume_data[:, :, anomalous_mask_slices], 2, 0)
            current_tumor_masks = np.moveaxis(tumor_mask[:, :, anomalous_mask_slices], 2, 0)

            anomalous_abdomen_images = np.append(anomalous_abdomen_images, current_anomalous_images.copy(), axis=0)
            tumor_masks = np.append(tumor_masks, current_tumor_masks.copy(), axis=0)

        # Libérer la mémoire
        del volume_data
        del segmentation_data

    # Retourner les jeux de données
    return healthy_abdomen_images, healthy_liver_masks, anomalous_abdomen_images, tumor_masks


def load_images(data_path, output_path, num_anomalous_data=1000, show_images=False):
    """
    Load images from the LiTS dataset and create healthy and anomalous datasets.
    """

    # MODIFICATION CLÉ: Rechercher les deux extensions (.nii et .nii.gz)
    # Utiliser os.path.join pour la compatibilité Windows/Linux
    volumes_dir = os.path.join(data_path, "volumes")
    segmentations_dir = os.path.join(data_path, "segmentations")

    # 1. Collecter tous les chemins de fichiers
    volume_files = sorted(
        glob.glob(os.path.join(volumes_dir, "*.nii")) + glob.glob(os.path.join(volumes_dir, "*.nii.gz")))
    segmentation_files = sorted(
        glob.glob(os.path.join(segmentations_dir, "*.nii")) + glob.glob(os.path.join(segmentations_dir, "*.nii.gz")))

    # 2. Vérification des noms de fichiers (surtout si la recherche n'a rien donné)
    if len(volume_files) == 0:
        print(f"ERREUR: Aucun fichier NIfTI (.nii ou .nii.gz) trouvé dans {volumes_dir}.")
        print(
            "Veuillez vérifier que les dossiers 'volumes' et 'segmentations' existent bien à l'intérieur du chemin fourni.")
        return None, None, None, None  # Retourne None si rien n'est trouvé.

    if len(volume_files) != len(segmentation_files):
        print(
            f"Avertissement: Le nombre de volumes ({len(volume_files)}) ne correspond pas au nombre de segmentations ({len(segmentation_files)}).")
        # On continue, mais cela peut causer des erreurs de désalignement.

    print(f"Number of volumes: {len(volume_files)}")

    healthy_abdomen_images, healthy_liver_masks, anomalous_abdomen_images, tumor_masks = generate_datasets(volume_files,
                                                                                                           segmentation_files)

    if healthy_abdomen_images is None or healthy_abdomen_images.size == 0:
        print(
            "ERREUR: generate_datasets n'a produit aucune donnée. Vérifiez les conditions de masquage ou le format des fichiers.")
        return None, None, None, None

    if show_images:
        slice_indices = np.random.randint(0, len(healthy_abdomen_images), 3)

        for idx in slice_indices:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(healthy_abdomen_images[idx, :, :], cmap='gray')
            plt.title("Healthy abdomen")
            plt.axis('off')
            plt.subplot(2, 2, 2)
            plt.imshow(healthy_liver_masks[idx, :, :], cmap='gray')
            plt.title("Liver mask")
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(anomalous_abdomen_images[idx, :, :], cmap='gray')
            plt.title("Anomalous abdomen")
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(tumor_masks[idx, :, :], cmap='gray')
            plt.title("Tumor mask")
            plt.axis('off')
            plt.show()

    # Séparation des données saines (healthy) en train et test
    all_healthy_indices = np.arange(len(healthy_abdomen_images))
    random.shuffle(all_healthy_indices)

    # 80% pour l'entraînement, 20% pour le test (healthy)
    split_idx = int(0.8 * len(healthy_abdomen_images))
    train_indices = all_healthy_indices[:split_idx]
    test_healthy_indices = all_healthy_indices[split_idx:]

    train_healthy_abdomen_images = healthy_abdomen_images[train_indices]
    train_healthy_liver_masks = healthy_liver_masks[train_indices]
    test_healthy_abdomen_images = healthy_abdomen_images[test_healthy_indices]
    test_healthy_liver_masks = healthy_liver_masks[test_healthy_indices]

    # Sous-échantillonnage des données anormales (anomalous) pour le test
    if len(anomalous_abdomen_images) > num_anomalous_data:
        test_anomalous_indices = np.random.choice(len(anomalous_abdomen_images), size=num_anomalous_data, replace=False)
        test_anomalous_abdomen_images = anomalous_abdomen_images[test_anomalous_indices]
        test_tumor_masks = tumor_masks[test_anomalous_indices]
    else:
        test_anomalous_abdomen_images = anomalous_abdomen_images
        test_tumor_masks = tumor_masks

    # Sauvegarde des jeux de données dans le dossier de sortie

    # Healthy Train Data
    save_path = os.path.join(output_path, "train_healthy_abdomen_dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(train_healthy_abdomen_images, f)
    print(
        f"Processed training dataset saved as {save_path}. Total of {len(train_healthy_abdomen_images)} healthy training images.")

    save_path = os.path.join(output_path, "train_healthy_liver_masks.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(train_healthy_liver_masks, f)
    print(f"Processed training liver masks saved as {save_path}.")

    # Healthy Test Data
    save_path = os.path.join(output_path, "test_healthy_abdomen_dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(test_healthy_abdomen_images, f)
    print(
        f"Processed healthy test dataset saved as {save_path}. Total of {len(test_healthy_abdomen_images)} healthy test images.")

    # Anomalous Test Data
    save_path = os.path.join(output_path, "test_anomalous_abdomen_dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(test_anomalous_abdomen_images, f)
    print(
        f"Processed anomalous test dataset saved as {save_path}. Total of {len(test_anomalous_abdomen_images)} anomalous test images.")

    save_path = os.path.join(output_path, "test_anomalous_tumor_masks.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(test_tumor_masks, f)
    print(f"Processed tumor masks saved as {save_path}.")

    return train_healthy_abdomen_images, train_healthy_liver_masks, test_healthy_abdomen_images, test_healthy_liver_masks, test_anomalous_abdomen_images, test_tumor_masks


def preprocess(server):
    """ Preprocess the LiTS dataset. """

    # Initialisation des chemins pour le mode Local
    # On utilise LITS_RAW_DATA_ABSOLUTE_PATH de helpers.py
    data_path = LITS_RAW_DATA_ABSOLUTE_PATH
    output_path = os.path.join("..", "output", "LiTS")  # Chemin de sortie par défaut

    # Vérification de l'existence du dossier de sortie
    os.makedirs(output_path, exist_ok=True)

    print("-" * 50)
    print(f"LiTS Preprocessing started in Local Mode.")
    print(f"Raw Data Path (Input): {data_path}")
    print(f"Output Path (Saving PKL): {os.path.abspath(output_path)}")
    print("-" * 50)

    load_images(data_path, output_path, num_anomalous_data=1000)


if __name__ == "__main__":
    using_server = 'None'  # ['IFL', 'TranslaTUM', 'None']
    preprocess(using_server)