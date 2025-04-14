import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# --- Configuration des chemins ---
# Adaptez ces chemins si nécessaire
dataset_base_folder = '..\datasets projet' # Dossier contenant Train, Test, label_train.csv
train_folder = os.path.join(dataset_base_folder, 'Train')
test_folder = os.path.join(dataset_base_folder, 'Test')
label_file = os.path.join(dataset_base_folder, 'label_train.csv')
output_train_csv_file = 'train_processed.csv' # Nom du fichier CSV pour l'entraînement
output_test_csv_file = 'test_processed.csv'   # Nom du fichier CSV pour le test

# --- Étape 1: Charger les labels d'entraînement ---
print(f"Chargement des labels depuis {label_file}...")
try:
    labels_df = pd.read_csv(label_file)
    labels_df = labels_df.set_index('ID') # Utiliser l'ID comme index
    print(f"{len(labels_df)} labels chargés.")
except FileNotFoundError:
    print(f"ERREUR : Le fichier de labels {label_file} n'a pas été trouvé.")
    exit()

# --- Fonction modifiée pour traiter un dossier (.npy) ---
def process_npy_folder(folder_path, is_train=True, labels=None):
    """
    Charge les fichiers .npy, les aplatit et récupère les labels si entraînement.

    Args:
        folder_path (str): Chemin vers le dossier .npy.
        is_train (bool): True si c'est le dossier d'entraînement.
        labels (pd.DataFrame): DataFrame des labels (indexé par ID) si is_train=True.

    Returns:
        list: Liste de dictionnaires pour le DataFrame.
    """
    all_data = []
    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
    print(f"Traitement de {len(npy_files)} fichiers dans {folder_path}...")

    if not npy_files:
        print(f"AVERTISSEMENT : Aucun fichier .npy trouvé dans {folder_path}")
        return []

    for npy_file_path in tqdm(npy_files, desc=f"Processing {os.path.basename(folder_path)}"):
        try:
            file_id = int(os.path.basename(npy_file_path).split('.')[0])
            data_matrix = np.load(npy_file_path)
            flattened_data = data_matrix.flatten() # Aplatit en 72*48 = 3456 features [source: 14]

            record = {'ID': file_id}
            record.update({f'f_{i}': val for i, val in enumerate(flattened_data)})

            if is_train:
                try:
                    record['target'] = labels.loc[file_id, 'target']
                except KeyError:
                    print(f"\nAVERTISSEMENT : ID {file_id} du dossier Train non trouvé dans {label_file}. Ligne ignorée.")
                    continue
            # Si ce n'est pas is_train (donc c'est le set de test), on n'ajoute PAS la colonne 'target'.

            all_data.append(record)

        except Exception as e:
            print(f"\nERREUR lors du traitement du fichier {npy_file_path}: {e}")

    return all_data

# --- Traiter le dossier Train ---
train_data_list = process_npy_folder(train_folder, is_train=True, labels=labels_df)

# --- Traiter le dossier Test ---
test_data_list = process_npy_folder(test_folder, is_train=False) # Pas besoin de passer les labels

# --- Créer et Sauvegarder le DataFrame d'entraînement ---
if train_data_list:
    print("\nCréation du DataFrame d'entraînement...")
    train_df = pd.DataFrame(train_data_list)
    # Réorganiser les colonnes : ID, target, puis les features
    cols_train = ['ID', 'target'] + [col for col in train_df.columns if col.startswith('f_')]
    train_df = train_df[cols_train]

    print(f"Sauvegarde des données d'entraînement dans {output_train_csv_file}...")
    try:
        train_df.to_csv(output_train_csv_file, index=False)
        print(f"Fichier '{output_train_csv_file}' créé avec succès.")
        print(f"Dimensions : {train_df.shape}")
        # print("Aperçu Train :\n", train_df.head())
    except Exception as e:
        print(f"ERREUR lors de la sauvegarde du fichier CSV d'entraînement : {e}")
else:
    print("Aucune donnée d'entraînement traitée, fichier non créé.")

# --- Créer et Sauvegarder le DataFrame de Test ---
if test_data_list:
    print("\nCréation du DataFrame de test...")
    test_df = pd.DataFrame(test_data_list)
     # Réorganiser les colonnes : ID, puis les features
    cols_test = ['ID'] + [col for col in test_df.columns if col.startswith('f_')]
    test_df = test_df[cols_test]

    print(f"Sauvegarde des données de test dans {output_test_csv_file}...")
    try:
        test_df.to_csv(output_test_csv_file, index=False)
        print(f"Fichier '{output_test_csv_file}' créé avec succès.")
        print(f"Dimensions : {test_df.shape}")
        # print("Aperçu Test :\n", test_df.head())
    except Exception as e:
        print(f"ERREUR lors de la sauvegarde du fichier CSV de test : {e}")
else:
    print("Aucune donnée de test traitée, fichier non créé.")

print("\nTraitement terminé.")