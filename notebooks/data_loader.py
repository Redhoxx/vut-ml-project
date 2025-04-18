import pandas as pd
from pathlib import Path

from notebooks import config


def load_train_data_from_csv(csv_path=config.TRAIN_PROCESSED_PATH):
    """
    Charge les données d'entraînement depuis un fichier CSV prétraité.

    Le CSV doit contenir une colonne ID, une colonne target, et les colonnes
    de features (ex: f_0 à f_3455).

    Args:
        csv_path (Path): Chemin vers le fichier train_processed.csv.

    Returns:
        tuple: Contient :
            - np.ndarray: X_train (features)
            - np.ndarray: y_train (étiquettes)
            - np.ndarray: train_ids (IDs)
            Retourne (None, None, None) en cas d'erreur.
    """
    print(f"Chargement des données d'entraînement depuis : {csv_path}")
    if not csv_path.is_file():
        print(f"Erreur: Le fichier CSV {csv_path} n'existe pas.")
        return None, None, None

    try:
        df = pd.read_csv(csv_path)
        print(f"Fichier CSV d'entraînement chargé : {len(df)} échantillons.")

        # Identifier les colonnes de features (ex: commençant par 'f_')
        feature_cols = [col for col in df.columns if col.startswith('f_')]
        if not feature_cols:
             # Alternative : prendre toutes sauf ID et target
             feature_cols = df.columns.drop([config.ID_COL, config.TARGET_COL]).tolist()

        if not feature_cols or config.TARGET_COL not in df.columns or config.ID_COL not in df.columns:
             raise ValueError("Colonnes ID, target ou features manquantes dans le CSV.")

        X_train = df[feature_cols].values
        y_train = df[config.TARGET_COL].values
        train_ids = df[config.ID_COL].values

        print(f"Chargement terminé.")
        print(f"  Shape de X_train: {X_train.shape}")
        print(f"  Shape de y_train: {y_train.shape}")

        return X_train, y_train, train_ids

    except Exception as e:
        print(f"Erreur lors du chargement ou traitement de {csv_path}: {e}")
        return None, None, None


def load_test_data_from_csv(csv_path=config.TEST_PROCESSED_PATH):
    """
    Charge les données de test depuis un fichier CSV prétraité.

    Le CSV doit contenir une colonne ID et les colonnes de features.
    L'ordre des lignes doit correspondre à celui attendu pour la soumission.

    Args:
        csv_path (Path): Chemin vers le fichier test_processed.csv.

    Returns:
        tuple: Contient :
            - np.ndarray: X_test (features)
            - np.ndarray: test_ids (IDs)
            Retourne (None, None) en cas d'erreur.
    """
    print(f"Chargement des données de test depuis : {csv_path}")
    if not csv_path.is_file():
        print(f"Erreur: Le fichier CSV {csv_path} n'existe pas.")
        return None, None

    try:
        df = pd.read_csv(csv_path)
        print(f"Fichier CSV de test chargé : {len(df)} échantillons.")

        # Identifier les colonnes de features
        feature_cols = [col for col in df.columns if col.startswith('f_')]
        if not feature_cols:
             # Alternative : prendre toutes sauf ID
             feature_cols = df.columns.drop([config.ID_COL]).tolist()

        if not feature_cols or config.ID_COL not in df.columns:
             raise ValueError("Colonnes ID ou features manquantes dans le CSV de test.")

        X_test = df[feature_cols].values
        test_ids = df[config.ID_COL].values

        print(f"Chargement terminé.")
        print(f"  Shape de X_test: {X_test.shape}")

        return X_test, test_ids

    except Exception as e:
        print(f"Erreur lors du chargement ou traitement de {csv_path}: {e}")
        return None, None


# Exemple d'utilisation simplifié
if __name__ == '__main__':
    print("Exécution du data_loader simplifié en mode test...")

    X_train, y_train, train_ids = load_train_data_from_csv()
    if X_train is not None:
        print("\nDonnées d'entraînement (CSV) chargées.")

    print("-" * 20)

    X_test, test_ids = load_test_data_from_csv()
    if X_test is not None:
        print("\nDonnées de test (CSV) chargées.")

    print("\nFin du test data_loader simplifié.")