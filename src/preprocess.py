import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import joblib

import config
import utils

def scale_features(X_train, X_test=None, scaler_path=config.SCALER_PATH):
    """
    Applique StandardScaler aux données.
    Le scaler est fitté sur X_train et sauvegardé, puis appliqué à X_train et X_test.
    Si X_test est None, applique seulement sur X_train (utile pour prédiction).
    Si scaler_path existe et contient un scaler chargé, il est utilisé pour transformer
    au lieu d'en fitter un nouveau (utile pour prédiction).

    Args:
        X_train (np.ndarray): Données d'entraînement.
        X_test (np.ndarray, optional): Données de test/validation. Defaults to None.
        scaler_path (Path, optional): Chemin pour sauvegarder/charger le scaler.

    Returns:
        tuple: Contient :
            - np.ndarray: X_train mis à l'échelle.
            - np.ndarray/None: X_test mis à l'échelle (ou None si X_test était None).
            - StandardScaler: Le scaler fitté ou chargé.
            Retourne (None, None, None) en cas d'erreur.
    """
    scaler = None
    loaded_scaler = False
    if scaler_path and scaler_path.exists():
        print(f"Chargement du scaler existant depuis {scaler_path}")
        scaler = utils.load_object(scaler_path)
        if scaler:
            loaded_scaler = True
        else:
            print(f"Avertissement: Impossible de charger le scaler depuis {scaler_path}. Un nouveau sera fitté.")

    if not loaded_scaler:
        print("Fitting d'un nouveau StandardScaler sur les données d'entraînement...")
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            if scaler_path:
                utils.save_object(scaler, scaler_path) # Sauvegarde le nouveau scaler
        except Exception as e:
            print(f"Erreur lors du fitting/scaling de X_train: {e}")
            return None, None, None
    else:
         print("Utilisation du scaler pré-chargé pour transformer X_train.")
         try:
             X_train_scaled = scaler.transform(X_train)
         except Exception as e:
            print(f"Erreur lors de la transformation de X_train avec le scaler chargé: {e}")
            return None, None, None


    X_test_scaled = None
    if X_test is not None:
        print("Transformation des données de test/validation avec le scaler.")
        try:
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Erreur lors de la transformation de X_test: {e}")
            # Peut-être retourner X_train_scaled et None pour X_test ? Ou tout None ?
            return X_train_scaled, None, scaler # Retourne au moins le train scaled

    print("Scaling terminé.")
    return X_train_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train, method=config.IMBALANCE_METHOD, k_neighbors=config.SMOTE_K_NEIGHBORS, random_state=config.SEED):
    """
    Applique une technique de ré-échantillonnage (SMOTE ou NearMiss) aux données d'entraînement.

    Args:
        X_train (np.ndarray): Features d'entraînement.
        y_train (np.ndarray): Étiquettes d'entraînement.
        method (str): 'smote' ou 'nearmiss'.
        k_neighbors (int): Nombre de voisins pour SMOTE.
        random_state (int): Seed pour la reproductibilité.

    Returns:
        tuple: Contient X_resampled, y_resampled. Retourne X_train, y_train si méthode='none' ou erreur.
    """
    if method.lower() == 'smote':
        print(f"Application de SMOTE avec k_neighbors={k_neighbors}...")
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"SMOTE terminé. Nouvelle shape de X: {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Erreur pendant SMOTE: {e}. Retour des données originales.")
            return X_train, y_train
    elif method.lower() == 'nearmiss':
        print(f"Application de NearMiss (version {config.NEAR_MISS_VERSION})...")
        try:
            nm = NearMiss(version=config.NEAR_MISS_VERSION)
            X_resampled, y_resampled = nm.fit_resample(X_train, y_train)
            print(f"NearMiss terminé. Nouvelle shape de X: {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Erreur pendant NearMiss: {e}. Retour des données originales.")
            return X_train, y_train
    elif method.lower() == 'class_weight' or method.lower() == 'none':
        print("Aucune technique de ré-échantillonnage appliquée (class_weight sera géré par le modèle ou aucune gestion).")
        return X_train, y_train
    else:
        print(f"Avertissement: Méthode d'imbalance '{method}' non reconnue. Retour des données originales.")
        return X_train, y_train

def reduce_dimension(X_train, y_train, X_test=None, method=config.REDUCTION_METHOD, n_components=config.PCA_N_COMPONENTS, k=config.SELECTKBEST_K, score_func=config.SELECTKBEST_SCORE_FUNC, reducer_path=config.DIM_REDUCER_PATH):
    """
    Applique une réduction de dimension (PCA ou SelectKBest).
    Fitté sur X_train, sauvegardé, puis appliqué à X_train et X_test.
    Si reducer_path existe, charge l'objet au lieu de le fitter.

    Args:
        X_train (np.ndarray): Données d'entraînement.
        y_train (np.ndarray): Étiquettes d'entraînement (requis pour SelectKBest).
        X_test (np.ndarray, optional): Données de test/validation. Defaults to None.
        method (str): 'pca', 'selectkbest', ou 'none'.
        n_components (float ou int): Pour PCA.
        k (int): Pour SelectKBest.
        score_func (callable): Pour SelectKBest.
        reducer_path (Path): Chemin pour sauvegarder/charger le réducteur.

    Returns:
        tuple: Contient :
            - np.ndarray: X_train réduit.
            - np.ndarray/None: X_test réduit (ou None).
            - object: Le réducteur fitté/chargé (PCA ou SelectKBest).
            Retourne (X_train, X_test, None) si method='none' ou erreur.
    """
    reducer = None
    loaded_reducer = False
    if method.lower() == 'none':
        print("Aucune réduction de dimension appliquée.")
        return X_train, X_test, None

    if reducer_path and reducer_path.exists():
        print(f"Chargement du réducteur existant depuis {reducer_path}")
        reducer = utils.load_object(reducer_path)
        if reducer:
            loaded_reducer = True
        else:
            print(f"Avertissement: Impossible de charger le réducteur depuis {reducer_path}. Un nouveau sera fitté.")

    X_train_reduced, X_test_reduced = None, None

    try:
        if not loaded_reducer:
            print(f"Fitting d'un nouveau réducteur ({method}) sur les données d'entraînement...")
            if method.lower() == 'pca':
                reducer = PCA(n_components=n_components, random_state=config.SEED)
                X_train_reduced = reducer.fit_transform(X_train)
                print(f"PCA: {reducer.n_components_} composantes sélectionnées expliquant {np.sum(reducer.explained_variance_ratio_)*100:.2f}% variance.")
            elif method.lower() == 'selectkbest':
                reducer = SelectKBest(score_func=score_func, k=k)
                X_train_reduced = reducer.fit_transform(X_train, y_train)
                print(f"SelectKBest: {k} meilleures features sélectionnées.")
            else:
                 print(f"Avertissement: Méthode de réduction '{method}' non reconnue.")
                 return X_train, X_test, None

            if reducer_path:
                utils.save_object(reducer, reducer_path)
        else:
            print(f"Utilisation du réducteur pré-chargé ({method}) pour transformer X_train.")
            X_train_reduced = reducer.transform(X_train)

        if X_test is not None:
            print("Transformation des données de test/validation avec le réducteur.")
            X_test_reduced = reducer.transform(X_test)

        print("Réduction de dimension terminée.")
        return X_train_reduced, X_test_reduced, reducer

    except Exception as e:
        print(f"Erreur pendant la réduction de dimension ({method}): {e}")
        return X_train, X_test, None # Retourne les données originales en cas d'erreur


def reshape_to_2d(X, rows=config.IMG_ROWS, cols=config.IMG_COLS):
    """
    Remodèle les données aplaties (n_samples, n_features) en format image 2D.
    Le format de sortie est (n_samples, rows, cols, 1) pour Keras/TF.

    Args:
        X (np.ndarray): Données d'entrée (shape: [n_samples, rows*cols]).
        rows (int): Nombre de lignes de l'image.
        cols (int): Nombre de colonnes de l'image.

    Returns:
        np.ndarray: Données remodelées (shape: [n_samples, rows, cols, 1]) ou None si erreur.
    """
    print(f"Remodelage des données en format ({rows}, {cols}, 1)...")
    try:
        if X.shape[1] != rows * cols:
            raise ValueError(f"Le nombre de features ({X.shape[1]}) ne correspond pas à rows*cols ({rows*cols})")
        # Ajouter un canal à la fin pour Keras/TF (format 'channels_last')
        X_reshaped = X.reshape(-1, rows, cols, 1)
        print(f"Remodelage terminé. Nouvelle shape: {X_reshaped.shape}")
        return X_reshaped
    except Exception as e:
        print(f"Erreur lors du remodelage: {e}")
        return None

print("Preprocess functions loaded.")
