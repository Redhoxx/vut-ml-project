import numpy as np
import pandas as pd
from pathlib import Path
import time
from collections import OrderedDict
import warnings

import config

from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift

from sklearn.feature_selection import VarianceThreshold

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def calculate_statistical_features(matrix_2d, percentiles=config.STAT_PERCENTILES):
    """Calcule les caractéristiques statistiques d'une matrice 2D."""
    features = OrderedDict()
    flat_matrix = matrix_2d.flatten()
    features['stat_mean'] = np.nanmean(matrix_2d)
    features['stat_std'] = np.nanstd(matrix_2d)
    features['stat_variance'] = np.nanvar(matrix_2d)
    features['stat_median'] = np.nanmedian(matrix_2d)
    features['stat_min'] = np.nanmin(matrix_2d)
    features['stat_max'] = np.nanmax(matrix_2d)
    stat_min_val = features['stat_min']
    stat_max_val = features['stat_max']
    features['stat_range'] = stat_max_val - stat_min_val if not (
                np.isnan(stat_min_val) or np.isnan(stat_max_val)) else 0
    stat_std_val = features['stat_std']
    features['stat_skewness'] = skew(flat_matrix, nan_policy='omit') if stat_std_val != 0 and not np.isnan(
        stat_std_val) else 0
    features['stat_kurtosis'] = kurtosis(flat_matrix, nan_policy='omit') if stat_std_val != 0 and not np.isnan(
        stat_std_val) else -3
    features['stat_energy'] = np.nansum(matrix_2d ** 2)
    features['stat_rms'] = np.sqrt(np.nanmean(matrix_2d ** 2)) if not np.all(np.isnan(matrix_2d)) else 0
    if len(flat_matrix[~np.isnan(flat_matrix)]) > 0:
        percentiles_values = np.percentile(flat_matrix[~np.isnan(flat_matrix)], percentiles)
        for p, val in zip(percentiles, percentiles_values):
            features[f'stat_p{p}'] = val
        p25 = features.get('stat_p25', np.nan)
        p75 = features.get('stat_p75', np.nan)
        features['stat_iqr'] = p75 - p25 if not (np.isnan(p25) or np.isnan(p75)) else 0
    else:
        for p in percentiles:
            features[f'stat_p{p}'] = 0.0
        features['stat_iqr'] = 0.0
    final_features = OrderedDict()
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            final_features[key] = 0.0
        else:
            final_features[key] = value
    return final_features


def get_stat_feature_names(percentiles=config.STAT_PERCENTILES):
    """Retourne la liste ordonnée des noms de caractéristiques statistiques."""
    names = ['stat_mean', 'stat_std', 'stat_variance', 'stat_median', 'stat_min', 'stat_max',
             'stat_range', 'stat_skewness', 'stat_kurtosis', 'stat_energy', 'stat_rms']
    names.extend([f'stat_p{p}' for p in percentiles])
    names.append('stat_iqr')
    return names


def calculate_glcm_features(matrix_2d, distances=config.GLCM_DISTANCES, angles_rad=config.GLCM_ANGLES_RAD, angles_deg=config.GLCM_ANGLES_DEG,
                            levels=config.GLCM_LEVELS, properties=config.GLCM_PROPERTIES):
    """Calcule les caractéristiques GLCM d'une matrice 2D."""
    features = OrderedDict()
    expected_names = get_glcm_feature_names(distances, angles_deg, properties)
    min_val, max_val = np.nanmin(matrix_2d), np.nanmax(matrix_2d)
    if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
        for name in expected_names:
            features[name] = 0.0
        return features
    image_norm = matrix_2d - min_val
    image_norm = (image_norm / (max_val - min_val) * (levels - 1)).astype(np.uint8)
    try:
        glcm = graycomatrix(image_norm, distances=distances, angles=angles_rad, levels=levels, symmetric=True,
                            normed=True)
        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            for i_d, d in enumerate(distances):
                for i_a, a_deg in enumerate(angles_deg):
                    feature_name = f'glcm_{prop}_d{d}_a{a_deg}'
                    prop_value = prop_values[i_d, i_a]
                    features[feature_name] = 0.0 if np.isnan(prop_value) or np.isinf(prop_value) else prop_value
    except ValueError as e:
        for name in expected_names:
            features[name] = 0.0
    final_features = OrderedDict()
    for name in expected_names:
        value = features.get(name, 0.0)
        if np.isnan(value) or np.isinf(value):
            final_features[name] = 0.0
        else:
            final_features[name] = value
    return final_features


def get_glcm_feature_names(distances=config.GLCM_DISTANCES, angles_deg=config.GLCM_ANGLES_DEG, properties=config.GLCM_PROPERTIES):
    """Retourne la liste ordonnée des noms de caractéristiques GLCM."""
    names = []
    for prop in properties:
        for d in distances:
            for a_deg in angles_deg:
                names.append(f'glcm_{prop}_d{d}_a{a_deg}')
    return names


def calculate_fft_features(matrix_2d, percentiles=config.FFT_PERCENTILES):
    """Calcule les caractéristiques FFT d'une matrice 2D."""
    features = OrderedDict()
    expected_names = get_fft_feature_names(percentiles)
    if np.isnan(matrix_2d).any():
        mean_val = np.nanmean(matrix_2d)
        if np.isnan(mean_val):
            mean_val = 0
        matrix_2d = np.nan_to_num(matrix_2d, nan=mean_val)
    try:
        fft_result = fft2(matrix_2d)
        fft_shifted = fftshift(fft_result)
        magnitude_spectrum = np.abs(fft_shifted)
        magnitude_flat = magnitude_spectrum.flatten()
        features['fft_mean_magnitude'] = np.mean(magnitude_spectrum)
        features['fft_std_magnitude'] = np.std(magnitude_spectrum)
        features['fft_max_magnitude'] = np.max(magnitude_spectrum)
        features['fft_min_magnitude'] = np.min(magnitude_spectrum)
        center_row, center_col = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        features['fft_dc_magnitude'] = magnitude_spectrum[center_row, center_col]
        magnitude_no_dc = magnitude_spectrum.copy()
        magnitude_no_dc[center_row, center_col] = 0
        features['fft_mean_magnitude_no_dc'] = np.mean(magnitude_no_dc[magnitude_no_dc != 0]) if np.any(
            magnitude_no_dc != 0) else 0
        features['fft_std_magnitude_no_dc'] = np.std(magnitude_no_dc[magnitude_no_dc != 0]) if np.any(
            magnitude_no_dc != 0) else 0
        features['fft_spectral_energy'] = np.sum(magnitude_spectrum ** 2)
        percentile_values = np.percentile(magnitude_flat, percentiles)
        for p, val in zip(percentiles, percentile_values):
            features[f'fft_mag_p{p}'] = val

    except Exception as e:
        for name in expected_names:
            features[name] = 0.0
    final_features = OrderedDict()
    for name in expected_names:
        value = features.get(name, 0.0)
        if np.isnan(value) or np.isinf(value):
            final_features[name] = 0.0
        else:
            final_features[name] = value
    return final_features


def get_fft_feature_names(percentiles=config.FFT_PERCENTILES):
    """Retourne la liste ordonnée des noms de caractéristiques FFT."""
    names = [
        'fft_mean_magnitude', 'fft_std_magnitude', 'fft_max_magnitude', 'fft_min_magnitude',
        'fft_dc_magnitude', 'fft_mean_magnitude_no_dc', 'fft_std_magnitude_no_dc',
        'fft_spectral_energy'
    ]
    names.extend([f'fft_mag_p{p}' for p in percentiles])
    return names


def process_data(input_csv_path, output_csv_path, has_target=True,
                 cols_to_drop_variance=None, cols_to_drop_corr=None):
    """
    Charge les données, extrait les caractéristiques, applique les filtres (si train)
    et sauvegarde le résultat.

    Retourne les listes des colonnes supprimées par variance et corrélation (si train),
    ou None, None (si test).
    """
    print(f"\n--- Traitement de {input_csv_path} ---")

    # --- Loading et Reshaping ---
    feature_cols_original = [f'f_{i}' for i in range(config.NUM_FEATURES_ORIGINAL)]
    cols_to_load = ['ID'] + feature_cols_original
    if has_target:
        cols_to_load.insert(1, config.TARGET_COLUMN)

    print(f"Chargement des données depuis : {input_csv_path}")
    if not Path(input_csv_path).is_file():
        raise FileNotFoundError(f"Fichier {input_csv_path} non trouvé.")
    try:
        df = pd.read_csv(input_csv_path, usecols=cols_to_load)
        print(f"Données chargées : {df.shape[0]} échantillons, {df.shape[1]} colonnes.")
    except Exception as e:
        print(f"Erreur lors du chargement du CSV : {e}")
        raise

    ids = df[[config.ID_COLUMN]]
    y = df[[config.TARGET_COLUMN]] if has_target else None
    X_flat = df[feature_cols_original].values

    print(f"Remodelage en images {config.IMG_ROWS}x{config.IMG_COLS}...")
    if X_flat.shape[1] != config.NUM_FEATURES_ORIGINAL:
        raise ValueError(f"Nombre incorrect de caractéristiques : {X_flat.shape[1]} vs {config.NUM_FEATURES_ORIGINAL}")
    try:
        X_2d = X_flat.reshape(-1, config.IMG_ROWS, config.IMG_COLS)
        print(f"Remodelage terminé. Shape de X_2d : {X_2d.shape}")
    except Exception as e:
        print(f"Erreur lors du remodelage des données : {e}")
        raise

    # --- Features extraction ---
    all_extracted_features_list = []
    stat_feature_names = get_stat_feature_names(config.STAT_PERCENTILES)
    glcm_feature_names = get_glcm_feature_names(config.GLCM_DISTANCES, config.GLCM_ANGLES_DEG, config.GLCM_PROPERTIES)
    fft_feature_names = get_fft_feature_names(config.FFT_PERCENTILES)
    calculated_feature_names = stat_feature_names + glcm_feature_names + fft_feature_names
    n_features_expected = len(calculated_feature_names)

    print(
        f"Début de l'extraction pour {len(X_2d)} échantillons ({n_features_expected} caractéristiques par échantillon)...")
    start_time_extraction = time.time()

    for i in tqdm(range(len(X_2d)), desc="Extraction des Caractéristiques"):
        img = X_2d[i]
        stat_features = calculate_statistical_features(img, config.STAT_PERCENTILES)
        glcm_features = calculate_glcm_features(img, config.GLCM_DISTANCES, config.GLCM_ANGLES_RAD, config.GLCM_ANGLES_DEG, config.GLCM_LEVELS,
                                                config.GLCM_PROPERTIES)
        fft_features = calculate_fft_features(img, config.FFT_PERCENTILES)
        all_calculated_features_dict = {**stat_features, **glcm_features, **fft_features}
        sample_features_ordered = [all_calculated_features_dict.get(name, 0.0) for name in calculated_feature_names]
        all_extracted_features_list.append(sample_features_ordered)

    end_time_extraction = time.time()
    print(f"Extraction terminée en {end_time_extraction - start_time_extraction:.2f} secondes.")

    X_engineered = pd.DataFrame(all_extracted_features_list, columns=calculated_feature_names)
    print(f"Nombre de caractéristiques avant filtrage : {X_engineered.shape[1]}")

    # --- Managing NaN/Inf ---
    if X_engineered.isnull().sum().sum() > 0:
        print(f"\nDétection de {X_engineered.isnull().sum().sum()} valeurs NaN. Imputation par 0...")
        X_engineered = X_engineered.fillna(0.0)  # Imputation simple par 0 pour l'exemple
        if X_engineered.isnull().sum().sum() > 0:
            print("Attention : NaNs persistants après imputation. Vérifiez les colonnes.")
        else:
            print("Imputation des NaNs terminée.")
    else:
        print("\nAucune valeur NaN détectée dans les caractéristiques extraites.")

    X_engineered.replace([np.inf, -np.inf], 0.0, inplace=True)
    print("Vérification et remplacement des Inf terminés.")

    features_removed_variance = None
    features_removed_corr = None

    if has_target:
        print(f"\nApplication du seuil de variance (seuil = {config.VARIANCE_THRESHOLD_VALUE}) sur les données train...")
        selector_var = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD_VALUE)
        try:
            selector_var.fit(X_engineered)
            features_to_keep_var_mask = selector_var.get_support()
            features_removed_variance = X_engineered.columns[~features_to_keep_var_mask].tolist()

            if len(features_removed_variance) > 0:
                print(f"Suppression de {len(features_removed_variance)} caractéristiques à faible variance :")
                X_engineered = X_engineered.loc[:, features_to_keep_var_mask]
            else:
                print("Aucune caractéristique supprimée par le seuil de variance.")
            print(f"Nombre de caractéristiques après filtre de variance : {X_engineered.shape[1]}")
        except ValueError as e:
            print(f"Erreur lors de l'application du seuil de variance : {e}")
            print("Vérifiez les NaNs ou les types de données non numériques restants.")
            features_removed_variance = []

        print(f"\nApplication du seuil de corrélation (seuil = {config.CORRELATION_THRESHOLD_VALUE}) sur les données train...")
        corr_matrix = X_engineered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        features_removed_corr = [column for column in upper_tri.columns if
                                 any(upper_tri[column] > config.CORRELATION_THRESHOLD_VALUE)]

        if len(features_removed_corr) > 0:
            print(f"Suppression de {len(features_removed_corr)} caractéristiques fortement corrélées :")
            X_engineered = X_engineered.drop(columns=features_removed_corr)
        else:
            print("Aucune caractéristique supprimée par le seuil de corrélation.")
        print(f"Nombre de caractéristiques après filtre de corrélation : {X_engineered.shape[1]}")

    else:
        if cols_to_drop_variance:
            cols_exist_var = [col for col in cols_to_drop_variance if col in X_engineered.columns]
            if cols_exist_var:
                print(
                    f"Suppression de {len(cols_exist_var)} caractéristiques à faible variance (déterminées sur train)...")
                X_engineered = X_engineered.drop(columns=cols_exist_var)
            print(f"Nombre de caractéristiques après filtre de variance : {X_engineered.shape[1]}")

        if cols_to_drop_corr:
            cols_exist_corr = [col for col in cols_to_drop_corr if col in X_engineered.columns]
            if cols_exist_corr:
                print(
                    f"Suppression de {len(cols_exist_corr)} caractéristiques fortement corrélées (déterminées sur train)...")
                X_engineered = X_engineered.drop(columns=cols_exist_corr)
            print(f"Nombre de caractéristiques après filtre de corrélation : {X_engineered.shape[1]}")

    # --- Save ---
    print(f"\nPréparation de la sauvegarde vers : {output_csv_path}")
    to_concat = [ids.reset_index(drop=True)]
    if has_target and y is not None:
        to_concat.append(y.reset_index(drop=True))
    to_concat.append(X_engineered.reset_index(drop=True))

    df_final_output = pd.concat(to_concat, axis=1)
    print(f"Shape du DataFrame final à sauvegarder : {df_final_output.shape}")
    print(f"Colonnes finales : {df_final_output.columns.tolist()}")

    try:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_final_output.to_csv(output_csv_path, index=False)
        print(f"Caractéristiques finales sauvegardées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du CSV final : {e}")

    if has_target:
        return features_removed_variance, features_removed_corr
    else:
        return None, None


if __name__ == "__main__":
    cols_var_to_drop, cols_corr_to_drop = process_data(
        config.TRAIN_PROCESSED_PATH, config.TRAIN_ENGINEERED_FEATURES_PATH, has_target=True
    )

    process_data(
        config.TEST_PROCESSED_PATH, config.TEST_ENGINEERED_FEATURES_PATH, has_target=False,
        cols_to_drop_variance=cols_var_to_drop,
        cols_to_drop_corr=cols_corr_to_drop
    )

    print("\nPipeline terminé.")