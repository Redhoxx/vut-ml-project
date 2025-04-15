import numpy as np
import pandas as pd
import argparse

# Importer les modules locaux
import config
import data_loader
import preprocess
import utils
# Importer les modèles si nécessaire pour charger des formats spécifiques (ex: Keras)
# from tensorflow.keras.models import load_model

def predict(model_path, scaler_path=None, reducer_path=None, approach=config.SELECTED_APPROACH):
    """
    Charge un modèle entraîné et génère les prédictions sur le jeu de test.
    """
    print(f"Début de la prédiction avec l'approche: {approach}")
    print(f"Chargement du modèle depuis: {model_path}")

    # --- 1. Charger le Modèle et les Préprocesseurs ---
    model = utils.load_object(model_path) # Adapter si format .h5 etc.
    scaler = None
    reducer = None

    if config.APPLY_SCALING and scaler_path and scaler_path.exists():
        scaler = utils.load_object(scaler_path)
        if not scaler:
            print("Erreur critique: Impossible de charger le scaler nécessaire.")
            return
    elif config.APPLY_SCALING:
         print("Avertissement: Scaling appliqué à l'entraînement mais scaler non trouvé.")
         # Que faire? Arrêter ou continuer sans scaling (risqué) ?
         return

    if approach == 'classic' and config.REDUCTION_METHOD != 'none' and reducer_path and reducer_path.exists():
        reducer = utils.load_object(reducer_path)
        if not reducer:
             print("Erreur critique: Impossible de charger le réducteur de dimension nécessaire.")
             return
    elif approach == 'classic' and config.REDUCTION_METHOD != 'none':
         print("Avertissement: Réduction de dim appliquée à l'entraînement mais réducteur non trouvé.")
         return


    if model is None:
        print("Erreur critique: Impossible de charger le modèle.")
        return

    # --- 2. Charger les Données de Test ---
    X_test, test_ids = data_loader.load_test_data_from_csv()
    if X_test is None:
        print("Erreur lors du chargement des données de test. Arrêt.")
        return

    # --- 3. Appliquer le Prétraitement ---
    print("Application du prétraitement aux données de test...")
    X_test_processed = X_test.copy()

    # 3.1 Scaling
    if scaler:
        try:
            X_test_processed = scaler.transform(X_test_processed)
            print("Scaling appliqué.")
        except Exception as e:
             print(f"Erreur lors du scaling des données de test: {e}")
             return

    # 3.2 Logique spécifique à l'approche
    if approach == 'classic':
        # 3.2.1 Réduction de dimension
        if reducer:
            try:
                X_test_processed = reducer.transform(X_test_processed)
                print(f"Réduction de dimension ({config.REDUCTION_METHOD}) appliquée.")
            except Exception as e:
                print(f"Erreur lors de la réduction de dimension des données de test: {e}")
                return

    elif approach == 'cnn':
        # 3.2.1 Remodelage en 2D
        X_test_processed = preprocess.reshape_to_2d(X_test_processed)
        if X_test_processed is None:
            print("Erreur lors du remodelage des données de test. Arrêt.")
            return
        print("Données de test remodelées en 2D.")
    else:
        print(f"Erreur: Approche '{approach}' non reconnue pour le prétraitement.")
        return

    # --- 4. Faire les Prédictions ---
    print("Génération des prédictions...")
    try:
        predictions_proba = None # Si le modèle peut donner des probas
        if hasattr(model, 'predict_proba'):
             predictions_proba = model.predict_proba(X_test_processed)
             predictions = np.argmax(predictions_proba, axis=1)
        elif hasattr(model, 'predict'):
             predictions = model.predict(X_test_processed)
             # Pour les NNs Keras, predict peut retourner des probas, prendre l'argmax
             if approach == 'cnn' and predictions.ndim > 1 and predictions.shape[1] > 1:
                 predictions = np.argmax(predictions, axis=1)
        else:
             raise AttributeError("Le modèle chargé n'a pas de méthode predict ou predict_proba.")

        print(f"Prédictions générées. Shape: {predictions.shape}")

    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return

    # --- 5. Sauvegarder le Fichier de Soumission ---
    utils.save_submission_file(test_ids, predictions, filepath=config.SUBMISSION_FILEPATH)

    print("Prédiction terminée.")


if __name__ == '__main__':
    # Utiliser argparse pour spécifier quel modèle charger
    parser = argparse.ArgumentParser(description="Generate predictions for Kaggle submission.")
    parser.add_argument('--approach', type=str, default=config.SELECTED_APPROACH, choices=['classic', 'cnn'],
                        help='Specify which trained approach model to use.')
    # Ajouter des arguments pour les chemins si on ne veut pas utiliser config directement
    # parser.add_argument('--model_file', type=str, help='Path to the trained model file.')
    # parser.add_argument('--scaler_file', type=str, help='Path to the scaler file.')
    # parser.add_argument('--reducer_file', type=str, help='Path to the dimension reducer file.')

    args = parser.parse_args()

    # Déterminer les chemins basés sur l'approche choisie
    model_p = config.MODELS_DIR / args.approach / "best_model.pkl" # Adapter extension (.h5?)
    scaler_p = config.MODELS_DIR / args.approach / "scaler.pkl"
    reducer_p = None
    if args.approach == 'classic' and config.REDUCTION_METHOD != 'none':
         reducer_p = config.MODELS_DIR / args.approach / "dim_reducer.pkl"

    # Vérifier si les fichiers existent
    if not model_p.exists():
         print(f"Erreur: Fichier modèle non trouvé à {model_p}")
    else:
        predict(model_path=model_p, scaler_path=scaler_p, reducer_path=reducer_p, approach=args.approach)

    print("Fin du script de prédiction.")
