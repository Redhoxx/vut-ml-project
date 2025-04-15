import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import time
import argparse

# Importer les modules locaux
import config
import data_loader
import preprocess
import evaluate
import utils
# Importer les modèles (à créer)
# from models import classic_models, cnn_model

def train():
    """Fonction principale pour l'entraînement et l'évaluation."""
    start_time = time.time()
    utils.set_seed(config.SEED)

    # --- 1. Charger les Données ---
    # Utiliser la version CSV simplifiée pour l'instant
    X, y, ids = data_loader.load_train_data_from_csv()
    if X is None:
        print("Erreur lors du chargement des données. Arrêt.")
        return

    # --- 2. Validation Croisée Stratifiée ---
    skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
    fold_scores = []
    fold_histories = [] # Pour stocker l'historique d'entraînement des NNs
    best_fold_score = -np.inf # Ou np.inf si la métrique doit être minimisée
    best_model = None
    best_scaler = None
    best_reducer = None
    best_fold_num = -1

    print(f"\nDébut de la validation croisée ({config.K_FOLDS} folds) pour l'approche: {config.SELECTED_APPROACH}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{config.K_FOLDS} ---")
        fold_start_time = time.time()

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        print(f"Train fold shape: {X_train_fold.shape}, Val fold shape: {X_val_fold.shape}")

        # --- 3. Prétraitement spécifique au Fold (et à l'approche) ---
        scaler_fold = None
        reducer_fold = None

        # 3.1 Scaling (commun mais fitté sur le fold)
        if config.APPLY_SCALING:
             # Ne pas sauvegarder/charger le scaler pour chaque fold, juste fitter/transformer
             temp_scaler = preprocess.StandardScaler()
             X_train_fold = temp_scaler.fit_transform(X_train_fold)
             X_val_fold = temp_scaler.transform(X_val_fold)
             scaler_fold = temp_scaler # Garder pour potentielle sauvegarde du meilleur fold
             print("Scaling appliqué au fold.")

        # 3.2 Gestion Imbalance (appliqué uniquement sur X_train_fold)
        if config.HANDLE_IMBALANCE and config.IMBALANCE_METHOD not in ['class_weight', 'none']:
            X_train_fold, y_train_fold = preprocess.handle_imbalance(
                X_train_fold, y_train_fold, method=config.IMBALANCE_METHOD
            )
            print(f"Imbalance ({config.IMBALANCE_METHOD}) gérée pour le fold d'entraînement.")

        # 3.3 Logique spécifique à l'approche
        if config.SELECTED_APPROACH == 'classic':
            print("Approche Classique sélectionnée.")
            # 3.3.1 Réduction de dimension (appliquée après scaling/imbalance)
            if config.REDUCTION_METHOD != 'none':
                 # Fitter/transformer sur le fold, ne pas sauvegarder/charger ici
                 temp_reducer = None
                 if config.REDUCTION_METHOD == 'pca':
                     temp_reducer = preprocess.PCA(n_components=config.PCA_N_COMPONENTS, random_state=config.SEED)
                     X_train_fold = temp_reducer.fit_transform(X_train_fold)
                     X_val_fold = temp_reducer.transform(X_val_fold)
                 elif config.REDUCTION_METHOD == 'selectkbest':
                     temp_reducer = preprocess.SelectKBest(score_func=config.SELECTKBEST_SCORE_FUNC, k=config.SELECTKBEST_K)
                     X_train_fold = temp_reducer.fit_transform(X_train_fold, y_train_fold)
                     X_val_fold = temp_reducer.transform(X_val_fold)
                 reducer_fold = temp_reducer # Garder pour sauvegarde
                 print(f"Réduction de dimension ({config.REDUCTION_METHOD}) appliquée au fold.")

            # --- 4. Définir et Entraîner le Modèle Classique ---
            print(f"Entraînement du modèle classique: {config.CLASSIC_MODEL_TYPE}")
            # TODO: Importer et instancier le modèle depuis classic_models.py
            # Ex: model = classic_models.get_model(config.CLASSIC_MODEL_TYPE, **config.XGB_PARAMS)
            model = None # Placeholder
            if config.CLASSIC_MODEL_TYPE == 'xgb':
                 # from xgboost import XGBClassifier
                 # model = XGBClassifier(**config.XGB_PARAMS)
                 print("Placeholder: Définir et entraîner XGBoost")
                 pass # Placeholder pour l'entraînement
            elif config.CLASSIC_MODEL_TYPE == 'rf':
                 # from sklearn.ensemble import RandomForestClassifier
                 # rf_params = config.RF_PARAMS.copy()
                 # if config.IMBALANCE_METHOD == 'class_weight':
                 #     rf_params['class_weight'] = 'balanced'
                 # model = RandomForestClassifier(**rf_params)
                 print("Placeholder: Définir et entraîner RandomForest")
                 pass # Placeholder pour l'entraînement
            # ... autres modèles classiques ...

            # Placeholder pour l'entraînement réel
            # try:
            #     train_params = {}
            #     if config.CLASSIC_MODEL_TYPE == 'xgb' and config.IMBALANCE_METHOD == 'class_weight':
            #          # Calculer les poids pour XGBoost si nécessaire (plus complexe)
            #          pass
            #     elif config.IMBALANCE_METHOD == 'class_weight':
            #          # Pour les modèles sklearn qui le supportent directement
            #           weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            #           class_weights_dict = dict(zip(np.unique(y_train_fold), weights))
            #           # Ajouter aux paramètres si le modèle le prend (ex: RandomForest, SVM)
            #           # Pour XGBoost, utiliser 'scale_pos_weight' (binaire) ou ajuster les poids manuellement

            #     # model.fit(X_train_fold, y_train_fold, **train_params) # Ajouter params spécifiques
            # except Exception as e:
            #     print(f"Erreur pendant l'entraînement du fold {fold+1}: {e}")
            #     continue # Passer au fold suivant

            history = None # Pas d'historique détaillé pour la plupart des modèles classiques

        elif config.SELECTED_APPROACH == 'cnn':
            print("Approche CNN sélectionnée.")
            # 3.3.1 Remodelage en 2D (AVANT scaling si scaler standard, APRES si normalisation image)
            X_train_fold = preprocess.reshape_to_2d(X_train_fold)
            X_val_fold = preprocess.reshape_to_2d(X_val_fold)
            if X_train_fold is None or X_val_fold is None:
                 print(f"Erreur de remodelage dans le fold {fold+1}. Arrêt du fold.")
                 continue
            print("Données remodelées en 2D pour le CNN.")

            # --- 4. Définir et Entraîner le Modèle CNN ---
            print("Entraînement du modèle CNN")
            # TODO: Importer et instancier le modèle depuis cnn_model.py
            # Ex: model = cnn_model.build_cnn_model(...)
            # Ex: model.compile(optimizer=config.CNN_OPTIMIZER, loss=config.CNN_LOSS, metrics=['accuracy', tf.keras.metrics.AUC(...)])
            model = None # Placeholder

            # Gestion de l'imbalance via class_weight pour NNs
            class_weights_dict = None
            if config.HANDLE_IMBALANCE and config.IMBALANCE_METHOD == 'class_weight':
                weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
                class_weights_dict = dict(zip(np.unique(y_train_fold), weights))
                print(f"Utilisation des poids de classe: {class_weights_dict}")

            # Placeholder pour l'entraînement réel
            # try:
            #     callbacks = []
            #     if config.EARLY_STOPPING_PATIENCE > 0:
            #          from tensorflow.keras.callbacks import EarlyStopping
            #          early_stopping = EarlyStopping(monitor='val_loss', # ou une autre métrique
            #                                         patience=config.EARLY_STOPPING_PATIENCE,
            #                                         restore_best_weights=True)
            #          callbacks.append(early_stopping)

            #     history = model.fit(X_train_fold, y_train_fold,
            #                         epochs=config.EPOCHS,
            #                         batch_size=config.BATCH_SIZE,
            #                         validation_data=(X_val_fold, y_val_fold),
            #                         class_weight=class_weights_dict,
            #                         callbacks=callbacks,
            #                         verbose=1) # Mettre à 0 ou 2 pour moins de logs
            #     fold_histories.append(history.history)
            # except Exception as e:
            #     print(f"Erreur pendant l'entraînement CNN du fold {fold+1}: {e}")
            #     continue
            history = None # Placeholder

        else:
            print(f"Erreur: Approche '{config.SELECTED_APPROACH}' non reconnue.")
            return

        # --- 5. Évaluation sur le Fold de Validation ---
        if model is not None:
            print("Évaluation du fold...")
            # TODO: Faire les prédictions
            # y_pred_val = model.predict(X_val_fold)
            # Si sortie softmax (NNs), prendre l'argmax
            # if config.SELECTED_APPROACH == 'cnn':
            #     y_pred_val = np.argmax(y_pred_val, axis=1)

            y_pred_val = np.random.randint(0, config.NUM_CLASSES, size=y_val_fold.shape) # **PLACEHOLDER PREDICTIONS**

            fold_metrics = evaluate.calculate_metrics(y_val_fold, y_pred_val, labels=list(range(config.NUM_CLASSES)))
            fold_scores.append(fold_metrics)

            # --- 6. Sauvegarde du Meilleur Modèle (basé sur la métrique primaire) ---
            current_score = fold_metrics.get(config.PRIMARY_METRIC, -np.inf)
            if current_score > best_fold_score:
                print(f"Nouveau meilleur score ({config.PRIMARY_METRIC}): {current_score:.4f} (Fold {fold+1})")
                best_fold_score = current_score
                best_model = model # Garder la référence au modèle (ou sauvegarder directement)
                best_scaler = scaler_fold # Sauvegarder le scaler de ce fold
                best_reducer = reducer_fold # Sauvegarder le réducteur de ce fold
                best_fold_num = fold + 1
                # Sauvegarder immédiatement ou juste garder la référence/index
                # utils.save_object(model, config.MODEL_SAVE_PATH + f"_fold{fold+1}.pkl") # Exemple
        else:
             print(f"Erreur: Aucun modèle entraîné pour le fold {fold+1}.")


        fold_time = time.time() - fold_start_time
        print(f"Fold {fold+1} terminé en {fold_time:.2f} secondes.")

    # --- 7. Afficher les Résultats Moyens de la CV ---
    if fold_scores:
        avg_scores = pd.DataFrame(fold_scores).mean().to_dict()
        std_scores = pd.DataFrame(fold_scores).std().to_dict()
        print("\n--- Résultats Moyens de la Validation Croisée ---")
        for metric, score in avg_scores.items():
            print(f"  Moyenne {metric}: {score:.4f} +/- {std_scores.get(metric, 0):.4f}")
        print(f"Meilleur score ({config.PRIMARY_METRIC}) obtenu au fold {best_fold_num}: {best_fold_score:.4f}")
    else:
        print("\nAucun fold n'a pu être évalué.")

    # --- 8. Sauvegarder le Meilleur Modèle et les Préprocesseurs associés ---
    # (Alternative: ré-entraîner sur tout X avec les meilleurs hyperparamètres trouvés)
    if best_model is not None:
        print(f"\nSauvegarde du meilleur modèle (du fold {best_fold_num}) et des préprocesseurs...")
        # TODO: Implémenter la sauvegarde réelle du modèle (format .pkl, .h5, .pt...)
        # utils.save_object(best_model, config.MODEL_SAVE_PATH) # Adapter l'extension
        if best_scaler:
            utils.save_object(best_scaler, config.SCALER_PATH)
        if best_reducer:
             utils.save_object(best_reducer, config.DIM_REDUCER_PATH)
    else:
         print("\nAucun meilleur modèle à sauvegarder.")


    # Afficher l'historique moyen pour les NNs si disponible
    # if fold_histories and config.SELECTED_APPROACH == 'cnn':
    #     # Agréger ou afficher l'historique du meilleur fold
    #     pass # Placeholder

    total_time = time.time() - start_time
    print(f"\nEntraînement complet terminé en {total_time:.2f} secondes.")

if __name__ == '__main__':
    # # Exemple pour lancer via ligne de commande (optionnel)
    # parser = argparse.ArgumentParser(description="Train a model for 5G base station classification.")
    # parser.add_argument('--approach', type=str, default=config.SELECTED_APPROACH, choices=['classic', 'cnn'],
    #                     help='Select the modeling approach.')
    # args = parser.parse_args()
    # config.SELECTED_APPROACH = args.approach # Met à jour la config si argument fourni

    print(f"Début de l'entraînement avec la configuration...")
    train()
    print("Fin de l'entraînement.")
