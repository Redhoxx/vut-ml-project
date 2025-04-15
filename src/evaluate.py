from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import pandas as pd

import config
import utils # Pour plot_confusion_matrix

def calculate_metrics(y_true, y_pred, y_proba=None, labels=None, target_names=None):
    """
    Calcule un ensemble de métriques de classification.

    Args:
        y_true (np.ndarray): Vraies étiquettes.
        y_pred (np.ndarray): Étiquettes prédites.
        y_proba (np.ndarray, optional): Probabilités prédites (pour AUC-PR etc.). Defaults to None.
        labels (list, optional): Liste ordonnée des labels (ex: [0, 1, 2]). Defaults to None.
        target_names (list, optional): Noms des classes correspondants aux labels. Defaults to None.


    Returns:
        dict: Dictionnaire contenant les scores des métriques.
    """
    if labels is None:
        labels = sorted(np.unique(y_true))
    if target_names is None:
        target_names = [f"Class {l}" for l in labels]

    metrics = {}
    print("\n--- Évaluation des Performances ---")

    # Métriques de base
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Métriques pour données déséquilibrées (Macro et Weighted F1, Precision, Recall)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Ajouter d'autres métriques si nécessaire (ex: AUC-PR si y_proba est fourni)
    # if y_proba is not None and config.NUM_CLASSES == 2: # Exemple pour binaire
    #     from sklearn.metrics import average_precision_score
    #     metrics['auc_pr'] = average_precision_score(y_true, y_proba[:, 1])
    # elif y_proba is not None and config.NUM_CLASSES > 2:
        # Calculer AUC-PR multi-classe (ex: one-vs-rest) si nécessaire
        # pass

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")

    # Rapport de classification détaillé
    print("\nClassification Report:")
    try:
        report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    # Matrice de confusion
    print("\nConfusion Matrix:")
    utils.plot_confusion_matrix(y_true, y_pred, classes=target_names, title='Confusion Matrix')

    return metrics

print("Evaluate functions loaded.")