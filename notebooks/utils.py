import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from notebooks import config


def set_seed(seed_value=config.SEED):
    """Fixe les seeds pour la reproductibilité."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed_value)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"Seed set to {seed_value}")

def save_object(obj, filepath):
    """Sauvegarde un objet Python (modèle, scaler, etc.) avec joblib."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, filepath)
        print(f"Object saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        return False

def load_object(filepath):
    """Charge un objet Python depuis un fichier avec joblib."""
    try:
        filepath = Path(filepath)
        if filepath.is_file():
            obj = joblib.load(filepath)
            print(f"Object loaded from {filepath}")
            return obj
        else:
            print(f"Error: File not found at {filepath}")
            return None
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues, normalize=False, save_path=None):
    """Affiche et sauvegarde (optionnel) la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        print("Normalized confusion matrix")
    else:
        fmt = 'd'
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap, ax=ax, values_format=fmt)

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
    plt.show()

def plot_training_history(history, metrics=['loss', 'accuracy', 'balanced_accuracy'], save_path=None):
    """Affiche l'historique d'entraînement (perte et métriques)."""
    if not hasattr(history, 'history'):
        print("Warning: History object does not seem to have a 'history' attribute.")
        return

    history_dict = history.history
    epochs = range(1, len(history_dict.get(metrics[0], [])) + 1)

    num_plots = len(metrics)
    plt.figure(figsize=(6 * num_plots, 5))

    for i, metric in enumerate(metrics):
        if metric in history_dict:
            plt.subplot(1, num_plots, i + 1)
            plt.plot(epochs, history_dict[metric], 'bo-', label=f'Training {metric}')
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                plt.plot(epochs, history_dict[val_metric], 'ro-', label=f'Validation {metric}')
            plt.title(f'Training and Validation {metric.capitalize()}')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
        else:
             print(f"Warning: Metric '{metric}' not found in history.")

    plt.tight_layout()
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving training history plot: {e}")
    plt.show()


def save_submission_file(ids, predictions, filepath=config.SUBMISSION_FILEPATH):
    """
    Crée et sauvegarde le fichier de soumission au format CSV.
    Utilise le format de test_format.csv comme référence pour les IDs si possible.
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        test_format_df = None
        if config.TEST_FORMAT_PATH.is_file():
             test_format_df = pd.read_csv(config.TEST_FORMAT_PATH)
             if 'ID' not in test_format_df.columns:
                 print("Warning: 'ID' column not found in test_format.csv. Using provided IDs.")
                 test_format_df = None
             elif len(test_format_df) != len(ids):
                 print(f"Warning: Length mismatch between test_format.csv ({len(test_format_df)}) and provided IDs ({len(ids)}). Using provided IDs.")
                 test_format_df = None
             else:
                 ids = test_format_df['ID'].values

        if len(ids) != len(predictions):
             raise ValueError(f"Length mismatch: {len(ids)} IDs and {len(predictions)} predictions.")

        submission_df = pd.DataFrame({config.ID_COL: ids, config.TARGET_COL: predictions})

        submission_df[config.ID_COL] = submission_df[config.ID_COL].astype(int)
        submission_df[config.TARGET_COL] = submission_df[config.TARGET_COL].astype(int)

        submission_df.to_csv(filepath, index=False)
        print(f"Submission file created successfully at: {filepath}")

    except Exception as e:
        print(f"Error creating submission file: {e}")

set_seed(config.SEED)

print("Utils loaded.")