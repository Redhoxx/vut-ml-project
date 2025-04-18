import numpy as np
from pathlib import Path
from sklearn.feature_selection import f_classif

# --- Chemins Principaux ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path('.').resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models_trained"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# --- Chemins Données Brutes ---
TRAIN_IMG_DIR = RAW_DATA_DIR / "train"
TEST_IMG_DIR = RAW_DATA_DIR / "test"
TRAIN_LABELS_PATH = RAW_DATA_DIR / "label_train.csv"
TEST_FORMAT_PATH = RAW_DATA_DIR / "test_format.csv"

# --- Chemins Données Prétraitées / Extraites ---
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.csv"
TRAIN_ENGINEERED_FEATURES_PATH = PROCESSED_DATA_DIR / 'train_engineered_features.csv'
TEST_ENGINEERED_FEATURES_PATH = PROCESSED_DATA_DIR / 'test_engineered_features.csv'
OUTPUT_FEATURES_TRAIN_DATA = PROCESSED_DATA_DIR / 'train_features_selected.csv'

# --- Paramètres des Données ---
IMG_ROWS = 72
IMG_COLS = 48
NUM_FEATURES_ORIGINAL = IMG_ROWS * IMG_COLS
NUM_FEATURES = NUM_FEATURES_ORIGINAL
NUM_CLASSES = 3
TARGET_COL = 'target'
ID_COL = 'ID'

# --- Contrôle de l'Expérience ---
SELECTED_APPROACH = 'classic'

# --- Paramètres d'Extraction de Caractéristiques ---
# >> Paramètres Statistiques
STAT_PERCENTILES = [10, 25, 75, 90]
# >> Paramètres GLCM
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES_DEG = [0, 45, 90, 135]
GLCM_ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
GLCM_LEVELS = 256
# >> Paramètres FFT
FFT_PERCENTILES = [10, 25, 50, 75, 90]

# --- Paramètres de Suppression de Bruit ---
VARIANCE_THRESHOLD_VALUE = 0.01
CORRELATION_THRESHOLD_VALUE = 0.95

# --- Paramètres de Prétraitement ---
APPLY_SCALING = False
HANDLE_IMBALANCE = True
SCALER_PATH = MODELS_DIR / SELECTED_APPROACH / "scaler.pkl"
IMBALANCE_METHOD = 'class_weight' # ou 'nearmiss', 'smote'
SMOTE_K_NEIGHBORS = 5
NEAR_MISS_VERSION = 3

# --- Paramètres d'Entraînement ---
SEED = 42
K_FOLDS = 5
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# --- Paramètres Spécifiques aux Modèles ---

# > Approche 1: Classique (feature engineering with a LGBM)
CLASSIC_MODEL_TYPE = 'lgbm'
# Best hyperparameters afer Optuna process
LGBM_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': NUM_CLASSES,
    'n_estimators': 1000,
    'learning_rate': 0.18762220075897995,
    'num_leaves': 194,
    'max_depth': 10,
    'min_child_samples': 85,
    'feature_fraction': 0.6810846735404653,
    'bagging_fraction': 0.9887756365183054,
    'bagging_freq': 3,
    'lambda_l1': 0.02515892173562238,
    'lambda_l2': 0.002927249328048098,
    'class_weight': 'balanced',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED
}
LGBM_EARLY_STOPPING_ROUNDS = 100

# > Approche 2: CNN
CNN_INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_DENSE_UNITS = [256, 128]
CNN_DROPOUT_RATE = 0.4
CNN_ACTIVATION = 'relu'
CNN_OUTPUT_ACTIVATION = 'softmax'
CNN_OPTIMIZER = 'adam'
CNN_LOSS = 'sparse_categorical_crossentropy'

# --- Paramètres d'Évaluation  ---
PRIMARY_METRIC = 'balanced_accuracy'
METRICS_TO_LOG = ['balanced_accuracy', 'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

# --- Sauvegarde des Modèles ---
MODEL_SAVE_PATH = MODELS_DIR / SELECTED_APPROACH / f"{"feat_extra" if SELECTED_APPROACH=='classic' else 'cnn'}_final_model.pkl"

# --- Soumission Kaggle ---
SUBMISSION_FILENAME = f"submission_{SELECTED_APPROACH}_{"feat_extra" if SELECTED_APPROACH=='classic' else 'cnn'}.csv"
SUBMISSION_FILEPATH = SUBMISSIONS_DIR / SUBMISSION_FILENAME

print("Config complète chargée.")
print(f"Selected Approach: {SELECTED_APPROACH}")
print(f"Data Directory: {DATA_DIR}")
print(f"Models Directory: {MODELS_DIR}")
print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")