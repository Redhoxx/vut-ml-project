from pathlib import Path
from sklearn.feature_selection import f_classif

# --- Chemins Principaux ---
BASE_DIR = Path(__file__).resolve().parent.parent
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

# --- Chemins Données Prétraitées (Optionnel) ---
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

# --- Paramètres des Données ---
IMG_ROWS = 72
IMG_COLS = 48
NUM_FEATURES = IMG_ROWS * IMG_COLS
NUM_CLASSES = 3
TARGET_COL = 'target'
ID_COL = 'ID'

# --- Contrôle de l'Expérience ---
SELECTED_APPROACH = 'classic'

# --- Paramètres de Prétraitement ---
APPLY_SCALING = True
SCALER_PATH = MODELS_DIR / SELECTED_APPROACH / "scaler.pkl"

HANDLE_IMBALANCE = True
IMBALANCE_METHOD = 'smote'
SMOTE_K_NEIGHBORS = 5
NEAR_MISS_VERSION = 3

# --- Réduction de Dimension (pour l'approche 'classic') ---
REDUCTION_METHOD = 'pca'
PCA_N_COMPONENTS = 0.95
SELECTKBEST_K = 500
SELECTKBEST_SCORE_FUNC = f_classif
DIM_REDUCER_PATH = MODELS_DIR / SELECTED_APPROACH / "dim_reducer.pkl"

# --- Paramètres d'Entraînement ---
SEED = 42
K_FOLDS = 5
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# --- Paramètres Spécifiques aux Modèles ---

# > Approche 1: Classique
CLASSIC_MODEL_TYPE = 'xgb'

XGB_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': NUM_CLASSES,
    'eval_metric': 'mlogloss',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
    'random_state': SEED,
    'use_label_encoder': False,
}

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'random_state': SEED,
    'n_jobs': -1,
    'class_weight': 'balanced' if IMBALANCE_METHOD == 'class_weight' else None
}

# > Approche 3: CNN
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

# --- Paramètres d'Évaluation ---
PRIMARY_METRIC = 'balanced_accuracy'
METRICS_TO_LOG = ['balanced_accuracy', 'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

# --- Sauvegarde des Modèles ---
MODEL_SAVE_PATH = MODELS_DIR / SELECTED_APPROACH / "best_model"

# --- Soumission Kaggle ---
SUBMISSION_FILENAME = f"submission_{SELECTED_APPROACH}_{REDUCTION_METHOD if SELECTED_APPROACH=='classic' else 'cnn'}.csv"
SUBMISSION_FILEPATH = SUBMISSIONS_DIR / SUBMISSION_FILENAME

print("Config loaded.")
print(f"Selected Approach: {SELECTED_APPROACH}")
print(f"Data Directory: {DATA_DIR}")
print(f"Models Directory: {MODELS_DIR}")