import lightgbm as lgb
from src import config

def get_lgbm_model(params=None):
    """
    Crée et retourne une instance du modèle LGBMClassifier.
    Utilise les paramètres de config.py par défaut.
    """
    if params is None:
        params = config.LGBM_PARAMS.copy()

    required_params = ['objective', 'metric', 'num_class', 'seed', 'n_jobs']
    for p in required_params:
        if p not in params:
             if hasattr(config, f'LGBM_{p.upper()}'):
                 params[p] = getattr(config, f'LGBM_{p.upper()}')
             elif p == 'num_class':
                 params[p] = config.NUM_CLASSES
             elif p == 'seed':
                 params[p] = config.SEED
             else:
                 print(f"Avertissement: Paramètre LGBM requis '{p}' manquant. Utilisation de valeurs par défaut si possible.")
                 if p == 'objective': params[p] = 'multiclass'
                 if p == 'metric': params[p] = 'multi_logloss'
                 if p == 'n_jobs': params[p] = -1


    print("Création du modèle LGBM avec les paramètres :")
    model = lgb.LGBMClassifier(**params)
    return model

print("Classic model definitions (LGBM) loaded.")