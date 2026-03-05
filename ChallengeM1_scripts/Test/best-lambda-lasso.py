import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from Linear import LinearModel

# ---------------------------------------------------------
# 1. Préparation de la Grille de recherche
# ---------------------------------------------------------
# On teste des valeurs logarithmiques (très petit à très grand)
lambda_values = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Nombre de splits temporels (ex: 5 coupures chronologiques)
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Pour stocker les scores moyens
mean_scores = []

print(f"Début de la Cross-Validation Temporelle sur {n_splits} splits...")
print("-" * 60)

# ---------------------------------------------------------
# 2. Boucle de Cross-Validation
# ---------------------------------------------------------
for lam in lambda_values:
    fold_scores = []
    
    # On itère sur les plis temporels
    for train_index, val_index in tscv.split(X):
        # A. Découpage temporel respecté
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # B. Standardisation (CRUCIAL pour Lasso)
        # On fit le scaler UNIQUEMENT sur le train pour ne pas tricher avec le futur
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold) # On applique la même transformation
        
        # C. Entraînement du modèle
        # On utilise votre classe LinearModel avec penalty='l1' (Lasso)
        model = LinearModel(
            learning_rate=0.01, 
            maxIter=2000, 
            tau=0.8,           # La cible demandée (Quantile 0.8)
            lambda_reg=lam,    # Le lambda qu'on teste
            penalty='l1'       # Lasso
        )
        
        # Pas besoin de verbose ici pour ne pas polluer l'affichage
        model.fit(X_train_scaled, y_train_fold, loss="pinball", verbose=False)
        
        # D. Prédiction et Evaluation sur le fold de validation
        y_pred_val = model.predict(X_val_scaled)
        
        # Calcul de la Pinball Loss (Quantile 0.8) pour ce fold
        # Formule : mean(max(0.8 * error, (0.8-1) * error))
        # Attention au sens : ici error = y_true - y_pred (ou l'inverse, tant qu'on est cohérent avec la formule)
        # Formule sklearn : pinball(y_true, y_pred)
        diff = y_val_fold - y_pred_val
        score_fold = np.mean(np.maximum(0.8 * diff, (0.8 - 1.0) * diff))
        
        fold_scores.append(score_fold)
    
    # Moyenne des scores pour ce lambda
    avg_score = np.mean(fold_scores)
    mean_scores.append(avg_score)
    print(f"Lambda = {lam:7.4f} | Pinball Loss Moyenne = {avg_score:.4f}")

# ---------------------------------------------------------
# 3. Sélection du meilleur Lambda
# ---------------------------------------------------------
best_idx = np.argmin(mean_scores)
best_lambda = lambda_values[best_idx]
best_score = mean_scores[best_idx]

print("-" * 60)
print(f"MEILLEUR RÉSULTAT : Lambda = {best_lambda} (Loss = {best_score:.4f})")

# ---------------------------------------------------------
# 4. Entraînement Final (Re-training)
# ---------------------------------------------------------
print("\nRe-entraînement du modèle final sur TOUTES les données avec le meilleur lambda...")

# --- 4. Plot du log(lambda) vs Error ---
plt.figure(figsize=(10, 6))
plt.errorbar(np.log10(lambdas), mean_scores, yerr=std_scores, fmt='-o', capsize=5)
plt.title('Validation Croisée : Impact de la régularisation Lasso')
plt.xlabel('log10(lambda_reg)')
plt.ylabel('Pinball Loss (Validation)')
plt.axvline(np.log10(best_lambda), color='r', linestyle='--', label=f'Best lambda: {best_lambda:.4f}')
plt.legend()
plt.grid(True)
plt.show()

# On re-scale sur la totalité des données cette fois
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X)

final_model = LinearModel(
    learning_rate=0.01, 
    maxIter=5000,      # On peut augmenter un peu pour le final
    tau=0.8, 
    lambda_reg=best_lambda, 
    penalty='l1'
)

final_model.fit(X_final_scaled, y, loss="pinball", verbose=True)

print("Modèle prêt pour les prédictions sur le fichier de soumission !")