
library(tidyverse)
library(mgcv)

# Chargement des données
train <- read_csv('train.csv')
test  <- read_csv('test.csv')

# Transformation des dates en chiffres (nécessaire pour les fonctions de lissage 's')
train$Time <- as.numeric(train$Date)
test$Time  <- as.numeric(test$Date)


# On regarde le temps, la saison (toy), la météo et la demande passée.
gam6 <- gam(Net_demand ~ s(Time, k=3, bs='cr') + 
              s(toy, k=30, bs='cc') + 
              s(Temp, k=10) + 
              s(Net_demand.1) + s(Net_demand.7) +
              s(Temp_s99) + as.factor(WeekDays) + BH +
              s(Wind) + te(Time, Nebulosity, k=c(4,10)), 
            data=train, method="REML")

# On part du principe que l'effet de l'été ou de l'hiver n'est pas le même en 2018 qu'en 2024.
gam7 <- gam(Net_demand ~ te(Time, toy, k=c(4, 20), bs=c('cr', 'cc')) + 
              s(Temp) + s(Net_demand.1) + s(Net_demand.7) +
              s(Temp_s99) + as.factor(WeekDays) + BH +
              s(Wind) + s(Nebulosity), 
            data=train, method="REML")

# Le vent est plus impactant quand il fait déjà froid. On crée une interaction.
gam8 <- gam(Net_demand ~ s(Time, k=3) + s(toy, k=30, bs='cc') + 
              te(Temp, Wind, k=c(8, 8)) + 
              s(Net_demand.1) + s(Net_demand.7) +
              s(Temp_s99) + as.factor(WeekDays) + BH +
              s(Nebulosity), 
            data=train, method="REML")

# On laisse l'ordinateur décider quelles variables servent à quelque chose
gam9 <- gam(Net_demand ~ s(Time, bs='cs') + s(toy, bs='cc') + 
              s(Temp, bs='cs') + s(Net_demand.1, bs='cs') + 
              s(Net_demand.7, bs='cs') + s(Temp_s99, bs='cs') +
              as.factor(WeekDays) + BH + s(Wind, bs='cs') + s(Nebulosity, bs='cs'), 
            data=train, method="REML", select=TRUE)

#calcul AIC
compare_aic <- AIC(gam6, gam7, gam8, gam9)
print(compare_aic)

# Calcul de la déviance
resultats <- data.frame(
  Modele = c("GAM 6 (Base)", "GAM 7 (Saison Évolutive)", "GAM 8 (Vent/Froid)", "GAM 9 (Auto-Sélection)"),
  Score_R2 = c(summary(gam6)$dev.expl, summary(gam7)$dev.expl, 
               summary(gam8)$dev.expl, summary(gam9)$dev.expl) * 100
)

print(resultats)

# -------------------------------------------------------------------------
# 3. Diagnostics et Plots pour le modèle "Gagnant" (par exemple GAM 8)
# -------------------------------------------------------------------------

# A. Vérification de la dimension 'k'
# Regardez la colonne p-value. Si p < 0.05 ET edf est très proche de k', augmentez k.
gam.check(gam8_full) 



# B. Visualisation des Splines (Effets partiels)
# 'pages=1' affiche tout sur la même fenêtre.
# 'scheme=2' affiche les surfaces 2D (pour te()) sous forme de cartes de chaleur très lisibles.
plot(gam8_full, pages=1, scheme=2, shade=TRUE, shade.col="lightblue", main="Effets GAM 8")

pdf("Analyse_GAM6.pdf", width=12, height=8) # Ouvre le fichier PDF
par(mfrow=c(3,4)) # Divise la page en grille (3 lignes, 4 colonnes)
plot(gam6_full, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off() # Ferme et sauvegarde le PDF

pdf("Analyse_GAM7.pdf", width=12, height=8) # Ouvre le fichier PDF
par(mfrow=c(3,4)) # Divise la page en grille (3 lignes, 4 colonnes)
plot(gam7_full, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off() # Ferme et sauvegarde le PDF

pdf("Analyse_GAM8.pdf", width=12, height=8) # Ouvre le fichier PDF
par(mfrow=c(3,4)) # Divise la page en grille (3 lignes, 4 colonnes)
plot(gam8_full, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off() # Ferme et sauvegarde le PDF

pdf("Analyse_GAM9.pdf", width=12, height=8) # Ouvre le fichier PDF
par(mfrow=c(3,4)) # Divise la page en grille (3 lignes, 4 colonnes)
plot(gam9_full, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off() # Ferme et sauvegarde le PDF


# Prédiction sur le TEST set avec le meilleur modèle
best_forecast <- predict(gam8_full, newdata=Data1)

# Préparation du fichier
submit <- submit_base
submit$Net_demand <- best_forecast

# Export
write.table(submit, file="submission_gam_best.csv", quote=FALSE, sep=",", dec='.', row.names=FALSE)








