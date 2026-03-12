
#' Fit and compare multiple GAM (Generalized Additive Model) specifications.
#'
#' This script trains four different GAM models on electricity demand data
#' and compares their performance using AIC and explained deviance:
#' - GAM6: Base model with smooth terms for main features.
#' - GAM7: Time-varying seasonality (tensor product interaction).
#' - GAM8: Temperature-wind interaction to capture trade-off effects.
#' - GAM9: Auto-select important features with shrinkage.
#'
#' Outputs:
#' - Four PDF files with diagnostic plots for each model.
#' - Comparison table of model performance metrics.
#' - Predictions saved to submission CSV file.
#'
#' Inspiration:
#' - R mgcv package for smooth spline fitting.
#' - Existing project data preparation.
#' - Teacher's code for GAM

library(tidyverse)
library(mgcv)

# Load training and test datasets.
train <- read_csv('train.csv')
test  <- read_csv('test.csv')

# Convert dates to numeric values (needed for smooth functions).
train$Time <- as.numeric(train$Date)
test$Time  <- as.numeric(test$Date)

# GAM 6: Base model with smooth terms for time, season, temperature, and lagged demand.
gam6 <- gam(Net_demand ~ s(Time, k=3, bs='cr') + 
              s(toy, k=30, bs='cc') + 
              s(Temp, k=10) + 
              s(Net_demand.1) + s(Net_demand.7) +
              s(Temp_s99) + as.factor(WeekDays) + BH +
              s(Wind) + te(Time, Nebulosity, k=c(4,10)), 
            data=train, method="REML")

# GAM 7: Allow seasonality to change over time (tensor product).
gam7 <- gam(Net_demand ~ te(Time, toy, k=c(4, 20), bs=c('cr', 'cc')) + 
              s(Temp) + s(Net_demand.1) + s(Net_demand.7) +
              s(Temp_s99) + as.factor(WeekDays) + BH +
              s(Wind) + s(Nebulosity), 
            data=train, method="REML")

# GAM 8: Temperature-wind interaction (wind impact depends on temperature).
gam8 <- gam(Net_demand ~ s(Time, k=3) + s(toy, k=30, bs='cc') + 
              te(Temp, Wind, k=c(8, 8)) + 
              s(Net_demand.1) + s(Net_demand.7) +
              s(Temp_s99) + as.factor(WeekDays) + BH +
              s(Nebulosity), 
            data=train, method="REML")

# GAM 9: Let the model auto-select which features matter (with shrinkage).
gam9 <- gam(Net_demand ~ s(Time, bs='cs') + s(toy, bs='cc') + 
              s(Temp, bs='cs') + s(Net_demand.1, bs='cs') + 
              s(Net_demand.7, bs='cs') + s(Temp_s99, bs='cs') +
              as.factor(WeekDays) + BH + s(Wind, bs='cs') + s(Nebulosity, bs='cs'), 
            data=train, method="REML", select=TRUE)

# AIC comparison.
compare_aic <- AIC(gam6, gam7, gam8, gam9)
print(compare_aic)

# Explained deviance (goodness of fit).
resultats <- data.frame(
  Modele = c("GAM 6 (Base)", "GAM 7 (Saison Évolutive)", "GAM 8 (Vent/Froid)", "GAM 9 (Auto-Sélection)"),
  Score_R2 = c(summary(gam6)$dev.expl, summary(gam7)$dev.expl, 
               summary(gam8)$dev.expl, summary(gam9)$dev.expl) * 100
)

print(resultats)

# -------------------------------------------------------------------------
# 3. Diagnostics et Plots pour le modèle "Gagnant" (par exemple GAM 8)
# -------------------------------------------------------------------------

# Check if the smoothing dimension k is adequate.
# If p-value < 0.05 AND edf is close to k, increase k for that term.
gam.check(gam8) 



# Plot smooth effects (partial effects).
# 'scheme=2' shows 2D tensor products as heatmaps.
plot(gam8, pages=1, scheme=2, shade=TRUE, shade.col="lightblue", main="GAM 8 Effects")

# Save diagnostic plots to PDF.
pdf("Analyse_GAM6.pdf", width=12, height=8)
par(mfrow=c(3,4))
plot(gam6, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off()

pdf("Analyse_GAM7.pdf", width=12, height=8)
par(mfrow=c(3,4))
plot(gam7, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off()

pdf("Analyse_GAM8.pdf", width=12, height=8)
par(mfrow=c(3,4))
plot(gam8, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off()

pdf("Analyse_GAM9.pdf", width=12, height=8)
par(mfrow=c(3,4))
plot(gam9, scheme=2, shade=TRUE, shade.col="lightblue")
dev.off()


# Predict on test set with best model.
best_forecast <- predict(gam8, newdata=test)

# Prepare and save submission file.
submit <- test
submit$Net_demand <- best_forecast

write.table(submit, file="submission_gam_best.csv", quote=FALSE, sep=",", dec='.', row.names=FALSE)








