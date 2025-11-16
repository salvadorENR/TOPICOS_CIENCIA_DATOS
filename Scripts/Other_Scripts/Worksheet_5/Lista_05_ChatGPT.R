# Lista05_NN_heart_with_capture.R
# Solución completa (a)–(f) — Lista 05: Redes Neuronales
# Autor: (poner su nombre)
# Fecha: (modificar según convenga)

# ---------------------------------------------------------------------
# INICIO: abrir captura de salida a archivo RN_Results_CGPT.txt
# ---------------------------------------------------------------------
output_file <- "RN_Results_CGPT.txt"
sink(file = output_file, split = TRUE)
cat("================================================================================\n")
cat("RESULTADOS DEL ANÁLISIS — Lista 05: Redes Neuronales\n")
cat("Generado:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("================================================================================\n\n")

# -----------------------------
# Paquetes necesarios
# -----------------------------
req_pkgs <- c("kmed", "dplyr", "rsample", "caret", "nnet", "recipes", "e1071")
for (p in req_pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)

library(kmed)
library(dplyr)
library(rsample)
library(caret)
library(nnet)
library(recipes)
library(e1071)

set.seed(125)

# -----------------------------
# (a) Cargar datos, descripción y binarización de Y
# -----------------------------
data("heart")
heart <- as.data.frame(heart)

cat("\n(a) Estructura del dataset 'heart':\n")
str(heart)
cat("\n(a) Resumen estadístico inicial:\n")
print(summary(heart))

cat("\n(a) Cuadro de frecuencia de 'class' (original):\n")
cuadro_class_abs <- table(heart$class)
cuadro_class_prop <- prop.table(cuadro_class_abs)
print(data.frame(Clase = names(cuadro_class_abs),
                 Frecuencia = as.vector(cuadro_class_abs),
                 Proporcion = round(as.vector(cuadro_class_prop), 4)))

heart <- heart %>%
  mutate(Y_bin = ifelse(class == 0, 0, 1))

heart$Y_bin <- factor(heart$Y_bin, levels = c(0,1), labels = c("neg","pos"))

cat("\nDistribución de Y_bin (absoluta / proporción):\n")
print(table(heart$Y_bin))
print(round(prop.table(table(heart$Y_bin)), 4))

# -----------------------------
# Partición estratificada 70/30 (misma partición para todo)
# -----------------------------
split_obj <- initial_split(heart, prop = 0.7, strata = "Y_bin")
train <- training(split_obj)
test  <- testing(split_obj)

cat("\nTamaños Train / Test:\n")
cat("Train:", nrow(train), " Test:", nrow(test), "\n")
cat("Proporciones en Train:\n"); print(prop.table(table(train$Y_bin)))
cat("Proporciones en Test:\n");  print(prop.table(table(test$Y_bin)))

# -----------------------------
# Funciones de utilidad
# -----------------------------
make_recipe <- function(data, predictors, outcome) {
  recipe_obj <- recipe(as.formula(paste(outcome, "~", paste(predictors, collapse = " + "))), data = data) %>%
    step_dummy(all_nominal_predictors(), -all_outcomes(), one_hot = FALSE) %>%
    step_center(all_numeric_predictors()) %>%
    step_scale(all_numeric_predictors())
  return(recipe_obj)
}

calc_binary_metrics <- function(y_true, y_pred) {
  cm <- table(Pred = as.character(y_pred), True = as.character(y_true))
  TP <- ifelse("pos" %in% rownames(cm) & "pos" %in% colnames(cm), cm["pos","pos"], 0)
  TN <- ifelse("neg" %in% rownames(cm) & "neg" %in% colnames(cm), cm["neg","neg"], 0)
  FP <- ifelse("pos" %in% rownames(cm) & "neg" %in% colnames(cm), cm["pos","neg"], 0)
  FN <- ifelse("neg" %in% rownames(cm) & "pos" %in% colnames(cm), cm["neg","pos"], 0)
  total <- TP+TN+FP+FN
  Accuracy <- ifelse(total>0, (TP+TN)/total, NA)
  Sensitivity <- ifelse((TP+FN)>0, TP/(TP+FN), NA)
  Specificity <- ifelse((TN+FP)>0, TN/(TN+FP), NA)
  PPV <- ifelse((TP+FP)>0, TP/(TP+FP), NA)
  NPV <- ifelse((TN+FN)>0, TN/(TN+FN), NA)
  Gmean <- ifelse(!is.na(Sensitivity) & !is.na(Specificity), sqrt(Sensitivity*Specificity), NA)
  F1 <- ifelse(!is.na(PPV) & !is.na(Sensitivity) & (PPV+Sensitivity)>0, 2*PPV*Sensitivity/(PPV+Sensitivity), NA)
  denom <- sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  MCC <- ifelse(denom>0, (TP*TN - FP*FN)/denom, NA)
  return(data.frame(Accuracy = Accuracy, Sensitivity = Sensitivity, Specificity = Specificity,
                    PPV = PPV, NPV = NPV, Gmean = Gmean, F1 = F1, MCC = MCC))
}

calc_multiclass_metrics <- function(y_true, y_pred) {
  cm_obj <- caret::confusionMatrix(y_pred, y_true)
  overall_acc <- cm_obj$overall["Accuracy"]
  overall_kappa <- cm_obj$overall["Kappa"]
  byClass <- as.data.frame(cm_obj$byClass)
  recall_per_class <- byClass[,"Sensitivity"]
  precision_per_class <- byClass[,"Pos Pred Value"]
  f1_per_class <- ifelse(!is.na(precision_per_class) & !is.na(recall_per_class) & (precision_per_class+recall_per_class)>0,
                         2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class),
                         NA)
  macro_recall <- mean(recall_per_class, na.rm = TRUE)
  macro_precision <- mean(precision_per_class, na.rm = TRUE)
  macro_f1 <- mean(f1_per_class, na.rm = TRUE)
  return(list(Accuracy = as.numeric(overall_acc),
              Kappa = as.numeric(overall_kappa),
              Macro_Recall = macro_recall,
              Macro_Precision = macro_precision,
              Macro_F1 = macro_f1,
              ByClass = byClass))
}

# -----------------------------
# Selección de predictores: usar todas las variables (menos 'class' que será usada en (f))
# -----------------------------
predictors_all <- setdiff(names(heart), c("class", "Y_bin"))

# -----------------------------
# Preprocesamiento: ajustar receta en TRAIN y aplicar a TRAIN/TEST
# -----------------------------
rec <- make_recipe(train, predictors_all, "Y_bin")
rec_prep <- prep(rec, training = train)
train_proc <- bake(rec_prep, new_data = train)
test_proc  <- bake(rec_prep, new_data = test)

xnames <- setdiff(names(train_proc), "Y_bin")
X_train <- train_proc[, xnames, drop = FALSE]
X_test  <- test_proc[, xnames, drop = FALSE]
y_train <- train_proc$Y_bin
y_test  <- test_proc$Y_bin

y_train <- factor(y_train, levels = c("neg","pos"))
y_test  <- factor(y_test,  levels = c("neg","pos"))

# -----------------------------
# (b) Red neuronal con size = 3
# -----------------------------
set.seed(2026)
nn3 <- nnet::nnet(Y_bin ~ ., data = train_proc, size = 3, decay = 0.01, maxit = 500, trace = FALSE)
pred_nn3_class <- predict(nn3, newdata = test_proc, type = "class")
metrics_nn3 <- calc_binary_metrics(y_test, pred_nn3_class)
cat("\n(b) Red neuronal (size = 3) - métricas en TEST:\n")
print(round(metrics_nn3, 4))

# -----------------------------
# (c) Red neuronal con size = 5
# -----------------------------
set.seed(2026)
nn5 <- nnet::nnet(Y_bin ~ ., data = train_proc, size = 5, decay = 0.01, maxit = 500, trace = FALSE)
pred_nn5_class <- predict(nn5, newdata = test_proc, type = "class")
metrics_nn5 <- calc_binary_metrics(y_test, pred_nn5_class)
cat("\n(c) Red neuronal (size = 5) - métricas en TEST:\n")
print(round(metrics_nn5, 4))

# -----------------------------
# (d) Red neuronal con size = 10
# -----------------------------
set.seed(2026)
nn10 <- nnet::nnet(Y_bin ~ ., data = train_proc, size = 10, decay = 0.01, maxit = 500, trace = FALSE)
pred_nn10_class <- predict(nn10, newdata = test_proc, type = "class")
metrics_nn10 <- calc_binary_metrics(y_test, pred_nn10_class)
cat("\n(d) Red neuronal (size = 10) - métricas en TEST:\n")
print(round(metrics_nn10, 4))

# -----------------------------
# (e) Comparación de las redes ajustadas
# -----------------------------
results_bin <- rbind(
  NN_size_3  = metrics_nn3,
  NN_size_5  = metrics_nn5,
  NN_size_10 = metrics_nn10
)
cat("\n(e) Tabla comparativa de redes (métricas resumen):\n")
print(round(results_bin, 4))

# -----------------------------
# (f) Clasificación multiclase usando 'class' original con nnet (size = 5)
# -----------------------------
train_multi <- train %>% mutate(class_factor = factor(class))
test_multi  <- test  %>% mutate(class_factor = factor(class))

rec_multi <- recipe(class_factor ~ ., data = train_multi) %>%
  update_role(Y_bin, new_role = "ID") %>%
  step_rm(all_of(c("Y_bin"))) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors())

rec_multi_prep <- prep(rec_multi, training = train_multi)
train_multi_proc <- bake(rec_multi_prep, new_data = train_multi)
test_multi_proc  <- bake(rec_multi_prep, new_data = test_multi)

set.seed(2026)
nn_multi5 <- nnet::nnet(class_factor ~ ., data = train_multi_proc, size = 5, decay = 0.01, maxit = 500, trace = FALSE)

pred_multi_class <- predict(nn_multi5, newdata = test_multi_proc, type = "class")
pred_multi_prob  <- predict(nn_multi5, newdata = test_multi_proc, type = "raw")

multi_metrics <- calc_multiclass_metrics(y_true = test_multi_proc$class_factor, y_pred = factor(pred_multi_class, levels = levels(test_multi_proc$class_factor)))

cat("\n(f) Métricas multiclase (nnet size = 5):\n")
cat("Accuracy:", round(multi_metrics$Accuracy, 4), "\n")
cat("Kappa  :", round(multi_metrics$Kappa, 4), "\n")
cat("Macro-Recall   :", round(multi_metrics$Macro_Recall, 4), "\n")
cat("Macro-Precision:", round(multi_metrics$Macro_Precision, 4), "\n")
cat("Macro-F1       :", round(multi_metrics$Macro_F1, 4), "\n")

cat("\n(f) Métricas por clase (ByClass):\n")
print(round(multi_metrics$ByClass, 4))

# -----------------------------
# Guardar objetos y tablas de resumen
# -----------------------------
save.image(file = "Lista05_NN_results.RData")

summary_bin_df <- data.frame(Model = rownames(results_bin), results_bin, row.names = NULL)
write.csv(summary_bin_df, "Lista05_NN_binary_results.csv", row.names = FALSE)

byclass_df <- multi_metrics$ByClass
write.csv(byclass_df, "Lista05_NN_multiclass_byclass.csv", row.names = TRUE)

cat("\nSe guardaron los siguientes archivos:\n")
cat("- Lista05_NN_results.RData\n")
cat("- Lista05_NN_binary_results.csv\n")
cat("- Lista05_NN_multiclass_byclass.csv\n")

# ---------------------------------------------------------------------
# FIN: cerrar captura y mostrar ruta del archivo
# ---------------------------------------------------------------------
cat("\n================================================================================\n")
cat("FIN DEL ANÁLISIS - Lista 05: Redes Neuronales\n")
cat("================================================================================\n\n")
sink()   # cierra la captura

cat("\n✓ Todos los resultados guardados en:", output_file, "\n")
cat("Ubicación completa del archivo:", normalizePath(output_file), "\n")
