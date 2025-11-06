#################################################################################
## Lista 02 – Métodos de Preprocesamiento de Datos (Cervical Cancer – UCI)    ##
## Autor: (tu nombre)                                                          ##
## Fecha: (hoy)                                                                ##
#################################################################################

set.seed(2026)

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(rsample)
  library(recipes)
  library(themis)   # SMOTE
  library(FNN)      # ENN (vecinos)
})

#==============================#
# Utilidades (funciones base)  #
#==============================#

## (i) Métricas (mismo estilo de tus scripts)
calc_metrics <- function(y_true, y_pred) {
  tab <- table(Pred = y_pred, Real = y_true)
  TP <- ifelse("1" %in% rownames(tab) & "1" %in% colnames(tab), tab["1","1"], 0)
  TN <- ifelse("0" %in% rownames(tab) & "0" %in% colnames(tab), tab["0","0"], 0)
  FP <- ifelse("1" %in% rownames(tab) & "0" %in% colnames(tab), tab["1","0"], 0)
  FN <- ifelse("0" %in% rownames(tab) & "1" %in% colnames(tab), tab["0","1"], 0)
  
  acc   <- ifelse((TP + TN + FP + FN) > 0, (TP + TN)/(TP + TN + FP + FN), NA)
  sens  <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
  esp   <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  ppv   <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  npv   <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)
  gmean <- ifelse(!is.na(sens) & !is.na(esp), sqrt(sens * esp), NA)
  f1    <- ifelse((ppv + sens) > 0, 2 * ppv * sens / (ppv + sens), NA)
  
  denom <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
  mcc   <- ifelse(denom > 0, (TP*TN - FP*FN) / denom, NA)
  
  data.frame(Accuracy = acc, Sensitivity = sens, Specificity = esp,
             PPV = ppv, NPV = npv, Gmean = gmean, F1 = f1, MCC = mcc)
}

## (ii) ENN (edición por vecinos – limpia mayoritaria que discrepa de sus vecinos)
ENN_manual <- function(data, target, k = 3, majority_class) {
  X <- data[, setdiff(names(data), target), drop = FALSE]
  y <- data[[target]]
  # knn requiere numéricos – convertimos factores a codificación numérica simple
  X_num <- as.data.frame(lapply(X, function(z) if(is.numeric(z)) z else as.numeric(as.factor(z))))
  nn <- knnx.index(as.matrix(X_num), as.matrix(X_num), k = k + 1)[, -1]
  remove_idx <- sapply(1:nrow(X_num), function(i) {
    if (y[i] == majority_class) {
      neigh_classes <- y[nn[i, ]]
      maj_class <- names(sort(table(neigh_classes), decreasing = TRUE))[1]
      return(maj_class != y[i])
    } else FALSE
  })
  data[!remove_idx, , drop = FALSE]
}

#===========================================================#
# (a) Imputación por kNN en todas las predictoras con NA    #
#     (y por qué: alta proporción de NA en varias columnas) #
#===========================================================#

# Cargar datos desde UCI (tratar "?" como NA)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
cc  <- read_csv(url, na = "?")
datos <- as.data.frame(cc)

# Variable respuesta
target_var <- "Biopsy"
stopifnot(target_var %in% names(datos))

# Asegurar factor binario "0","1"
y <- datos[[target_var]]
if (!is.factor(y)) y <- factor(as.character(y))
if (!all(levels(y) %in% c("0","1"))) {
  y <- factor(as.character(y), levels = c("0","1"))
}
datos[[target_var]] <- y

# Auditoría de NA para justificar imputación
na_counts <- sapply(datos, function(v) sum(is.na(v)))
na_pct    <- round(100 * na_counts / nrow(datos), 2)
na_tbl    <- data.frame(var = names(na_counts), NA_count = na_counts, NA_percent = na_pct) |>
  arrange(desc(NA_percent))
cat("\n[NA TOP 12]\n"); print(head(na_tbl, 12))

# (Opcional) Eliminar variables con >90% NA – deja la línea activada si lo deseas
datos <- datos[, colMeans(is.na(datos)) < 0.9]

# Split estratificado 70/30 (se reutiliza en todo)
set.seed(2026)
split_obj <- initial_split(datos, prop = 0.70, strata = all_of(target_var))
treino0 <- training(split_obj)
teste0  <- testing(split_obj)

# Recipe: normalizar + imputar por kNN (sobre entrenamiento) y aplicar a ambos
rec_imp <- recipe(as.formula(paste(target_var, "~ .")), data = treino0) %>%
  step_impute_knn(all_numeric_predictors(), neighbors = 5) %>%  # 1) imputar
  step_normalize(all_numeric_predictors())                      # 2) normalizar

imp_prep   <- prep(rec_imp)
treino_imp <- bake(imp_prep, new_data = NULL)
teste_imp  <- bake(imp_prep, new_data = teste0)

cat("\n[Imputación kNN aplicada: entrenamiento/test listos]\n")

#===========================================================#
# (b) Describir proporción de clases (barras + porcentajes) #
#===========================================================#

g_base <- datos %>%
  count(!!sym(target_var)) %>%
  mutate(prop = 100 * n / sum(n))
print(g_base)

ggplot(g_base, aes(x = !!sym(target_var), y = n, fill = !!sym(target_var))) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(prop,1), "%")), vjust = -0.3) +
  labs(x = "Clase (Biopsy)", y = "Frecuencia absoluta",
       title = "Distribución de la respuesta (conteos y porcentajes)") +
  theme_minimal(base_size = 12)

minority_prop <- min(g_base$prop/100)
cat(sprintf("\n[Proporción minoritaria (base original) = %.4f]\n", minority_prop))

#===========================================================#
# Helper común: ajuste y evaluación de Regresión Logística  #
#===========================================================#

eval_logit <- function(train_df, test_df, cutoff = 0.5, target = target_var) {
  form <- as.formula(paste(target, "~ ."))
  mod  <- glm(form, data = train_df, family = binomial)
  prob <- predict(mod, newdata = test_df, type = "response")
  pred <- ifelse(prob >= cutoff, "1", "0")
  pred <- factor(pred, levels = c("0","1"))
  list(metrics = calc_metrics(test_df[[target]], pred),
       cutoff = cutoff,
       n_train = nrow(train_df))
}

results <- list()  # iremos apilando filas en orden (c)–(f)

#===========================================================#
# (c) SMOTE + Regresión logística (punto de corte 0.5)      #
#     → 1ª línea de la tabla final                           #
#===========================================================#

rec_smote <- recipe(as.formula(paste(target_var, "~ .")), data = treino_imp) %>%
  step_smote(all_outcomes())
smote_prep  <- prep(rec_smote)
treino_smote <- bake(smote_prep, new_data = NULL)

res_c <- eval_logit(treino_smote, teste_imp, cutoff = 0.5)
row_c <- cbind(Modelo = "SMOTE (cut=0.5)", Ntrain = res_c$n_train, Cutoff = res_c$cutoff, res_c$metrics)

#===========================================================#
# (d) ENN + Regresión logística                             #
#     (punto de corte = proporción de la clase minoritaria)  #
#     → 2ª línea de la tabla                                 #
#===========================================================#

maj_class  <- names(sort(table(treino_imp[[target_var]]), decreasing = TRUE))[1]
treino_ENN <- ENN_manual(treino_imp, target = target_var, k = 3, majority_class = maj_class)

# Nueva proporción en entrenamiento tras ENN (para reportar)
prop_ENN <- round(prop.table(table(treino_ENN[[target_var]])), 4)
cat("\n[Proporciones tras ENN (train)]\n"); print(prop_ENN)

cut_ENN <- as.numeric(min(prop.table(table(treino_imp[[target_var]]))))  # pide usar la proporción minoritaria
res_d <- eval_logit(treino_ENN, teste_imp, cutoff = cut_ENN)
row_d <- cbind(Modelo = sprintf("ENN (cut=%.3f)", cut_ENN), Ntrain = res_d$n_train, Cutoff = res_d$cutoff, res_d$metrics)

#===========================================================#
# (e) SMOTE + ENN + Regresión logística (corte 0.5)         #
#     → 3ª línea de la tabla                                 #
#===========================================================#

treino_ENN2 <- ENN_manual(treino_imp, target = target_var, k = 3, majority_class = maj_class)
rec_smote2  <- recipe(as.formula(paste(target_var, "~ .")), data = treino_ENN2) %>%
  step_smote(all_outcomes())
smote2_prep      <- prep(rec_smote2)
treino_ENN_SMOTE <- bake(smote2_prep, new_data = NULL)

res_e <- eval_logit(treino_ENN_SMOTE, teste_imp, cutoff = 0.5)
row_e <- cbind(Modelo = "SMOTE+ENN (cut=0.5)", Ntrain = res_e$n_train, Cutoff = res_e$cutoff, res_e$metrics)

#===========================================================#
# (f) Base desbalanceada + Reg Log con 2 cortes             #
#     → 4ª línea (cut=0.5) y 5ª línea (cut=prop.min)        #
#===========================================================#

# Sin SMOTE/ENN: solo imputación/normalización
# (i) corte 0.5
res_f1 <- eval_logit(treino_imp, teste_imp, cutoff = 0.5)
row_f1 <- cbind(Modelo = "Desbalanceada (cut=0.5)", Ntrain = res_f1$n_train, Cutoff = res_f1$cutoff, res_f1$metrics)

# (ii) corte = proporción de la clase minoritaria (de la base original)
cut_min <- minority_prop
res_f2 <- eval_logit(treino_imp, teste_imp, cutoff = cut_min)
row_f2 <- cbind(Modelo = sprintf("Desbalanceada (cut=%.3f)", cut_min), Ntrain = res_f2$n_train, Cutoff = res_f2$cutoff, res_f2$metrics)

#===========================================================#
# Tabla final (orden exacto del enunciado)                  #
#===========================================================#

tabla_final <- dplyr::bind_rows(row_c, row_d, row_e, row_f1, row_f2)
tabla_final <- tibble::as_tibble(tabla_final)
print(tabla_final)

#===========================================================#
# (g) Guía para la discusión (para tu informe)              #
#===========================================================#
# - En desbalance, prioriza F1, G-mean y MCC (son más robustas que Accuracy).
# - Compara las 5 filas de `tabla_final` y elige la mejor por esas tres métricas.
# - Explica cómo cambia el rendimiento al mover el punto de corte (0.5 vs. prop. minoritaria).
# - Justifica la metodología final (p. ej., SMOTE o SMOTE+ENN) según el mejor trade-off.
