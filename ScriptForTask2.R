################################################################################
## Lista 02 – Métodos de Preprocesamiento de Datos (Cervical Cancer – UCI)
## Script alineado con los códigos de clase (rpart, recipes, SMOTE/ENN)
## Autor: Víctor Mauricio Ochoa García, Salvador Enrique Rodríguez Hernández
## Fecha: Sys.Date()
################################################################################

## ---------------------------------------------------------------------------
## 0) Paquetes y setup (mismo stack de clase)
## ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(readr)     # lectura CSV
  library(dplyr)     # manipulación
  library(ggplot2)   # gráficos
  library(rsample)   # split estratificado
  library(recipes)   # preprocesamiento
  library(themis)    # SMOTE en recipes
  library(FNN)       # ENN (vecinos)
  library(rpart)     # árbol de decisión (como en clase)
})

set.seed(2026)

## ---------------------------------------------------------------------------
## Datos base: UCI Cervical Cancer (trata "?" como NA) + respuesta factor 0/1
## ---------------------------------------------------------------------------
url   <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
cc    <- read_csv(url, na = "?")
datos <- as.data.frame(cc)

target_var <- "Biopsy"
stopifnot(target_var %in% names(datos))

# Asegurar factor binario "0"/"1" (alineado con scripts)
y <- datos[[target_var]]
if (!is.factor(y)) y <- factor(as.character(y))
if (!all(levels(y) %in% c("0","1"))) {
  y <- factor(as.character(y), levels = c("0","1"))
}
datos[[target_var]] <- y

## ---------------------------------------------------------------------------
## Utilidades (idéntico espíritu a clase)
## ---------------------------------------------------------------------------

# Métricas para escenario desbalanceado
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

# ENN manual (como en clase, sobre matriz numérica)
ENN_manual <- function(data, target, k = 3, majority_class) {
  X <- data[, setdiff(names(data), target), drop = FALSE]
  y <- data[[target]]
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

# Entrenar rpart y evaluar por umbral (usamos probabilidades para poder variar corte)
eval_tree <- function(train_df, test_df, cutoff = 0.5, target = target_var) {
  form <- as.formula(paste(target, "~ ."))
  fit  <- rpart(form, data = train_df, method = "class")
  prob <- predict(fit, newdata = test_df, type = "prob")[, "1"]
  pred <- factor(ifelse(prob >= cutoff, "1", "0"), levels = c("0","1"))
  list(metrics = calc_metrics(test_df[[target]], pred),
       cutoff = cutoff,
       n_train = nrow(train_df))
}

## ===========================================================================
## (a) Imputación kNN en todas las predictoras con NA: justificación y evidencia
## ===========================================================================
na_counts <- sapply(datos, function(v) sum(is.na(v)))
na_pct    <- round(100 * na_counts / nrow(datos), 2)
na_tbl    <- data.frame(Variable = names(na_counts), NA_count = na_counts, NA_percent = na_pct) |>
  arrange(desc(NA_percent))
cat("\n[a] Top 12 variables con mayor % de NA\n"); print(head(na_tbl, 12))

## ===========================================================================
## (b) Proporción de clases de la respuesta (barras con porcentajes)
## ===========================================================================
g_base <- datos %>%
  count(!!sym(target_var)) %>%
  mutate(prop = 100 * n / sum(n))
cat("\n[b] Frecuencia/porcentaje por clase:\n"); print(g_base)

ggplot(g_base, aes(x = !!sym(target_var), y = n, fill = !!sym(target_var))) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(prop,1), "%")), vjust = -0.3, size = 4) +
  labs(x = "Clase (Biopsy)", y = "Pacientes",
       title = "Distribución de la respuesta (conteos y porcentajes)") +
  theme_minimal(base_size = 12)

minority_prop <- min(g_base$prop / 100)
cat(sprintf("\n[b] Proporción clase minoritaria (base completa) = %.4f\n", minority_prop))

# Split estratificado 70/30 (como en clase)
set.seed(2026)
split_obj <- initial_split(datos, prop = 0.70, strata = all_of(target_var))
treino0 <- training(split_obj)
teste0  <- testing(split_obj)

## ===========================================================================
## Preparación común (alineado al patrón de clase):
##  - padronización (step_normalize)
##  - imputación kNN (step_impute_knn, k=5)
## Nota: clase a veces usa normalize -> impute -> (opcional SMOTE)
## ===========================================================================
rec_base <- recipe(as.formula(paste(target_var, "~ .")), data = treino0) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_numeric_predictors(), neighbors = 5)

prep_base   <- prep(rec_base)
treino_proc <- bake(prep_base, new_data = NULL)
teste_proc  <- bake(prep_base, new_data = teste0)

## ===========================================================================
## (c) SMOTE + Padronización/Imputación + Árbol (corte = 0.5)
##     (misma lógica que "SMOTE + PADRONIZACIÓN" en clase)
## ===========================================================================
rec_smote <- recipe(as.formula(paste(target_var, "~ .")), data = treino0) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_numeric_predictors(), neighbors = 5) %>%
  step_smote(all_outcomes())               # SMOTE solo en entrenamiento

prep_smote   <- prep(rec_smote)
treino_smote <- bake(prep_smote, new_data = NULL)
teste_smote  <- bake(prep_smote, new_data = teste0)   # sin SMOTE (solo norm + impute)

res_c <- eval_tree(treino_smote, teste_smote, cutoff = 0.5)
row_c <- cbind(Modelo = "SMOTE (cut=0.5)", Ntrain = res_c$n_train, Cutoff = res_c$cutoff, res_c$metrics)
cat("\n[c] Resultados: SMOTE + Árbol (cut=0.5)\n"); print(row_c)

## ===========================================================================
## (d) ENN + Padronización/Imputación + Árbol (corte = prop. minoritaria)
##     (como "ENN + PADRONIZACIÓN" en clase; ENN tras normalizar/imputar)
## ===========================================================================
maj_class <- names(sort(table(treino_proc[[target_var]]), decreasing = TRUE))[1]
treino_ENN <- ENN_manual(treino_proc, target = target_var, k = 3, majority_class = maj_class)

prop_ENN <- prop.table(table(treino_ENN[[target_var]]))
cat("\n[d] Proporciones tras ENN (train):\n"); print(prop_ENN)

cut_ENN <- as.numeric(min(prop.table(table(treino_proc[[target_var]]))))  # proporción minoritaria original (train proc)
res_d   <- eval_tree(treino_ENN, teste_proc, cutoff = cut_ENN)
row_d   <- cbind(Modelo = sprintf("ENN (cut=%.3f)", cut_ENN), Ntrain = res_d$n_train, Cutoff = res_d$cutoff, res_d$metrics)
cat("\n[d] Resultados: ENN + Árbol (cut=prop.min)\n"); print(row_d)

## ===========================================================================
## (e) SMOTE + ENN + Padronización/Imputación + Árbol (corte = 0.5)
##     (clase: normalizar/imputar -> ENN -> SMOTE sobre train)
## ===========================================================================
# 1) base tratada (norm + impute)
treino_proc2 <- treino_proc
teste_proc2  <- teste_proc

# 2) ENN en train tratado
treino_ENN2 <- ENN_manual(treino_proc2, target = target_var, k = 3, majority_class = maj_class)

# 3) SMOTE sobre el train post-ENN
rec_smote2 <- recipe(as.formula(paste(target_var, "~ .")), data = treino_ENN2) %>%
  step_smote(all_outcomes())
prep_smote2      <- prep(rec_smote2)
treino_ENN_SMOTE <- bake(prep_smote2, new_data = NULL)

# 4) Evaluar árbol
res_e <- eval_tree(treino_ENN_SMOTE, teste_proc2, cutoff = 0.5)
row_e <- cbind(Modelo = "SMOTE+ENN (cut=0.5)", Ntrain = res_e$n_train, Cutoff = res_e$cutoff, res_e$metrics)
cat("\n[e] Resultados: SMOTE+ENN + Árbol (cut=0.5)\n"); print(row_e)

## ===========================================================================
## (f) Base desbalanceada + Padronización/Imputación + Árbol (dos cortes)
##     (idéntico espíritu a clase: sin SMOTE/ENN, dos umbrales)
## ===========================================================================
# Sin técnicas de balanceo: usar treino_proc/teste_proc
res_f1 <- eval_tree(treino_proc, teste_proc, cutoff = 0.5)
row_f1 <- cbind(Modelo = "Desbalanceada (cut=0.5)", Ntrain = res_f1$n_train, Cutoff = res_f1$cutoff, res_f1$metrics)
cat("\n[f1] Resultados: Desbalanceada + Árbol (cut=0.5)\n"); print(row_f1)

# corte = proporción minoritaria de la base completa (como guía)
res_f2 <- eval_tree(treino_proc, teste_proc, cutoff = minority_prop)
row_f2 <- cbind(Modelo = sprintf("Desbalanceada (cut=%.3f)", minority_prop),
                Ntrain = res_f2$n_train, Cutoff = res_f2$cutoff, res_f2$metrics)
cat("\n[f2] Resultados: Desbalanceada + Árbol (cut=prop.min)\n"); print(row_f2)

## ===========================================================================
## (g) Tabla comparativa final y “mejor” por F1 (desempate MCC y G-mean)
## ===========================================================================
tabla_final <- dplyr::bind_rows(row_c, row_d, row_e, row_f1, row_f2)
print(tabla_final)

# Elegir “mejor” por F1, luego MCC y G-mean (como guía de discusión)
pick <- tabla_final %>%
  mutate(across(c(F1, MCC, Gmean), as.numeric)) %>%
  arrange(desc(F1), desc(MCC), desc(Gmean)) %>%
  slice(1)

best_model <- pick$Modelo
best_f1    <- as.numeric(pick$F1)
best_mcc   <- as.numeric(pick$MCC)
best_gmean <- as.numeric(pick$Gmean)

cat("\n------------------\n[g] Conclusión principal\n------------------\n")
cat(sprintf("Mejor estrategia: %s\n", best_model))
cat(sprintf("F1 = %.3f | MCC = %.3f | G-mean = %.3f\n", best_f1, best_mcc, best_gmean))
cat("Notas:\n - En desbalance, F1, G-mean y MCC son más informativas que Accuracy.\n",
    " - El punto de corte afecta sensibilidad/especificidad; tras SMOTE, 0.5 suele ser razonable.\n",
    " - ENN ayuda si la mayoría tiene ruido; calibrar k y orden ENN/SMOTE según datos.\n", sep = "")
