# Archivo de salida para capturar resultados
output_file <- "SVM_Analysis_Results.txt"
sink(file = output_file, split = TRUE)

# Cargar paquetes necesarios (instalar si falta)
req_pkgs <- c("kmed", "dplyr", "rsample", "caret", "mRMRe", "e1071")
for (p in req_pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(kmed)
library(dplyr)
library(rsample)
library(caret)
library(mRMRe)
library(e1071)

set.seed(125)  # semilla para reproducibilidad

# (a) Cargar el conjunto 'heart' y construir la variable binaria Y
data("heart")
heart <- as.data.frame(heart)

# Regla de binarización: class == 0 -> ausencia (0), class in 1:4 -> presencia (1)
heart <- heart %>%
  mutate(Y = ifelse(class == 0, 0, 1)) %>%
  select(-class)

# Mantener versión factor para modelado y métricas
heart$Y <- factor(heart$Y, levels = c(0,1), labels = c("neg","pos"))

cat("\n(a) Binarización completada. Distribución de Y:\n")
print(prop.table(table(heart$Y)))

# (b.1) Obtener ranking mRMR para todas las variables candidatas
# Preparar copia numérica estricta requerida por mRMRe
heart_num <- heart
heart_num[] <- lapply(heart_num, function(x) {
  if (is.factor(x) || is.character(x)) as.numeric(as.character(x)) else as.numeric(x)
})
heart_num$Y <- as.numeric(as.character(ifelse(heart$Y == "pos", 1, 0)))

stopifnot(all(sapply(heart_num, is.numeric)))
stopifnot(all(heart_num$Y %in% c(0,1)))

# Número de predictores candidatos (excluyendo Y)
p <- ncol(heart_num) - 1
if (p < 1) stop("No se encontraron predictores para mRMR.")

# Ejecutar mRMR solicitando p features para obtener el ranking completo
res_mrmr_all <- mRMR.classic(
  data = mRMR.data(data = heart_num),
  target_indices = which(colnames(heart_num) == "Y"),
  feature_count = p
)

feature_idx_all <- res_mrmr_all@filters[[1]]
feature_scores_all <- NULL
if (!is.null(res_mrmr_all@scores) && length(res_mrmr_all@scores) >= 1) {
  tmp <- try(res_mrmr_all@scores[[1]], silent = TRUE)
  if (!inherits(tmp, "try-error") && length(tmp) == length(feature_idx_all)) {
    feature_scores_all <- tmp
  }
}
if (is.null(feature_scores_all)) feature_scores_all <- rep(NA_real_, length(feature_idx_all))

# Construir tabla mRMR con Variable y Score (Score redondeado a 2 decimales)
tabla_mrmr_all <- data.frame(
  Variable = colnames(heart_num)[feature_idx_all],
  Score = round(feature_scores_all, 2),
  stringsAsFactors = FALSE
)

# Ordenar por Score si están disponibles, si no mantener orden de selección
if (!all(is.na(tabla_mrmr_all$Score))) {
  tabla_mrmr_all_ord <- tabla_mrmr_all[order(-tabla_mrmr_all$Score, na.last = TRUE), , drop = FALSE]
} else {
  tabla_mrmr_all_ord <- tabla_mrmr_all
}

cat("\n(b.1) Ranking mRMR (todas las variables) - tabla Variable / Score:\n")
print(tabla_mrmr_all_ord, row.names = FALSE)

# (b.2) Selección mediante forward stepwise (AIC) y mapeo de dummies a variables originales
# Preparar datos completos (sin NA)
heart_step <- na.omit(heart)

# Eliminar predictores de varianza casi nula si existen
nzv <- nearZeroVar(heart_step, saveMetrics = TRUE)
if (any(nzv$nzv)) {
  remove_cols <- rownames(nzv)[nzv$nzv]
  message("Eliminando predictores de varianza casi nula antes de step(): ", paste(remove_cols, collapse = ", "))
  heart_step <- heart_step[, !(colnames(heart_step) %in% remove_cols), drop = FALSE]
}

# Construir fórmula completa con predictores disponibles
pred_names <- setdiff(names(heart_step), "Y")
if (length(pred_names) == 0) stop("No hay predictores disponibles para stepwise.")
full_formula <- as.formula(paste("Y ~", paste(pred_names, collapse = " + ")))

# Ajustar modelo nulo y ejecutar forward stepwise (AIC)
null_model <- glm(Y ~ 1, data = heart_step, family = binomial)
step_model_try <- try(stats::step(null_model, scope = list(upper = full_formula), direction = "forward", trace = FALSE),
                      silent = TRUE)

if (inherits(step_model_try, "try-error")) {
  warning("step() falló. vars_step y mapping estarán vacíos.")
  vars_step <- character(0)
  mapping_df <- data.frame(Coef = character(0), Term = character(0), stringsAsFactors = FALSE)
  vars_step_original <- character(0)
} else {
  step_model <- step_model_try
  coef_names <- names(coef(step_model))[-1]  # nombres de coeficientes (pueden ser dummies)
  vars_step <- if (length(coef_names) == 0) character(0) else coef_names
  
  # Construir mapeo coeficiente -> término original usando model.matrix y terms
  mm <- model.matrix(step_model)
  assign_vec <- attr(mm, "assign")
  term_labels <- attr(terms(step_model), "term.labels")
  
  mapping_rows <- data.frame(
    ColName = colnames(mm),
    Assign = assign_vec,
    stringsAsFactors = FALSE
  )
  mapping_rows <- mapping_rows[mapping_rows$Assign > 0, , drop = FALSE]
  mapping_rows$Term <- term_labels[mapping_rows$Assign]
  
  mapping_df <- mapping_rows[, c("ColName", "Term")]
  names(mapping_df) <- c("Coef", "Term")
  vars_step_original <- unique(mapping_df$Term)
  
  cat("\n(b.2) Nombres de coeficientes (dummies) seleccionados por stepwise:\n")
  print(vars_step)
  cat("\nMapeo (coeficiente -> término original):\n")
  print(mapping_df, row.names = FALSE)
  cat("\nVariables originales incluidas por stepwise (sin niveles/dummies):\n")
  print(vars_step_original)
}

# (b.3) Lista de variables por criterio de dominio (selección por conocimiento clínico)
vars_domain <- c("age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal")
vars_domain <- intersect(vars_domain, names(heart))
cat("\n(b.3) Variables por criterio de dominio (lista fija intersectada con nombres del dataset):\n")
print(vars_domain)

# Función que devuelve las variables comunes entre top-k mRMR, stepwise (términos originales) y dominio
get_common_vars <- function(k = 6, verbose = TRUE) {
  if (!exists("tabla_mrmr_all_ord")) stop("Ejecutar primero el ranking mRMR (tabla_mrmr_all_ord).")
  if (k <= 0) stop("k debe ser >= 1")
  if (k > nrow(tabla_mrmr_all_ord)) {
    warning("k > número de variables candidatas; se usará el máximo disponible.")
    k <- nrow(tabla_mrmr_all_ord)
  }
  
  topk <- head(tabla_mrmr_all_ord$Variable, k)
  step_vars_use <- if (exists("vars_step_original") && length(vars_step_original) > 0) vars_step_original else character(0)
  domain_vars_use <- if (exists("vars_domain")) vars_domain else character(0)
  
  commons <- intersect(topk, intersect(step_vars_use, domain_vars_use))
  
  if (length(commons) == 0) {
    if (verbose) message("Intersección vacía para k=", k, ". Se devolverá top-k mRMR como alternativa.")
    return(topk)
  } else {
    if (verbose) message("Encontradas ", length(commons), " variable(s) comunes para k=", k, ": ", paste(commons, collapse = ", "))
    return(commons)
  }
}

# Usar k = 6 según solicitud
k_chosen <- 6
common_vars_k6 <- get_common_vars(k = k_chosen, verbose = TRUE)
cat("\nVariables comunes a utilizar (k=", k_chosen, "):\n", sep = "")
print(common_vars_k6)

# Resumen con los objetos de selección
lista_seleccion <- list(
  mRMR_full_table = tabla_mrmr_all_ord,
  mRMR_topk_k = head(tabla_mrmr_all_ord$Variable, k_chosen),
  Stepwise_coefs = if (exists("vars_step")) vars_step else character(0),
  Stepwise_terms = if (exists("vars_step_original")) vars_step_original else character(0),
  Mapping_stepwise = if (exists("mapping_df")) mapping_df else data.frame(),
  Dominio = vars_domain,
  Comunes_k6 = common_vars_k6
)

cat("\nResumen de selección disponible (mRMR table, mapping stepwise, dominio, comunes):\n")
print(names(lista_seleccion))

# Preparar un único split estratificado (70/30) para las etapas de modelado
split_obj <- initial_split(heart, prop = 0.7, strata = "Y")
train <- training(split_obj)
test  <- testing(split_obj)

cat("\nTamaños Train / Test:", nrow(train), "/", nrow(test), "\n")
cat("Distribución Y en Train:\n"); print(prop.table(table(train$Y)))
cat("Distribución Y en Test:\n"); print(prop.table(table(test$Y)))

# Preparar matrices de diseño (dummies) y normalizar usando parámetros del conjunto de entrenamiento
predictors_c <- intersect(common_vars_k6, names(heart))
if (length(predictors_c) == 0) stop("No se seleccionaron predictores — ajustar k o métodos.")

train_sub <- train[, c(predictors_c, "Y"), drop = FALSE]
test_sub  <- test[,  c(predictors_c, "Y"), drop = FALSE]

logical_cols <- union(names(train_sub)[sapply(train_sub, is.logical)], names(test_sub)[sapply(test_sub, is.logical)])
if (length(logical_cols) > 0) {
  message("Convirtiendo columnas lógicas a factor: ", paste(logical_cols, collapse = ", "))
  train_sub[logical_cols] <- lapply(train_sub[logical_cols], function(x) as.factor(as.character(x)))
  test_sub[logical_cols]  <- lapply(test_sub[logical_cols],  function(x) as.factor(as.character(x)))
}

dv <- dummyVars(~ ., data = train_sub[, predictors_c, drop = FALSE], fullRank = FALSE)
X_train <- predict(dv, newdata = train_sub[, predictors_c, drop = FALSE])
X_test  <- predict(dv, newdata = test_sub[, predictors_c, drop = FALSE])

# Normalizar columnas según media y desviación estándar del entrenamiento
col_means <- apply(X_train, 2, mean)
col_sds   <- apply(X_train, 2, sd)
col_sds[col_sds == 0] <- 1
X_train_s <- scale(X_train, center = col_means, scale = col_sds)
X_test_s  <- scale(X_test,  center = col_means, scale = col_sds)

y_train <- train_sub$Y
y_test  <- test_sub$Y
if (!is.factor(y_train)) y_train <- factor(y_train, levels = c(0,1), labels = c("neg","pos"))
if (!is.factor(y_test))  y_test  <- factor(y_test,  levels = c(0,1), labels = c("neg","pos"))

# Función para calcular métricas a partir de predicciones y etiquetas verdaderas
calc_metrics <- function(y_true, y_pred) {
  tab <- table(Pred = as.character(y_pred), True = as.character(y_true))
  TP <- ifelse("pos" %in% rownames(tab) & "pos" %in% colnames(tab), tab["pos","pos"], 0)
  TN <- ifelse("neg" %in% rownames(tab) & "neg" %in% colnames(tab), tab["neg","neg"], 0)
  FP <- ifelse("pos" %in% rownames(tab) & "neg" %in% colnames(tab), tab["pos","neg"], 0)
  FN <- ifelse("neg" %in% rownames(tab) & "pos" %in% colnames(tab), tab["neg","pos"], 0)
  acc <- ifelse((TP+TN+FP+FN)>0, (TP+TN)/(TP+TN+FP+FN), NA)
  sens <- ifelse((TP+FN)>0, TP/(TP+FN), NA)
  spec <- ifelse((TN+FP)>0, TN/(TN+FP), NA)
  ppv  <- ifelse((TP+FP)>0, TP/(TP+FP), NA)
  npv  <- ifelse((TN+FN)>0, TN/(TN+FN), NA)
  gmean <- ifelse(!is.na(sens) & !is.na(spec), sqrt(sens * spec), NA)
  f1 <- ifelse(!is.na(ppv) & !is.na(sens) & (ppv+sens) > 0, 2 * ppv * sens / (ppv + sens), NA)
  denom <- sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  mcc <- ifelse(denom > 0, (TP*TN - FP*FN)/denom, NA)
  data.frame(Accuracy = acc, Sensitivity = sens, Specificity = spec, PPV = ppv, NPV = npv, Gmean = gmean, F1 = f1, MCC = mcc, row.names = NULL)
}

# (c) SVM lineal - margen duro (cost muy grande)
svm_hard <- svm(x = X_train_s, y = y_train, kernel = "linear", type = "C-classification", cost = 1e6, scale = FALSE)
pred_hard <- predict(svm_hard, newdata = X_test_s)
metrics_hard <- calc_metrics(y_test, pred_hard)

# (d) SVM lineal - margen blando (cost moderado)
svm_soft <- svm(x = X_train_s, y = y_train, kernel = "linear", type = "C-classification", cost = 1, scale = FALSE)
pred_soft <- predict(svm_soft, newdata = X_test_s)
metrics_soft <- calc_metrics(y_test, pred_soft)

# (e) Ajuste SVM lineal por validación cruzada sobre C (entrenado en TRAIN)
grid_C_lin <- 10^seq(-3, 3, by = 1)
cat("\n(e) Rejilla para C (lineal):\n"); print(grid_C_lin)

set.seed(123)
tune_lin <- tune.svm(x = X_train_s, y = y_train, kernel = "linear", cost = grid_C_lin, tunecontrol = tune.control(cross = 5))
cat("\nResumen ajuste SVM lineal:\n"); print(summary(tune_lin))
best_C_lin <- tune_lin$best.parameters$cost
cat("Mejor C (lineal) =", best_C_lin, "\n")

best_lin_model <- tune_lin$best.model
pred_tuned_lin <- predict(best_lin_model, newdata = X_test_s)
metrics_tuned_lin <- calc_metrics(y_test, pred_tuned_lin)

# (f) Ajuste SVM RBF por validación cruzada sobre (C, gamma) (entrenado en TRAIN)
grid_C <- 10^seq(-3, 3, by = 1)
grid_gamma <- 2^seq(-7, 0, by = 1)

cat("\n(f) Rejilla para RBF: C y gamma\n")
cat("Rejilla C:\n"); print(grid_C)
cat("Rejilla gamma:\n"); print(grid_gamma)

set.seed(123)
tune_rbf_2d <- tune.svm(x = X_train_s, y = y_train,
                        kernel = "radial",
                        cost = grid_C,
                        gamma = grid_gamma,
                        tunecontrol = tune.control(cross = 5))

cat("\nResumen ajuste RBF:\n"); print(summary(tune_rbf_2d))
best_C_rbf  <- tune_rbf_2d$best.parameters$cost
best_gamma  <- tune_rbf_2d$best.parameters$gamma
cat("\nMejores parámetros para RBF: C =", best_C_rbf, ", gamma =", best_gamma, "\n")

best_rbf_2d_model <- tune_rbf_2d$best.model
pred_rbf_2d <- predict(best_rbf_2d_model, newdata = X_test_s)
metrics_rbf_2d <- calc_metrics(y_test, pred_rbf_2d)

# Tabla comparativa final con las métricas calculadas
comparison <- rbind(
  Hard = metrics_hard,
  Soft = metrics_soft,
  TunedLinear = metrics_tuned_lin,
  RBF_Tuned2D = metrics_rbf_2d
)

cat("\n=== Tabla comparativa final (fila1=Hard, fila2=Soft, fila3=TunedLinear, fila4=RBF_Tuned2D) ===\n")
print(round(comparison, 6))

cat("\nMatriz de confusión - Hard:\n"); print(table(Pred = pred_hard, True = y_test))
cat("\nMatriz de confusión - Soft:\n"); print(table(Pred = pred_soft, True = y_test))
cat("\nMatriz de confusión - TunedLinear:\n"); print(table(Pred = pred_tuned_lin, True = y_test))
cat("\nMatriz de confusión - RBF Tuned (C,gamma):\n"); print(table(Pred = pred_rbf_2d, True = y_test))

cat("\nMejor C lineal:", best_C_lin, "\n")
cat("Mejor C RBF:", best_C_rbf, " Mejor gamma RBF:", best_gamma, "\n")

cat("\n")
cat(rep("=", 80), "\n", sep = "")
cat("FIN DEL ANÁLISIS\n")
cat(rep("=", 80), "\n", sep = "")

sink()
cat("Todos los resultados se guardaron en:", output_file, "\n")
cat("Ubicación del archivo:", normalizePath(output_file), "\n")
