

output_file <- "Resultados_Lista05.txt"
# abrir sink; guardamos un handler para cerrarlo al final
sink(output_file, split = TRUE)
cat("Resultados - Lista 05 (a)-(f) - usando variables Lista 04 para (e)\n\n")

# -----------------------
# Paquetes mínimos (clase)
# -----------------------
req <- c("kmed","dplyr","rsample","caret","e1071","nnet")
for(p in req) if(!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(kmed); library(dplyr); library(rsample); library(caret); library(e1071); library(nnet)

set.seed(125)

# -----------------------
# (a) Cargar 'heart' explícitamente desde kmed y binarizar Y
# -----------------------
data("heart", package = "kmed")
heart <- as.data.frame(heart)

# Convertir a character primero para evitar comparaciones erróneas con factores
if(! "class" %in% names(heart)) stop("La columna 'class' no está en el dataset 'heart' (kmed).")
class_chr <- as.character(heart[["class"]])
heart <- heart %>%
  mutate(Y = ifelse(class_chr == "0" | class_chr == "0.0" | class_chr == 0, "neg", "pos")) %>%
  select(-class)
heart$Y <- factor(heart$Y, levels = c("neg","pos"))

cat("(a) Distribución Y (dataset completo):\n")
print(table(heart$Y))
cat("\n")

# -----------------------
# Partición estratificada 70/30 (misma que Lista 04)
# -----------------------
split_obj <- initial_split(heart, prop = 0.7, strata = "Y")
train <- training(split_obj)
test  <- testing(split_obj)

cat("Tamaños train/test:", nrow(train), "/", nrow(test), "\n")
cat("Proporción train:\n"); print(prop.table(table(train$Y)))
cat("Proporción test:\n");  print(prop.table(table(test$Y)))
cat("\n")

# -----------------------
# Preprocesamiento general (dummies + normalización) PARA RN (b-d) y multiclas
# Usamos todos los predictores disponibles en el dataset para las RN
# -----------------------
predictors_all <- setdiff(names(train), "Y")
dv_all <- dummyVars(~ ., data = train[, predictors_all], fullRank = FALSE)
Xtr_all <- predict(dv_all, newdata = train[, predictors_all])
Xte_all <- predict(dv_all, newdata = test[, predictors_all])

means_all <- apply(Xtr_all, 2, mean)
sds_all   <- apply(Xtr_all, 2, sd); sds_all[sds_all == 0] <- 1
Xtr_all_s <- scale(Xtr_all, center = means_all, scale = sds_all)
Xte_all_s <- scale(Xte_all, center = means_all, scale = sds_all)

y_tr <- train$Y
y_te <- test$Y

train_df_all <- data.frame(Xtr_all_s, Y = y_tr)
test_df_all  <- data.frame(Xte_all_s,  Y = y_te)

# -----------------------
# Métricas 
# -----------------------
calc_metrics <- function(y_true, y_pred) {
  tab <- table(Pred = as.character(y_pred), True = as.character(y_true))
  TP <- ifelse("pos" %in% rownames(tab) & "pos" %in% colnames(tab), tab["pos","pos"], 0)
  TN <- ifelse("neg" %in% rownames(tab) & "neg" %in% colnames(tab), tab["neg","neg"], 0)
  FP <- ifelse("pos" %in% rownames(tab) & "neg" %in% colnames(tab), tab["pos","neg"], 0)
  FN <- ifelse("neg" %in% rownames(tab) & "pos" %in% colnames(tab), tab["neg","pos"], 0)
  
  acc  <- ifelse((TP+TN+FP+FN)>0, (TP+TN)/(TP+TN+FP+FN), NA)
  sens <- ifelse((TP+FN)>0, TP/(TP+FN), NA)
  spec <- ifelse((TN+FP)>0, TN/(TN+FP), NA)
  ppv  <- ifelse((TP+FP)>0, TP/(TP+FP), NA)
  npv  <- ifelse((TN+FN)>0, TN/(TN+FN), NA)
  gmean <- ifelse(!is.na(sens) & !is.na(spec), sqrt(sens * spec), NA)
  f1 <- ifelse(!is.na(ppv) & !is.na(sens) & (ppv+sens)>0, 2 * ppv * sens / (ppv + sens), NA)
  denom <- sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  mcc <- ifelse(denom > 0, (TP*TN - FP*FN)/denom, NA)
  data.frame(Accuracy = acc, Sensitivity = sens, Specificity = spec,
             PPV = ppv, NPV = npv, Gmean = gmean, F1 = f1, MCC = mcc)
}

# -----------------------
# MULTICLASS METRICS: función integrada (F1 macro, Recall macro, MCC multiclase)
# -----------------------
multiclass_metrics <- function(y_true, y_pred, digits = 4, out_file = NULL) {
  # asegurar factores con mismos niveles (ordenados por niveles conjuntos)
  y_true <- as.factor(y_true)
  y_pred <- as.factor(y_pred)
  lev <- union(levels(y_true), levels(y_pred))
  y_true <- factor(y_true, levels = lev)
  y_pred <- factor(y_pred, levels = lev)
  
  # matriz de confusión
  cm <- table(Pred = y_pred, True = y_true)
  
  # calcular precisión (precision), recall y F1 por clase
  K <- length(lev)
  precision <- numeric(K)
  recall <- numeric(K)
  f1 <- numeric(K)
  names(precision) <- names(recall) <- names(f1) <- lev
  
  for (i in seq_along(lev)) {
    cls <- lev[i]
    TP <- ifelse(cls %in% rownames(cm) & cls %in% colnames(cm), cm[cls, cls], 0)
    FP <- ifelse(cls %in% rownames(cm), sum(cm[cls, ]) - TP, 0)
    FN <- ifelse(cls %in% colnames(cm), sum(cm[, cls]) - TP, 0)
    precision[i] <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
    recall[i]    <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    f1[i]        <- ifelse(!is.na(precision[i]) & !is.na(recall[i]) & (precision[i] + recall[i]) > 0,
                           2 * precision[i] * recall[i] / (precision[i] + recall[i]), NA)
  }
  
  # macro-averages (ignorar NA cuando se promedia)
  precision_macro <- mean(precision, na.rm = TRUE)
  recall_macro    <- mean(recall, na.rm = TRUE)
  f1_macro        <- mean(f1, na.rm = TRUE)
  
  # MCC multiclase: usar representación one-hot y correlación de Pearson entre vectores aplanados
  n <- length(y_true)
  K <- length(lev)
  onehot_true <- matrix(0L, nrow = n, ncol = K)
  onehot_pred <- matrix(0L, nrow = n, ncol = K)
  colnames(onehot_true) <- colnames(onehot_pred) <- lev
  for (i in seq_len(n)) {
    onehot_true[i, as.character(y_true[i])] <- 1L
    onehot_pred[i, as.character(y_pred[i])] <- 1L
  }
  v_true <- as.vector(onehot_true)
  v_pred <- as.vector(onehot_pred)
  num <- cov(v_true, v_pred)
  den <- sqrt(var(v_true) * var(v_pred))
  mcc_multi <- ifelse(den == 0, NA, as.numeric(num / den))
  
  # organizar resultados
  per_class <- data.frame(
    Class = lev,
    Precision = round(precision, digits),
    Recall = round(recall, digits),
    F1 = round(f1, digits),
    stringsAsFactors = FALSE
  )
  summary_metrics <- list(
    precision_macro = round(precision_macro, digits),
    recall_macro = round(recall_macro, digits),
    f1_macro = round(f1_macro, digits),
    mcc_multiclass = round(mcc_multi, digits)
  )
  
  # imprimir resultados a consola
  cat("\n--- MATRIZ DE CONFUSIÓN (Pred \\ True) ---\n")
  print(cm)
  cat("\n--- Métricas por clase ---\n")
  print(per_class, row.names = FALSE)
  cat("\n--- Métricas resumen (macro / MCC multiclase) ---\n")
  print(summary_metrics)
  
  # añadir a archivo de resultados (append)
  if (!is.null(out_file)) {
    cat("\n\n--- MULTICLASS METRICS ---\n", file = out_file, append = TRUE)
    cat("Confusion matrix (Pred x True):\n", file = out_file, append = TRUE)
    capture.output(print(cm), file = out_file, append = TRUE)
    capture.output(print(per_class, row.names = FALSE), file = out_file, append = TRUE)
    cat("\nSummary metrics:\n", file = out_file, append = TRUE)
    capture.output(print(summary_metrics), file = out_file, append = TRUE)
    cat("\n% LaTeX: tabla de métricas multiclase\n", file = out_file, append = TRUE)
    cat("\\begin{tabular}{l c}\n", file = out_file, append = TRUE)
    cat(sprintf("Metric & Value \\\\\\\\ \n"), file = out_file, append = TRUE)
    cat(sprintf("Accuracy & %s \\\\\\\\ \n", format(round(sum(diag(cm))/sum(cm), digits))), file = out_file, append = TRUE)
    cat(sprintf("F1\\_macro & %s \\\\\\\\ \n", summary_metrics$f1_macro), file = out_file, append = TRUE)
    cat(sprintf("Recall\\_macro & %s \\\\\\\\ \n", summary_metrics$recall_macro), file = out_file, append = TRUE)
    cat(sprintf("MCC\\_multiclass & %s \\\\\\\\ \n", summary_metrics$mcc_multiclass), file = out_file, append = TRUE)
    cat("\\end{tabular}\n", file = out_file, append = TRUE)
  }
  
  invisible(list(confusion_matrix = cm, per_class = per_class, summary = summary_metrics))
}

# -----------------------
# (b),(c),(d) Entrenar NNet size = 3,5,10 (usando todas las variables como en la versión básica)
# -----------------------
fit_nnet <- function(size_hidden) {
  set.seed(125)
  form <- as.formula(paste("Y ~", paste(colnames(train_df_all)[colnames(train_df_all)!="Y"], collapse = " + ")))
  mod <- nnet(formula = form, data = train_df_all, size = size_hidden, decay = 0.01, maxit = 200, trace = FALSE)
  pred <- predict(mod, newdata = test_df_all, type = "class")
  list(model = mod, pred = pred, metrics = calc_metrics(y_te, pred))
}

cat("---- NNET size=3 ----\n")
res_n3 <- fit_nnet(3); print(round(res_n3$metrics,6)); cat("\n")
cat("---- NNET size=5 ----\n")
res_n5 <- fit_nnet(5); print(round(res_n5$metrics,6)); cat("\n")
cat("---- NNET size=10 ----\n")
res_n10 <- fit_nnet(10); print(round(res_n10$metrics,6)); cat("\n")

cat("Resumen NNet (b-d):\n")
print(round(rbind(res_n3$metrics, res_n5$metrics, res_n10$metrics),6))
cat("\n")

# -----------------------
# (e) SVM: USAR las variables seleccionadas en Lista 04
# Variables: thal, thalach, cp, oldpeak, exang, ca
# -----------------------
vars_lista04 <- c("thal","thalach","cp","oldpeak","exang","ca")
# Verificar que existan en el dataset procesado (usar nombres de heart)
vars_present <- intersect(vars_lista04, names(heart))
if(length(vars_present) < length(vars_lista04)) {
  cat("ADVERTENCIA: faltan variables de Lista 04 en el dataset procesado. Variables encontradas:\n")
  print(vars_present)
  cat("Se usará únicamente las variables presentes para ajustar los SVM.\n\n")
}

# Preparar train_sub/test_sub con estas columnas + Y (solo las presentes)
train_sub <- train[, c(vars_present, "Y"), drop = FALSE]
test_sub  <- test[,  c(vars_present, "Y"), drop = FALSE]

# Si no quedan predictores, abortar con mensaje
if(ncol(train_sub) <= 1) {
  stop("No hay predictores disponibles para SVM (vars_lista04 no presentes). Revise el dataset.")
}

dv_sub <- dummyVars(~ ., data = train_sub[, vars_present, drop = FALSE], fullRank = FALSE)
Xtr_sub <- predict(dv_sub, newdata = train_sub[, vars_present, drop = FALSE])
Xte_sub <- predict(dv_sub, newdata = test_sub[, vars_present, drop = FALSE])

means_sub <- apply(Xtr_sub, 2, mean); sds_sub <- apply(Xtr_sub, 2, sd); sds_sub[sds_sub==0] <- 1
Xtr_sub_s <- scale(Xtr_sub, center = means_sub, scale = sds_sub)
Xte_sub_s  <- scale(Xte_sub,  center = means_sub, scale = sds_sub)

y_tr_sub <- train_sub$Y
y_te_sub <- test_sub$Y

# SVM - Hard margin
svm_hard <- svm(x = Xtr_sub_s, y = y_tr_sub, kernel = "linear", type = "C-classification", cost = 1e6, scale = FALSE)
pred_hard <- predict(svm_hard, newdata = Xte_sub_s)
met_hard <- calc_metrics(y_te_sub, pred_hard)
cat("SVM Hard (usando variables Lista04):\n"); print(round(met_hard,6)); cat("\n")

# SVM - Soft margin (C=1)
svm_soft <- svm(x = Xtr_sub_s, y = y_tr_sub, kernel = "linear", type = "C-classification", cost = 1, scale = FALSE)
pred_soft <- predict(svm_soft, newdata = Xte_sub_s)
met_soft <- calc_metrics(y_te_sub, pred_soft)
cat("SVM Soft (usando variables Lista04):\n"); print(round(met_soft,6)); cat("\n")

# Tuned linear (CV over C)
gridC <- 10^seq(-3,3,1)
set.seed(125)
tune_lin <- tune.svm(x = Xtr_sub_s, y = y_tr_sub, kernel = "linear", cost = gridC, tunecontrol = tune.control(cross = 5))
bestC_lin <- tune_lin$best.parameters$cost
pred_tlin <- predict(tune_lin$best.model, newdata = Xte_sub_s)
met_tlin <- calc_metrics(y_te_sub, pred_tlin)
cat("SVM Tuned Linear (usando variables Lista04) - best C:", bestC_lin, "\n"); print(round(met_tlin,6)); cat("\n")

# Tuned RBF (CV over C,gamma)
gridC2 <- 10^seq(-3,3,1)
gridG  <- 2^seq(-7,0,1)
set.seed(125)
tune_rbf <- tune.svm(x = Xtr_sub_s, y = y_tr_sub, kernel = "radial", cost = gridC2, gamma = gridG, tunecontrol = tune.control(cross = 5))
bestC_rbf <- tune_rbf$best.parameters$cost
bestG_rbf <- tune_rbf$best.parameters$gamma
pred_rbf <- predict(tune_rbf$best.model, newdata = Xte_sub_s)
met_rbf <- calc_metrics(y_te_sub, pred_rbf)
cat("SVM Tuned RBF (usando variables Lista04) - best C:", bestC_rbf, " best gamma:", bestG_rbf, "\n"); print(round(met_rbf,6)); cat("\n")

cat("Resumen SVM (usando variables Lista04):\n")
print(round(rbind(met_hard, met_soft, met_tlin, met_rbf),6))
cat("\n")

# -----------------------
# (f) Multiclase con variable original 'class' usando nnet size=5
# -----------------------
# cargar de nuevo el heart original desde kmed para tener 'class'
data("heart", package = "kmed")
heart_m <- as.data.frame(heart)
heart_m$class <- factor(as.character(heart_m$class))

set.seed(125)
split_m <- initial_split(heart_m, prop = 0.7, strata = "class")
train_m <- training(split_m); test_m <- testing(split_m)

preds_m <- setdiff(names(train_m), "class")
dv_m <- dummyVars(~ ., data = train_m[, preds_m], fullRank = TRUE)
Xtr_m <- predict(dv_m, newdata = train_m[, preds_m])
Xte_m <- predict(dv_m, newdata = test_m[, preds_m])
means_m <- apply(Xtr_m, 2, mean); sds_m <- apply(Xtr_m, 2, sd); sds_m[sds_m==0] <- 1
Xtr_m_s <- scale(Xtr_m, center = means_m, scale = sds_m)
Xte_m_s <- scale(Xte_m, center = means_m, scale = sds_m)

train_m_df <- data.frame(Xtr_m_s, class = train_m$class)
test_m_df  <- data.frame(Xte_m_s, class = test_m$class)

set.seed(125)
nn_multi <- nnet(class ~ ., data = train_m_df, size = 5, decay = 0.01, maxit = 500, trace = FALSE)
pred_multi <- predict(nn_multi, newdata = test_m_df, type = "class")

cm_multi <- table(Pred = pred_multi, True = test_m_df$class)
cat("(f) Multiclase - matriz de confusión:\n"); print(cm_multi); cat("\n")
acc_multi <- sum(diag(cm_multi))/sum(cm_multi)
cat(sprintf("(f) Multiclase - Accuracy overall: %.4f\n", acc_multi))
cat("\n")

# calcular métricas multiclase adicionales (F1 macro, Recall macro, MCC multiclase)
mc_res <- multiclass_metrics(test_m_df$class, pred_multi, digits = 4, out_file = output_file)

# -----------------------
# cerrar sink
# -----------------------
cat("FIN DEL SCRIPT - Resultados guardados en:", output_file, "\n")
sink()
cat("Archivo generado:", normalizePath(output_file), "\n")
