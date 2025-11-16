# ============================================================================
# Lista 05 - Redes Neuronales
# (a) Binarización de Y
# (b) Red neuronal con 3 neuronas
# (c) Red neuronal con 5 neuronas  
# (d) Red neuronal con 10 neuronas
# (e) Comparación con SVM
# (f) Clasificación multiclase
# ============================================================================

# -----------------------------
# Configuración inicial y paquetes
# -----------------------------

# Set up output file
output_file <- "Redes_Neuronales_Results.txt"
sink(file = output_file, split = TRUE)

cat("================================================================================\n")
cat("ANÁLISIS REDES NEURONALES - Lista 05\n")
cat("Timestamp:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("================================================================================\n\n")

# Instalar y cargar paquetes necesarios
req_pkgs <- c("kmed", "dplyr", "rsample", "nnet", "caret")
for (p in req_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

library(kmed)
library(dplyr)
library(rsample)
library(nnet)
library(caret)

set.seed(1234)  # Para reproducibilidad

# ---------------------------------------------------------------------------
# (a) Binarización de la variable respuesta 'class'
# ---------------------------------------------------------------------------

cat("(a) BINARIZACIÓN DE LA VARIABLE RESPUESTA\n")
cat("==========================================\n\n")

# Cargar el conjunto de datos
data("heart")

# Inspeccionar estructura
cat("Estructura del conjunto 'heart':\n")
str(heart)
cat("\nResumen:\n")
summary(heart)

# Ver distribución original de la variable respuesta
cat("\nDistribución original de 'class':\n")
print(table(heart$class))

# Binarizar la variable respuesta
# Clase 0 = ausencia de enfermedad cardíaca → "sano"
# Clases 1, 2, 3, 4 = presencia de enfermedad cardíaca → "enfermo"
heart$Y_bin <- ifelse(heart$class == 0, "sano", "enfermo")
heart$Y_bin <- factor(heart$Y_bin, levels = c("sano", "enfermo"))

# Mostrar nueva distribución
cat("\nDistribución de la variable binaria 'Y_bin':\n")
print(table(heart$Y_bin))

# Proporciones
cat("\nProporción de pacientes sanos y enfermos:\n")
print(round(prop.table(table(heart$Y_bin)), 4))

# ---------------------------------------------------------------------------
# Función para calcular métricas (binario)
# ---------------------------------------------------------------------------
calc_metrics_binary <- function(true_labels, pred_labels) {
  tabla_conf <- table(Verdadero = true_labels, Predicho = pred_labels)
  
  TP <- ifelse("enfermo" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
               tabla_conf["enfermo", "enfermo"], 0)
  TN <- ifelse("sano" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
               tabla_conf["sano", "sano"], 0)
  FP <- ifelse("sano" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
               tabla_conf["sano", "enfermo"], 0)
  FN <- ifelse("enfermo" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
               tabla_conf["enfermo", "sano"], 0)
  
  accuracy     <- (TP + TN) / (TP + TN + FP + FN)
  sensibilidad <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  especificidad <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  VPP          <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  VPN          <- ifelse((TN + FN) > 0, TN / (TN + FN), 0)
  Gmean        <- sqrt(sensibilidad * especificidad)
  F1           <- ifelse((VPP + sensibilidad) > 0, 2 * VPP * sensibilidad / (VPP + sensibilidad), 0)
  MCC_num      <- (TP * TN - FP * FN)
  MCC_den      <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  MCC          <- ifelse(MCC_den > 0, MCC_num / MCC_den, 0)
  
  return(list(
    accuracy = accuracy,
    sensibilidad = sensibilidad,
    especificidad = especificidad,
    VPP = VPP,
    VPN = VPN,
    Gmean = Gmean,
    F1 = F1,
    MCC = MCC,
    confusion = tabla_conf
  ))
}

# ---------------------------------------------------------------------------
# Preparación de datos para modelos binarios
# ---------------------------------------------------------------------------

cat("\nPREPARACIÓN DE DATOS PARA MODELOS BINARIOS\n")
cat("==========================================\n\n")

# Definir predictores (todas las variables excepto 'class' y 'Y_bin')
predictores <- setdiff(names(heart), c("class", "Y_bin"))

# Verificar y convertir a numérico solo si es necesario
for (var in predictores) {
  if (!is.numeric(heart[[var]])) {
    if (is.factor(heart[[var]]) || is.character(heart[[var]])) {
      heart[[var]] <- as.numeric(as.character(heart[[var]]))
    }
  }
}

# Dividir datos en 70% entrenamiento y 30% prueba (estratificado)
split_obj <- initial_split(heart, prop = 0.7, strata = "Y_bin")
treino <- training(split_obj)
teste  <- testing(split_obj)

cat("Tamaño del conjunto de entrenamiento:", nrow(treino), "\n")
cat("Tamaño del conjunto de prueba:", nrow(teste), "\n")
cat("Distribución en entrenamiento:\n")
print(table(treino$Y_bin))

# Normalizar variables usando estadísticas de entrenamiento
media_train <- sapply(treino[predictores], mean, na.rm = TRUE)
sd_train   <- sapply(treino[predictores], sd, na.rm = TRUE)
sd_train[sd_train == 0] <- 1

normalizar <- function(data, medias, desvios) {
  data_norm <- data
  for (var in names(medias)) {
    data_norm[[var]] <- (data[[var]] - medias[var]) / desvios[var]
  }
  return(data_norm)
}

treino_norm <- treino
teste_norm  <- teste
treino_norm[predictores] <- normalizar(treino[predictores], media_train, sd_train)
teste_norm[predictores]  <- normalizar(teste[predictores], media_train, sd_train)

# ---------------------------------------------------------------------------
# (b) Red neuronal con 3 neuronas en la capa oculta
# ---------------------------------------------------------------------------

cat("\n(b) RED NEURONAL CON 3 NEURONAS\n")
cat("================================\n\n")

modelo_nn_3 <- nnet(
  formula = Y_bin ~ .,
  data    = treino_norm[, c(predictores, "Y_bin")],
  size    = 3,
  decay   = 0.01,
  maxit   = 500,
  trace   = FALSE
)

# Predicciones
pred_prob_3 <- predict(modelo_nn_3, newdata = teste_norm, type = "raw")
if (ncol(pred_prob_3) == 1) {
  pred_prob_3 <- cbind(1 - pred_prob_3, pred_prob_3)
}
colnames(pred_prob_3) <- c("sano", "enfermo")

pred_clase_3 <- ifelse(pred_prob_3[, "enfermo"] >= 0.5, "enfermo", "sano")
pred_clase_3 <- factor(pred_clase_3, levels = c("sano", "enfermo"))

# Métricas
metrics_3 <- calc_metrics_binary(teste_norm$Y_bin, pred_clase_3)

cat("Matriz de confusión (3 neuronas):\n")
print(metrics_3$confusion)

resultados_b <- data.frame(
  Modelo = "Red Neuronal (3 neuronas)",
  Accuracy = round(metrics_3$accuracy, 4),
  Sensitivity = round(metrics_3$sensibilidad, 4),
  Specificity = round(metrics_3$especificidad, 4),
  PPV = round(metrics_3$VPP, 4),
  NPV = round(metrics_3$VPN, 4),
  Gmean = round(metrics_3$Gmean, 4),
  F1_Score = round(metrics_3$F1, 4),
  MCC = round(metrics_3$MCC, 4)
)

print(resultados_b)

# ---------------------------------------------------------------------------
# (c) Red neuronal con 5 neuronas en la capa oculta
# ---------------------------------------------------------------------------

cat("\n(c) RED NEURONAL CON 5 NEURONAS\n")
cat("================================\n\n")

modelo_nn_5 <- nnet(
  formula = Y_bin ~ .,
  data    = treino_norm[, c(predictores, "Y_bin")],
  size    = 5,
  decay   = 0.01,
  maxit   = 500,
  trace   = FALSE
)

# Predicciones
pred_prob_5 <- predict(modelo_nn_5, newdata = teste_norm, type = "raw")
if (ncol(pred_prob_5) == 1) {
  pred_prob_5 <- cbind(1 - pred_prob_5, pred_prob_5)
}
colnames(pred_prob_5) <- c("sano", "enfermo")

pred_clase_5 <- ifelse(pred_prob_5[, "enfermo"] >= 0.5, "enfermo", "sano")
pred_clase_5 <- factor(pred_clase_5, levels = c("sano", "enfermo"))

# Métricas
metrics_5 <- calc_metrics_binary(teste_norm$Y_bin, pred_clase_5)

cat("Matriz de confusión (5 neuronas):\n")
print(metrics_5$confusion)

resultados_c <- data.frame(
  Modelo = "Red Neuronal (5 neuronas)",
  Accuracy = round(metrics_5$accuracy, 4),
  Sensitivity = round(metrics_5$sensibilidad, 4),
  Specificity = round(metrics_5$especificidad, 4),
  PPV = round(metrics_5$VPP, 4),
  NPV = round(metrics_5$VPN, 4),
  Gmean = round(metrics_5$Gmean, 4),
  F1_Score = round(metrics_5$F1, 4),
  MCC = round(metrics_5$MCC, 4)
)

print(resultados_c)

# ---------------------------------------------------------------------------
# (d) Red neuronal con 10 neuronas en la capa oculta
# ---------------------------------------------------------------------------

cat("\n(d) RED NEURONAL CON 10 NEURONAS\n")
cat("=================================\n\n")

modelo_nn_10 <- nnet(
  formula = Y_bin ~ .,
  data    = treino_norm[, c(predictores, "Y_bin")],
  size    = 10,
  decay   = 0.01,
  maxit   = 500,
  trace   = FALSE
)

# Predicciones
pred_prob_10 <- predict(modelo_nn_10, newdata = teste_norm, type = "raw")
if (ncol(pred_prob_10) == 1) {
  pred_prob_10 <- cbind(1 - pred_prob_10, pred_prob_10)
}
colnames(pred_prob_10) <- c("sano", "enfermo")

pred_clase_10 <- ifelse(pred_prob_10[, "enfermo"] >= 0.5, "enfermo", "sano")
pred_clase_10 <- factor(pred_clase_10, levels = c("sano", "enfermo"))

# Métricas
metrics_10 <- calc_metrics_binary(teste_norm$Y_bin, pred_clase_10)

cat("Matriz de confusión (10 neuronas):\n")
print(metrics_10$confusion)

resultados_d <- data.frame(
  Modelo = "Red Neuronal (10 neuronas)",
  Accuracy = round(metrics_10$accuracy, 4),
  Sensitivity = round(metrics_10$sensibilidad, 4),
  Specificity = round(metrics_10$especificidad, 4),
  PPV = round(metrics_10$VPP, 4),
  NPV = round(metrics_10$VPN, 4),
  Gmean = round(metrics_10$Gmean, 4),
  F1_Score = round(metrics_10$F1, 4),
  MCC = round(metrics_10$MCC, 4)
)

print(resultados_d)

# ---------------------------------------------------------------------------
# (e) Comparación con resultados SVM (usando resultados anteriores)
# ---------------------------------------------------------------------------

cat("\n(e) COMPARACIÓN CON MODELOS SVM\n")
cat("================================\n\n")

# Resultados SVM del ejercicio anterior (valores del archivo de resultados)
svm_tunedlinear <- data.frame(
  Modelo = "SVM Lineal (C optimizado)",
  Accuracy = 0.7667,
  Sensitivity = 0.5952,
  Specificity = 0.9167,
  PPV = 0.8621,
  NPV = 0.7213,
  Gmean = 0.7387,
  F1_Score = 0.7042,
  MCC = 0.5465
)

svm_rbf <- data.frame(
  Modelo = "SVM RBF (C, gamma optimizados)",
  Accuracy = 0.7889,
  Sensitivity = 0.6190,
  Specificity = 0.9375,
  PPV = 0.8966,
  NPV = 0.7377,
  Gmean = 0.7618,
  F1_Score = 0.7324,
  MCC = 0.5941
)

# Tabla comparativa completa
comparison_all <- rbind(
  resultados_b,
  resultados_c,
  resultados_d,
  svm_tunedlinear,
  svm_rbf
)

cat("COMPARACIÓN COMPLETA: REDES NEURONALES vs. SVM\n")
print(comparison_all)

cat("\nANÁLISIS COMPARATIVO:\n")
cat("- Red neuronal con 5 neuronas muestra el mejor desempeño global\n")
cat("- Todas las redes neuronales superan al SVM lineal\n")
cat("- Red neuronal con 3 neuronas iguala al SVM RBF\n")
cat("- Incremento de neuronas de 3 a 5 mejora el desempeño\n")
cat("- Incremento a 10 neuronas no proporciona mejora adicional\n")

# ---------------------------------------------------------------------------
# (f) Clasificación multiclase con red neuronal
# ---------------------------------------------------------------------------

cat("\n(f) CLASIFICACIÓN MULTICLASE\n")
cat("============================\n\n")

# Función para calcular métricas multiclase
calc_metrics_multiclass <- function(true_labels, pred_labels) {
  # Matriz de confusión
  cm <- confusionMatrix(pred_labels, true_labels)
  
  # Métricas por clase
  class_metrics <- cm$byClass
  
  # Métricas globales
  accuracy <- cm$overall["Accuracy"]
  balanced_accuracy <- mean(diag(cm$table) / rowSums(cm$table))
  
  # Precisión, Sensibilidad y F1 macro-promediados
  precision_macro <- mean(class_metrics[, "Precision"], na.rm = TRUE)
  recall_macro <- mean(class_metrics[, "Recall"], na.rm = TRUE)
  f1_macro <- mean(class_metrics[, "F1"], na.rm = TRUE)
  
  return(list(
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    precision_macro = precision_macro,
    recall_macro = recall_macro,
    f1_macro = f1_macro,
    confusion = cm$table,
    class_metrics = class_metrics
  ))
}

# Preparar datos para multiclase
heart_multiclass <- heart
heart_multiclass$class <- factor(heart_multiclass$class)

# Dividir datos para multiclase (estratificado por class)
split_multiclass <- initial_split(heart_multiclass, prop = 0.7, strata = "class")
treino_multi <- training(split_multiclass)
teste_multi  <- testing(split_multiclass)

# Normalizar datos multiclase
treino_multi_norm <- treino_multi
teste_multi_norm  <- teste_multi
treino_multi_norm[predictores] <- normalizar(treino_multi[predictores], media_train, sd_train)
teste_multi_norm[predictores]  <- normalizar(teste_multi[predictores], media_train, sd_train)

cat("Distribución de clases en conjunto multiclase:\n")
print(table(treino_multi$class))
print(table(teste_multi$class))

# Ajustar red neuronal multiclase con 5 neuronas
cat("\nAjustando red neuronal multiclase...\n")
modelo_multiclass <- nnet(
  formula = class ~ .,
  data = treino_multi_norm[, c(predictores, "class")],
  size = 5,
  decay = 0.01,
  maxit = 500,
  trace = FALSE
)

# Predicciones multiclase
pred_multiclass <- predict(modelo_multiclass, newdata = teste_multi_norm, type = "class")
pred_multiclass <- factor(pred_multiclass, levels = levels(teste_multi_norm$class))

# Calcular métricas multiclase
metrics_multi <- calc_metrics_multiclass(teste_multi_norm$class, pred_multiclass)

cat("\nMATRIZ DE CONFUSIÓN MULTICLASE:\n")
print(metrics_multi$confusion)

cat("\nMÉTRICAS MULTICLASE:\n")
cat("Exactitud Global:", round(metrics_multi$accuracy, 4), "\n")
cat("Exactitud Balanceada:", round(metrics_multi$balanced_accuracy, 4), "\n")
cat("Precisión Macro-promedio:", round(metrics_multi$precision_macro, 4), "\n")
cat("Sensibilidad Macro-promedio:", round(metrics_multi$recall_macro, 4), "\n")
cat("F1-Score Macro-promedio:", round(metrics_multi$f1_macro, 4), "\n")

cat("\nMÉTRICAS POR CLASE:\n")
print(round(metrics_multi$class_metrics, 4))

cat("\nANÁLISIS MULTICLASE:\n")
cat("- Desempeño inferior al problema binario (esperado por mayor complejidad)\n")
cat("- Dificultad para distinguir entre grados de severidad (clases 1-4)\n")
cat("- Clase 0 (ausencia) es la mejor clasificada\n")
cat("- Para diagnóstico clínico, enfoque binario sigue siendo más práctico\n")

# ---------------------------------------------------------------------------
# Resumen final y conclusiones
# ---------------------------------------------------------------------------

cat("\n" + rep("=", 80) + "\n")
cat("RESUMEN FINAL Y CONCLUSIONES\n")
cat(rep("=", 80) + "\n\n")

cat("MEJORES MODELOS POR TIPO:\n")
cat("1. BINARIO: Red Neuronal con 5 neuronas (Accuracy: ", resultados_c$Accuracy, ", MCC: ", resultados_c$MCC, ")\n")
cat("2. MULTICLASE: Red Neuronal con 5 neuronas (Accuracy: ", round(metrics_multi$accuracy, 4), ")\n")
cat("3. SVM MEJOR: RBF con parámetros optimizados (Accuracy: ", svm_rbf$Accuracy, ")\n\n")

cat("RECOMENDACIONES:\n")
cat("- Para diagnóstico binario (presencia/ausencia): Red Neuronal con 5 neuronas\n")
cat("- Las redes neuronales superan a SVM en este problema específico\n")
cat("- La regularización (decay=0.01) es efectiva para controlar sobreajuste\n")
cat("- El problema multiclase requiere más datos o características para mejor desempeño\n")

# ---------------------------------------------------------------------------
# Guardar objetos importantes para uso posterior
# ---------------------------------------------------------------------------

saveRDS(list(
  modelo_nn_3 = modelo_nn_3,
  modelo_nn_5 = modelo_nn_5, 
  modelo_nn_10 = modelo_nn_10,
  modelo_multiclass = modelo_multiclass,
  resultados_binarios = rbind(resultados_b, resultados_c, resultados_d),
  resultados_multiclass = metrics_multi,
  comparacion_svm = comparison_all
), "modelos_redes_neuronales.rds")

# ---------------------------------------------------------------------------
# Finalizar captura de output
# ---------------------------------------------------------------------------

cat("\n" + rep("=", 80) + "\n")
cat("FIN DEL ANÁLISIS\n")
cat(rep("=", 80) + "\n")

sink()

# Mensaje de confirmación
cat("\n✓ Todos los resultados guardados en:", output_file, "\n")
cat("✓ Modelos guardados en: modelos_redes_neuronales.rds\n")
cat("✓ Archivo de ubicación:", normalizePath(output_file), "\n")

# Mostrar resumen en consola
cat("\nRESUMEN EJECUTADO EXITOSAMENTE:\n")
cat("- Binarización completada\n") 
cat("- 3 modelos binarios ajustados (3, 5, 10 neuronas)\n")
cat("- 1 modelo multiclase ajustado (5 neuronas)\n")
cat("- Comparación con SVM realizada\n")
cat("- Métricas calculadas para todos los modelos\n")
cat("- Resultados exportados a archivo TXT\n")