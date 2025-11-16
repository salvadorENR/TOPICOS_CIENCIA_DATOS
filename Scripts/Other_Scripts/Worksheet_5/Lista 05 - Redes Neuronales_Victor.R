# ---------------------------------------------------------------------------
# (a) Binarización de la variable respuesta 'class' en el conjunto heart
#
# Objetivo:
# - Crear una variable binaria: ausencia vs. presencia de enfermedad cardíaca
# - Justificar clínicamente la elección de las dos clases
# ---------------------------------------------------------------------------

# Paso 1: Instalar y cargar el paquete necesario
# Descomentar si no está instalado
# install.packages("kmed")

library(kmed)

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

# Paso 2: Binarizar la variable respuesta
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

# Análisis final
cat("
Análisis:
La variable 'class' original tiene cinco niveles: 0 (ausencia) y 1–4 (grados crecientes de enfermedad coronaria).
Para formular un problema de clasificación binaria, se definió:
  - Clase negativa ('sano'): pacientes sin enfermedad cardíaca (class = 0).
  - Clase positiva ('enfermo'): pacientes con cualquier grado de enfermedad (class ≥ 1).

Esta binarización es clínicamente adecuada porque permite predecir la **presencia o ausencia** de enfermedad,
un objetivo diagnóstico fundamental. Además, evita asumir diferencias ordinales fuertes entre grados de severidad.

El conjunto resultante tiene ", sum(heart$Y_bin == "sano"), " pacientes sanos y ",
    sum(heart$Y_bin == "enfermo"), " pacientes enfermos, lo cual será útil para entrenamiento y prueba.
")


# ---------------------------------------------------------------------------
# (b) Red neuronal con 3 neuronas en la capa oculta
#
# Objetivo:
# - Ajustar una red neuronal (MLP) con nnet() usando 3 neuronas ocultas
# - Evaluar desempeño en prueba con métricas completas
# - Registrar resultados en la primera fila de la tabla
# ---------------------------------------------------------------------------

# Cargar paquetes necesarios
library(rsample)
library(nnet)

# Asumimos que 'heart' ya está cargado y Y_bin ya fue creado en (a)
if (!exists("Y_bin") || !"Y_bin" %in% names(heart)) {
  heart$Y_bin <- factor(ifelse(heart$class == 0, "sano", "enfermo"),
                        levels = c("sano", "enfermo"))
}

# Paso 1: Definir predictores (todas las variables excepto 'class' y 'Y_bin')
predictores <- setdiff(names(heart), c("class", "Y_bin"))

# Verificar y convertir a numérico solo si es necesario
for (var in predictores) {
  if (!is.numeric(heart[[var]])) {
    # Si es factor o character, convertir a numérico conservando valores
    if (is.factor(heart[[var]]) || is.character(heart[[var]])) {
      # Convertir sin alterar los niveles originales
      heart[[var]] <- as.numeric(as.character(heart[[var]]))
    }
  }
}

# Paso 2: Dividir datos en 70% entrenamiento y 30% prueba (estratificado)
set.seed(1234)
split_obj <- initial_split(
  heart,
  prop = 0.7,
  strata = "Y_bin"
)
treino <- training(split_obj)
teste  <- testing(split_obj)

# Confirmar que ambas clases están en entrenamiento
cat("Distribución de Y_bin en entrenamiento:\n")
print(table(treino$Y_bin))

if (any(table(treino$Y_bin) == 0)) {
  stop("Una clase está vacía en entrenamiento. Revisa la estratificación.")
}

# Paso 3: Normalizar solo variables numéricas usando estadísticas de entrenamiento
media_train <- sapply(treino[predictores], mean, na.rm = TRUE)
sd_train   <- sapply(treino[predictores], sd,   na.rm = TRUE)
sd_train[sd_train == 0] <- 1  # evitar división por cero

# Función de normalización
normalizar <- function(data, medias, desvios) {
  data_norm <- data
  for (var in names(medias)) {
    data_norm[[var]] <- (data[[var]] - medias[var]) / desvios[var]
  }
  return(data_norm)
}

# Aplicar normalización
treino_norm <- treino
teste_norm  <- teste

treino_norm[predictores] <- normalizar(treino[predictores], media_train, sd_train)
teste_norm[predictores]  <- normalizar(teste[predictores],  media_train, sd_train)

# Asegurar que Y_bin sea factor
treino_norm$Y_bin <- factor(treino_norm$Y_bin, levels = c("sano", "enfermo"))
teste_norm$Y_bin  <- factor(teste_norm$Y_bin,  levels = c("sano", "enfermo"))

# Paso 4: Ajustar red neuronal con 3 neuronas en la capa oculta
cat("\n=== Ajustando red neuronal con 3 neuronas ocultas ===\n")

modelo_nn_3 <- tryCatch({
  nnet(
    formula = as.factor(Y_bin) ~ .,
    data    = treino_norm[, c(predictores, "Y_bin")],
    size    = 3,           # número de neuronas en la capa oculta
    decay   = 0.01,        # regularización L2
    maxit   = 500,         # máximo de iteraciones
    trace   = FALSE,       # no mostrar progreso
    seed    = 1234         # reproducibilidad
  )
}, error = function(e) {
  cat("Error al ajustar el modelo nnet:\n")
  cat(e$message, "\n")
  return(NULL)
})

# Verificar si el modelo se ajustó
if (is.null(modelo_nn_3)) {
  stop("No se pudo ajustar la red neuronal con 3 neuronas.")
}

# Paso 5: Predicciones en conjunto de prueba
pred_prob <- tryCatch({
  predict(modelo_nn_3, newdata = teste_norm, type = "raw")
}, error = function(e) {
  cat("Error al predecir:\n")
  cat(e$message, "\n")
  return(NULL)
})

if (is.null(pred_prob)) {
  stop("No se pudieron generar predicciones para el modelo con 3 neuronas.")
}

# Manejar salida de predict: asegurar dos columnas
if (is.vector(pred_prob)) {
  pred_prob <- matrix(pred_prob, ncol = 2)
}
if (ncol(pred_prob) == 1) {
  pred_prob <- cbind(1 - pred_prob, pred_prob)
}
colnames(pred_prob) <- c("sano", "enfermo")

# Clasificar según probabilidad de clase positiva ("enfermo")
prob_enfermo <- pred_prob[, "enfermo"]
pred_clase <- ifelse(prob_enfermo >= 0.5, "enfermo", "sano")
pred_clase <- factor(pred_clase, levels = c("sano", "enfermo"))

# Matriz de confusión
tabla_conf <- table(
  Verdadero = teste_norm$Y_bin,
  Predicho = pred_clase
)
cat("\nMatriz de confusión (prueba):\n")
print(tabla_conf)

# Extraer TP, TN, FP, FN
TP <- ifelse("enfermo" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "enfermo"], 0)
TN <- ifelse("sano" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["sano", "sano"], 0)
FP <- ifelse("sano" %in% rownames(tabla_conf) & "enfermo" %in% colnames(tabla_conf), 
             tabla_conf["sano", "enfermo"], 0)
FN <- ifelse("enfermo" %in% rownames(tabla_conf) & "sano" %in% colnames(tabla_conf), 
             tabla_conf["enfermo", "sano"], 0)

# Calcular métricas
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

# Mostrar primera fila de la tabla de resultados
resultados_b <- data.frame(
  Modelo = "Red Neuronal (3 neuronas)",
  Accuracy = round(accuracy, 4),
  Sensibilidad = round(sensibilidad, 4),
  Especificidad = round(especificidad, 4),
  VPP = round(VPP, 4),
  VPN = round(VPN, 4),
  G_mean = round(Gmean, 4),
  F1_Score = round(F1, 4),
  MCC = round(MCC, 4)
)

cat("\nResultados (primera fila de la tabla):\n")
print(resultados_b)

# Análisis
cat("
Análisis del modelo de red neuronal con 3 neuronas:
El modelo logró una exactitud de ", round(accuracy, 4), " en el conjunto de prueba.
La sensibilidad de ", round(sensibilidad, 4), " indica su capacidad para detectar pacientes enfermos.
El valor predictivo positivo (VPP) fue de ", round(VPP, 4), ".
El F1-score de ", round(F1, 4), " refleja un equilibrio entre precisión y exhaustividad.
El coeficiente MCC de ", round(MCC, 4), " confirma un desempeño global razonable.
A pesar de su simplicidad, la red aprovecha relaciones no lineales entre los predictores.
")

# ---------------------------------------------------------------------------
# (c) Red neuronal con 5 neuronas en la capa oculta
#
# Objetivo:
# - Ajustar red neuronal con nnet() usando 5 neuronas ocultas
# - Usar misma partición y preprocesamiento del ítem (b)
# - Evaluar desempeño en prueba con métricas completas
# - Registrar resultados en la segunda fila de la tabla
# ---------------------------------------------------------------------------

cat("\n=== Ajustando red neuronal con 5 neuronas ocultas ===\n")

# Ajustar modelo con 5 neuronas en la capa oculta
modelo_nn_5 <- tryCatch({
  nnet(
    formula = as.factor(Y_bin) ~ .,
    data    = treino_norm[, c(predictores, "Y_bin")],
    size    = 5,           # 5 neuronas ocultas
    decay   = 0.01,        # regularización L2
    maxit   = 500,         # máximo de iteraciones
    trace   = FALSE,       # no mostrar progreso
    seed    = 1234         # reproducibilidad
  )
}, error = function(e) {
  cat("Error al ajustar el modelo:\n")
  cat(e$message, "\n")
  return(NULL)
})

if (is.null(modelo_nn_5)) {
  stop("No se pudo ajustar la red neuronal con 5 neuronas.")
}

# Predicciones en conjunto de prueba
pred_prob_5 <- tryCatch({
  predict(modelo_nn_5, newdata = teste_norm, type = "raw")
}, error = function(e) {
  cat("Error al predecir:\n")
  cat(e$message, "\n")
  return(NULL)
})

if (is.null(pred_prob_5)) {
  stop("No se pudieron generar predicciones para el modelo con 5 neuronas.")
}

# Manejar salida de predict: asegurar dos columnas
if (is.vector(pred_prob_5)) {
  pred_prob_5 <- matrix(pred_prob_5, ncol = 2)
}
if (ncol(pred_prob_5) == 1) {
  pred_prob_5 <- cbind(1 - pred_prob_5, pred_prob_5)
}
colnames(pred_prob_5) <- c("sano", "enfermo")

# Clasificar según umbral 0.5
pred_clase_5 <- ifelse(pred_prob_5[, "enfermo"] >= 0.5, "enfermo", "sano")
pred_clase_5 <- factor(pred_clase_5, levels = c("sano", "enfermo"))

# Matriz de confusión
tabla_conf_5 <- table(
  Verdadero = teste_norm$Y_bin,
  Predicho = pred_clase_5
)
print(tabla_conf_5)

# Extraer TP, TN, FP, FN
TP <- ifelse("enfermo" %in% rownames(tabla_conf_5) & "enfermo" %in% colnames(tabla_conf_5), 
             tabla_conf_5["enfermo", "enfermo"], 0)
TN <- ifelse("sano" %in% rownames(tabla_conf_5) & "sano" %in% colnames(tabla_conf_5), 
             tabla_conf_5["sano", "sano"], 0)
FP <- ifelse("sano" %in% rownames(tabla_conf_5) & "enfermo" %in% colnames(tabla_conf_5), 
             tabla_conf_5["sano", "enfermo"], 0)
FN <- ifelse("enfermo" %in% rownames(tabla_conf_5) & "sano" %in% colnames(tabla_conf_5), 
             tabla_conf_5["enfermo", "sano"], 0)

# Calcular métricas
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

# Mostrar segunda fila de la tabla de resultados
resultados_c <- data.frame(
  Modelo = "Red Neuronal (5 neuronas)",
  Accuracy = round(accuracy, 4),
  Sensibilidad = round(sensibilidad, 4),
  Especificidad = round(especificidad, 4),
  VPP = round(VPP, 4),
  VPN = round(VPN, 4),
  G_mean = round(Gmean, 4),
  F1_Score = round(F1, 4),
  MCC = round(MCC, 4)
)

print(resultados_c)

# Análisis breve
cat("
Análisis del modelo con 5 neuronas:
Al aumentar el número de neuronas a 5, la capacidad del modelo para aprender patrones no lineales mejora.
Su desempeño debe compararse directamente con el modelo de 3 neuronas para evaluar si esta complejidad adicional reduce el error en el conjunto de prueba o introduce sobreajuste.
")

# ---------------------------------------------------------------------------
# (d) Red neuronal con 10 neuronas en la capa oculta
#
# Objetivo:
# - Ajustar red neuronal con nnet() usando 10 neuronas ocultas
# - Usar misma partición y preprocesamiento del ítem (b)
# - Evaluar desempeño en prueba con métricas completas
# - Registrar resultados en la tercera fila de la tabla
# ---------------------------------------------------------------------------


cat("\n=== Ajustando red neuronal con 10 neuronas ocultas ===\n")

# Ajustar modelo con 10 neuronas en la capa oculta
modelo_nn_10 <- tryCatch({
  nnet(
    formula = as.factor(Y_bin) ~ .,
    data    = treino_norm[, c(predictores, "Y_bin")],
    size    = 10,           # 10 neuronas ocultas
    decay   = 0.01,         # regularización L2
    maxit   = 500,          # máximo de iteraciones
    trace   = FALSE,        # no mostrar progreso
    seed    = 1234          # reproducibilidad
  )
}, error = function(e) {
  cat("Error al ajustar el modelo:\n")
  cat(e$message, "\n")
  return(NULL)
})

if (is.null(modelo_nn_10)) {
  stop("No se pudo ajustar la red neuronal con 10 neuronas.")
}

# Predicciones en conjunto de prueba
pred_prob_10 <- tryCatch({
  predict(modelo_nn_10, newdata = teste_norm, type = "raw")
}, error = function(e) {
  cat("Error al predecir:\n")
  cat(e$message, "\n")
  return(NULL)
})

if (is.null(pred_prob_10)) {
  stop("No se pudieron generar predicciones para el modelo con 10 neuronas.")
}

# Manejar salida de predict
if (is.vector(pred_prob_10)) {
  pred_prob_10 <- matrix(pred_prob_10, ncol = 2)
}
if (ncol(pred_prob_10) == 1) {
  pred_prob_10 <- cbind(1 - pred_prob_10, pred_prob_10)
}
colnames(pred_prob_10) <- c("sano", "enfermo")

# Clasificar según probabilidad de clase positiva ("enfermo")
pred_clase_10 <- ifelse(pred_prob_10[, "enfermo"] >= 0.5, "enfermo", "sano")
pred_clase_10 <- factor(pred_clase_10, levels = c("sano", "enfermo"))

# Matriz de confusión
tabla_conf_10 <- table(
  Verdadero = teste_norm$Y_bin,
  Predicho = pred_clase_10
)
print(tabla_conf_10)

# Extraer TP, TN, FP, FN
TP <- ifelse("enfermo" %in% rownames(tabla_conf_10) & "enfermo" %in% colnames(tabla_conf_10), 
             tabla_conf_10["enfermo", "enfermo"], 0)
TN <- ifelse("sano" %in% rownames(tabla_conf_10) & "sano" %in% colnames(tabla_conf_10), 
             tabla_conf_10["sano", "sano"], 0)
FP <- ifelse("sano" %in% rownames(tabla_conf_10) & "enfermo" %in% colnames(tabla_conf_10), 
             tabla_conf_10["sano", "enfermo"], 0)
FN <- ifelse("enfermo" %in% rownames(tabla_conf_10) & "sano" %in% colnames(tabla_conf_10), 
             tabla_conf_10["enfermo", "sano"], 0)

# Calcular métricas
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

# Mostrar tercera fila de la tabla de resultados
resultados_d <- data.frame(
  Modelo = "Red Neuronal (10 neuronas)",
  Accuracy = round(accuracy, 4),
  Sensibilidad = round(sensibilidad, 4),
  Especificidad = round(especificidad, 4),
  VPP = round(VPP, 4),
  VPN = round(VPN, 4),
  G_mean = round(Gmean, 4),
  F1_Score = round(F1, 4),
  MCC = round(MCC, 4)
)

print(resultados_d)

# Análisis breve
cat("
Análisis del modelo con 10 neuronas:
Al aumentar el número de neuronas a 10, la capacidad del modelo para capturar patrones complejos aumenta.
Sin embargo, también crece el riesgo de sobreajuste si no se controla adecuadamente.
Su desempeño debe compararse con los modelos más simples (3 y 5 neuronas) para determinar si la mejora en entrenamiento se traduce en mejor generalización en prueba.
")
